import html as htmllib
import json
import os
import random
import re
import sys
import time
from urllib.parse import quote

import requests

# ------------------------------------------------------------------ config
CATEGORIES = [
    "Finance", "Health", "News", "Education", "History", "Crime",
    "Paranormal", "Science Experiment", "Coding", "Question Solve",
    "Trending Now", "New Song", "Mathematics", "Biology", "New AI Update",
    "Gaming News", "Treating Tips", "Market News",
]

LIVE = {
    "News", "Trending Now", "New Song", "New AI Update",
    "Gaming News", "Market News", "Finance", "Crime",
}

STYLE = {
    "Finance": "in-depth personal-finance or economy analysis with real examples, numbers and practical steps",
    "Health": "evidence-based wellness deep dive; explain mechanisms, research findings and limits; no miracle claims",
    "News": "neutral, well-sourced analysis of a news story: background, what happened, why it matters, what comes next",
    "Education": "practical study, learning or career guide with frameworks, steps and examples",
    "History": "rich narrative analysis of a real historical event or person: causes, key moments with dates, consequences, legacy",
    "Crime": "factual true-crime or crime-prevention analysis based only on publicly reported facts; never accuse private individuals",
    "Paranormal": "storytelling deep dive into a famous legend or reported phenomenon; present as folklore/reports and include scientific explanations",
    "Science Experiment": "safe hands-on experiment: materials, step-by-step method, safety notes, the science behind it, variations",
    "Coding": "detailed tutorial with working code in <pre><code> blocks, explanations of each step, common mistakes, best practices",
    "Question Solve": "take one popular exam, school or logic question and solve it step by step, explain the concept, show alternate methods and common errors",
    "Trending Now": "analysis of a topic trending online: origin, why it spread, what people say, what it means",
    "New Song": "spotlight/review of a song: artist background, sound and production, lyrical themes (describe, never quote lyrics), reception",
    "Mathematics": "concept taught from basics to advanced with worked examples, tricks, and practice problems with answers",
    "Biology": "clear, deep explainer with mechanisms, real-world examples, and notable discoveries",
    "New AI Update": "analysis of an AI tool/model/product: what it does, how it works, who benefits, limits, alternatives",
    "Gaming News": "news and analysis of a game release, update or industry event with context and community reaction",
    "Treating Tips": "practical home-care and everyday-problem tips with reasons behind each; advise seeing a professional for medical issues",
    "Market News": "stock, crypto or commodity market analysis: drivers, key figures, sector impact, risks; not financial advice",
}

DISCLAIMER_CATS = {"Health", "Finance", "Market News", "Treating Tips", "Crime"}

KEYS = {
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "cohere": "COHERE_API_KEY",
    "meta": "LLAMA_API_KEY",
}
ORDER = [p.strip() for p in os.getenv("PROVIDER_ORDER", "gemini,openrouter,cohere,meta").split(",") if p.strip() in KEYS]

STATE_FILE = "state.json"
SEP = "[[[SECTION]]]"
DEAD = set()
STOP = {"with", "that", "this", "from", "your", "what", "best", "guide", "complete",
        "ultimate", "about", "explained", "everything", "need", "know", "2025", "2026"}

VOICE = """Writing rules (apply strictly):
- Write like a seasoned specialist journalist: confident, natural, conversational yet authoritative. Vary sentence length. Use concrete examples, comparisons, numbers, cause-and-effect analysis and "why this matters" insight.
- Go deep: explain background, mechanics, different viewpoints, risks and practical takeaways. No fluff, no repetition, no generic filler, no cliches such as "in today's fast-paced world", "delve", "unlock", "game-changer".
- Never mention being an AI. Never invent quotes, statistics, studies or sources. If unsure of a fact, omit it or say it is reported/estimated.
- Short paragraphs (2-4 sentences). Clean HTML only: <p>, <h3>, <ul>/<ol>/<li>, <strong>, <table> (only if a comparison helps), <pre><code> for code. No <h1>/<h2>, no markdown, no inline styles."""


# ------------------------------------------------------------------ helpers
def env(name):
    v = os.getenv(name)
    if not v:
        sys.exit(f"Missing environment variable: {name}")
    return v


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"next": 0, "titles": {}}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def esc(s):
    return htmllib.escape(str(s), quote=True)


def words_of(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 3 and w not in STOP}


def too_similar(text, used):
    a = words_of(text)
    if not a:
        return False
    for u in used:
        b = words_of(u)
        if b and len(a & b) / len(a | b) >= 0.5:
            return True
    return False


def clean_html(t):
    t = t.strip()
    t = re.sub(r"^```(?:html)?\s*|\s*```$", "", t).strip()
    t = re.sub(r"^\s*<h[1-2][^>]*>.*?</h[1-2]>", "", t, flags=re.S)
    t = re.sub(r"^\s*#{1,3} [^\n]*\n", "", t)
    t = re.sub(r"^###\s+(.+)$", r"<h3>\1</h3>", t, flags=re.M)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    if "<p" not in t:
        t = "".join(f"<p>{p.strip()}</p>" for p in re.split(r"\n\s*\n", t) if p.strip())
    return t.strip()


def text_len(html):
    return len(re.sub(r"<[^>]+>", " ", html).split())


# ------------------------------------------------------------------ AI providers
def call_gemini(prompt, max_tokens, temp, search):
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temp, "maxOutputTokens": max_tokens},
    }
    if "flash" in model:
        payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
    if search:
        payload["tools"] = [{"google_search": {}}]
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": env("GEMINI_API_KEY"), "Content-Type": "application/json"},
        json=payload, timeout=240,
    )
    r.raise_for_status()
    cand = r.json()["candidates"][0]
    text = "".join(p.get("text", "") for p in cand["content"]["parts"]).strip()
    sources = []
    for c in cand.get("groundingMetadata", {}).get("groundingChunks", []):
        w = c.get("web")
        if w and w.get("uri"):
            sources.append((w.get("title", "Source"), w["uri"]))
    return text, sources


def _openai_style(url, key, model, prompt, max_tokens, temp, extra=None):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    headers.update(extra or {})
    r = requests.post(
        url, headers=headers, timeout=240,
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": max_tokens, "temperature": temp},
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or "", []


def call_openrouter(prompt, max_tokens, temp, search):
    return _openai_style(
        "https://openrouter.ai/api/v1/chat/completions", env("OPENROUTER_API_KEY"),
        os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
        prompt, max_tokens, temp, {"X-Title": "ai-blogger-autopilot"},
    )


def call_meta(prompt, max_tokens, temp, search):
    return _openai_style(
        "https://api.llama.com/compat/v1/chat/completions", env("LLAMA_API_KEY"),
        os.getenv("LLAMA_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8"),
        prompt, max_tokens, temp,
    )


def call_cohere(prompt, max_tokens, temp, search):
    r = requests.post(
        "https://api.cohere.com/v2/chat",
        headers={"Authorization": f"Bearer {env('COHERE_API_KEY')}", "Content-Type": "application/json"},
        json={"model": os.getenv("COHERE_MODEL", "command-a-03-2025"),
              "messages": [{"role": "user", "content": prompt}],
              "max_tokens": min(max_tokens, 4000), "temperature": temp},
        timeout=240,
    )
    r.raise_for_status()
    parts = r.json()["message"]["content"]
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text"), []


CALLERS = {"gemini": call_gemini, "openrouter": call_openrouter, "cohere": call_cohere, "meta": call_meta}


def avail(skip=()):
    return [n for n in ORDER if os.getenv(KEYS[n]) and n not in DEAD and n not in skip]


def ask(prompt, max_tokens=4000, temp=0.75, search=False, skip=()):
    names = [n for n in avail(skip) if (n == "gemini" or not search)]
    if not names:
        raise RuntimeError("No AI provider available")
    errors = []
    for name in names:
        for attempt in range(2):
            try:
                text, src = CALLERS[name](prompt, max_tokens, temp, search)
                text = (text or "").strip()
                if not text:
                    raise RuntimeError("empty reply")
                time.sleep(3)
                return text, src, name
            except Exception as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                msg = f"{name}: {code or ''} {str(e)[:120]}"
                errors.append(msg)
                print("  provider error ->", msg)
                if code in (401, 403, 404):
                    DEAD.add(name)
                    break
                if attempt == 0:
                    time.sleep(20 if code == 429 else 8)
    raise RuntimeError("All providers failed: " + " | ".join(errors))


def parse_json(text):
    t = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"[\{\[]", t)
    if not m:
        raise ValueError("no json")
    end = max(t.rfind("}"), t.rfind("]"))
    return json.loads(t[m.start():end + 1])


def ask_json(prompt, check, max_tokens=4000, temp=0.7):
    skip = set()
    for _ in range(5):
        if not avail(skip):
            skip.clear()
        text, _, name = ask(
            prompt + "\n\nReturn ONLY valid JSON. No markdown fences, no commentary.",
            max_tokens, temp, skip=skip,
        )
        try:
            data = parse_json(text)
            if check(data):
                return data
            print("  invalid structure from", name)
        except Exception as e:
            print("  bad JSON from", name, str(e)[:80])
        skip.add(name)
    raise RuntimeError("Could not get valid JSON from any provider")
    # ------------------------------------------------------------------ Blogger
def access_token():
    r = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": env("GOOGLE_CLIENT_ID"),
            "client_secret": env("GOOGLE_CLIENT_SECRET"),
            "refresh_token": env("GOOGLE_REFRESH_TOKEN"),
            "grant_type": "refresh_token",
        },
        timeout=60,
    )
    if r.status_code != 200:
        sys.exit(f"OAuth refresh failed: {r.text}")
    return r.json()["access_token"]


def blog_get(token, params):
    try:
        r = requests.get(
            f"https://www.googleapis.com/blogger/v3/blogs/{env('BLOGGER_BLOG_ID')}/posts",
            params=params, headers={"Authorization": f"Bearer {token}"}, timeout=60,
        )
        return r.json().get("items", []) if r.status_code == 200 else []
    except Exception:
        return []


def publish(outline, content, category, token):
    labels = [category] + [t for t in outline["tags"] if str(t).lower() != category.lower()][:4]
    body = {"kind": "blogger#post", "title": outline["title"], "content": content, "labels": labels}
    if os.getenv("DRY_RUN") == "1":
        print(json.dumps(body, indent=2, ensure_ascii=False)[:5000])
        return "dry-run"
    last = ""
    for attempt in range(3):
        r = requests.post(
            f"https://www.googleapis.com/blogger/v3/blogs/{env('BLOGGER_BLOG_ID')}/posts?isDraft=false",
            headers={"Authorization": f"Bearer {token}"}, json=body, timeout=120,
        )
        if r.status_code in (200, 201):
            url = r.json().get("url", "")
            print("Published:", url)
            return url
        last = f"{r.status_code} {r.text[:300]}"
        time.sleep(10)
    sys.exit(f"Blogger publish failed: {last}")


# ------------------------------------------------------------------ content steps
def pick_topic(category, used):
    rejected, live_ok = [], bool(os.getenv("GEMINI_API_KEY"))
    topic = ""
    for _ in range(6):
        avoid = "\n".join(f"- {t}" for t in (used[-80:] + rejected)) or "(none yet)"
        search = live_ok and category in LIVE
        mode = ("Use Google Search to find a genuinely current topic from the last few days."
                if search else
                "Pick a specific evergreen topic. Do not claim anything is 'latest', 'new' or 'today'.")
        prompt = f"""You are the chief editor of a high-quality English blog. Category: "{category}".
Today is {time.strftime('%B %d, %Y', time.gmtime())}.
Suggest ONE fresh, specific, search-friendly article topic with real reader demand and low-to-medium competition.
{mode}
It must be clearly different in subject from every earlier article below:
{avoid}
Reply with only the topic in one line, no quotes, no extra text."""
        try:
            text, _, _ = ask(prompt, max_tokens=300, temp=0.95, search=search)
        except Exception as e:
            if search:
                print("  search topic failed, going evergreen:", str(e)[:100])
                live_ok = False
                continue
            raise
        lines = [l for l in text.splitlines() if l.strip()]
        topic = lines[0].strip(" \"'*#-") if lines else ""
        if topic and not too_similar(topic, used):
            return topic, live_ok
        rejected.append(topic)
        print("  topic too similar, retrying:", topic)
    if topic:
        return topic, live_ok
    raise RuntimeError("Could not pick a topic")


def get_research(topic):
    try:
        text, src, _ = ask(
            f"Using Google Search, gather the latest accurate facts, figures, dates, context and differing viewpoints about: {topic}. "
            "Return detailed bullet notes. Include only verified information.",
            max_tokens=2500, temp=0.3, search=True,
        )
        return text, src
    except Exception as e:
        print("  research unavailable:", str(e)[:100])
        return "", []


def ok_outline(d):
    keys = ("title", "meta_description", "primary_keyword", "tags", "image_prompt", "image_alt", "sections")
    return (isinstance(d, dict) and all(k in d for k in keys) and isinstance(d["sections"], list)
            and len(d["sections"]) >= 5 and isinstance(d["tags"], list)
            and all(isinstance(s, dict) and "heading" in s and "focus" in s for s in d["sections"]))


def make_outline(category, topic, research, live_ok):
    prompt = f"""Plan a comprehensive, best-in-class blog article.
Category: {category}
Topic: {topic}
Style: {STYLE[category]}
{"Research notes:\n" + research if research else ""}
{"" if live_ok or category not in LIVE else "Keep the angle evergreen; do not assert any recent event as fact."}

Return a JSON object with exactly these keys:
- "title": under 60 characters, primary keyword near the start, compelling, not clickbait.
- "meta_description": 140-155 characters, includes the primary keyword, promises clear value.
- "primary_keyword": 2-4 words.
- "tags": array of 3-5 short keyword tags (do not include the category name).
- "image_prompt": vivid description of a clean illustrative featured image, no text/logos, no real people's faces.
- "image_alt": SEO alt text for it.
- "sections": array of 6 to 7 objects {{"heading": "...", "focus": "what exactly to cover"}} that together cover the topic completely in logical order. Headings must be search-friendly (use the primary keyword or close variants in at least two). The last section must be practical (how to apply this / what to do next)."""
    o = ask_json(prompt, ok_outline, max_tokens=3000)
    o["sections"] = o["sections"][:7]
    o["title"] = str(o["title"]).strip().strip('"')
    md = str(o["meta_description"]).strip()
    o["meta_description"] = md if len(md) <= 160 else md[:157].rsplit(" ", 1)[0] + "..."
    return o


def write_intro(category, topic, o, research):
    heads = "; ".join(s["heading"] for s in o["sections"])
    prompt = f"""{VOICE}

Write ONLY the introduction (180-250 words, 2-4 short paragraphs) for an article titled "{o['title']}".
Category: {category}. Topic: {topic}. Primary keyword: {o['primary_keyword']} (use it naturally in the first paragraph).
The article covers: {heads}.
Open with a hook showing why the reader should care, state what they will learn, do not summarise every section.
{"Use these facts if helpful:\n" + research if research else ""}
Return only HTML paragraphs."""
    text, _, _ = ask(prompt, max_tokens=1500)
    return clean_html(text)


def write_sections(category, topic, o, research):
    secs = o["sections"]
    structure = "\n".join(f"{i+1}. {s['heading']}" for i, s in enumerate(secs))
    out = []
    for i in range(0, len(secs), 3):
        group = secs[i:i + 3]
        listing = "\n".join(f"### {s['heading']} :: {s['focus']}" for s in group)
        prompt = f"""{VOICE}

You are writing part of the article "{o['title']}" (category: {category}, topic: {topic}).
Full article structure:
{structure}

Write the body for ONLY these {len(group)} section(s), in this order:
{listing}

For each section write 350-500 words of substantive, analytical content (explain, analyse, give examples and context). Do NOT write the section heading itself. Do not repeat points from other sections. Use <h3> sub-headings where they help.
Put the exact line {SEP} between sections (not before the first, not after the last).
{"Rely on these facts:\n" + research if research else ""}"""
        skip, got = set(), None
        for _ in range(4):
            if not avail(skip):
                skip.clear()
            text, _, name = ask(prompt, max_tokens=6000, temp=0.75, skip=skip)
            parts = [clean_html(p) for p in text.split(SEP) if p.strip()]
            if len(parts) == len(group) and all(text_len(p) >= 200 for p in parts):
                got = parts
                break
            print("  weak/incorrect sections from", name, "->", [text_len(p) for p in parts])
            skip.add(name)
        if not got:
            raise RuntimeError("Sections failed")
        out.extend(got)
    return out


def ok_closing(d):
    return (isinstance(d, dict) and isinstance(d.get("key_takeaways"), list) and len(d["key_takeaways"]) >= 3
            and isinstance(d.get("faqs"), list) and len(d["faqs"]) >= 3
            and all(isinstance(f, dict) and "q" in f and "a" in f for f in d["faqs"])
            and isinstance(d.get("conclusion_html"), str))


def write_closing(category, topic, o):
    prompt = f"""{VOICE}

Article: "{o['title']}" (category: {category}, topic: {topic}).
Sections covered: {'; '.join(s['heading'] for s in o['sections'])}.
Return a JSON object with:
- "key_takeaways": array of 5-6 specific, useful plain-text bullet points.
- "faqs": array of 4 to 5 objects {{"q": "...", "a": "..."}}: real questions people search about this topic, each answer 50-90 words, plain text.
- "conclusion_html": 120-180 words in <p> tags, uses the primary keyword "{o['primary_keyword']}" once, ends with a clear natural next step for the reader."""
    return ask_json(prompt, ok_closing, max_tokens=3500)


def get_image(prompt):
    url = (f"https://image.pollinations.ai/prompt/{quote(str(prompt)[:300])}"
           f"?width=1200&height=630&nologo=true&seed={random.randint(1, 999999)}")
    try:
        r = requests.get(url, timeout=75)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image") and len(r.content) > 5000:
            return url
    except Exception:
        pass
    print("  image unavailable, posting without image")
    return None


def assemble(o, intro, bodies, closing, category, topic, sources, related, img):
    secs = o["sections"]
    toc = "".join(f'<li><a href="#s{i+1}">{esc(s["heading"])}</a></li>' for i, s in enumerate(secs))
    parts = ['<div style="line-height:1.8;">']
    if img:
        parts.append(
            f'<div style="text-align:center;margin-bottom:18px;"><img src="{img}" alt="{esc(o["image_alt"])}" '
            f'title="{esc(o["title"])}" style="max-width:100%;height:auto;border-radius:8px;" loading="lazy"/></div>')
    parts += [
        f'<p><em>Published {time.strftime("%B %d, %Y", time.gmtime())}</em></p>',
        intro,
        "<!--more-->",
        f'<div style="background:#f6f8fa;border-left:4px solid #4a6cf7;padding:12px 18px;margin:20px 0;color:#222;">'
        f'<strong>In this article</strong><ul>{toc}</ul></div>',
    ]
    for i, (s, b) in enumerate(zip(secs, bodies)):
        parts.append(f'<h2 id="s{i+1}">{esc(s["heading"])}</h2>\n{b}')
    parts.append("<h2>Key Takeaways</h2><ul>" + "".join(f"<li>{esc(t)}</li>" for t in closing["key_takeaways"]) + "</ul>")
    parts.append("<h2>Frequently Asked Questions</h2>" + "".join(
        f'<h3>{esc(f["q"])}</h3><p>{esc(f["a"])}</p>' for f in closing["faqs"]))
    parts.append("<h2>Final Thoughts</h2>" + clean_html(closing["conclusion_html"]))
    if category in DISCLAIMER_CATS:
        parts.append("<p><em>Disclaimer: This article is for general information and education only. "
                     "It is not medical, legal or financial advice. Consult a qualified professional before making decisions.</em></p>")
    if category == "New Song":
        parts.append(f'<p><a href="https://www.youtube.com/results?search_query={quote(topic)}" '
                     'target="_blank" rel="noopener nofollow">Watch and listen on YouTube</a></p>')
    if sources:
        seen, lis = set(), []
        for title, uri in sources:
            if title not in seen and len(lis) < 6:
                seen.add(title)
                lis.append(f'<li><a href="{esc(uri)}" target="_blank" rel="noopener nofollow">{esc(title)}</a></li>')
        parts.append("<h3>Sources and further reading</h3><ul>" + "".join(lis) + "</ul>")
    if related:
        parts.append("<h3>Related articles</h3><ul>" + "".join(
            f'<li><a href="{esc(p["url"])}">{esc(p["title"])}</a></li>' for p in related) + "</ul>")
    parts.append("</div>")
    return "\n".join(parts)


# ------------------------------------------------------------------ main
def main():
    if not avail():
        sys.exit("No AI provider key found. Add at least one API key as a GitHub secret.")
    print("Providers:", avail())

    dry = os.getenv("DRY_RUN") == "1"
    state = load_state()
    token = "" if dry else access_token()

    category = CATEGORIES[state["next"] % len(CATEGORIES)]
    print("Category:", category)

    existing = [p.get("title", "") for p in blog_get(token, {"maxResults": 50, "fields": "items(title)"})] if token else []
    mine = state["titles"].setdefault(category, [])
    used = list({*existing, *mine, *[t for v in state["titles"].values() for t in v]})

    topic, live_ok = pick_topic(category, used)
    print("Topic:", topic, "| live:", live_ok)

    research, sources = ("", [])
    if live_ok and category in LIVE:
        research, sources = get_research(topic)

    article = None
    for attempt in range(3):
        try:
            outline = make_outline(category, topic, research, live_ok)
            intro = write_intro(category, topic, outline, research)
            bodies = write_sections(category, topic, outline, research)
            closing = write_closing(category, topic, outline)
            total = text_len(intro + " ".join(bodies))
            print("Words:", total)
            if total >= 1300:
                article = (outline, intro, bodies, closing)
                break
            print("  too short, retrying")
        except Exception as e:
            print("  attempt failed:", str(e)[:200])
        ORDER.append(ORDER.pop(0))
    if not article:
        sys.exit("Could not produce a complete article this run. Nothing posted; will retry next run.")

    outline, intro, bodies, closing = article
    img = get_image(outline["image_prompt"])
    related = blog_get(token, {"labels": category, "maxResults": 4, "fields": "items(title,url)"}) if token else []
    content = assemble(outline, intro, bodies, closing, category, topic, sources, related, img)
    publish(outline, content, category, token)

    mine.extend([topic, outline["title"]])
    state["titles"][category] = mine[-80:]
    state["next"] = (state["next"] + 1) % len(CATEGORIES)
    save_state(state)


if __name__ == "__main__":
    main()
