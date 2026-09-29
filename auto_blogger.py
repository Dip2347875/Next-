import html as htmllib
import json
import os
import random
import re
import sys
import time
from urllib.parse import quote

import requests

CATEGORIES = [
    "Finance", "Health", "News", "Education", "History", "Crime",
    "Paranormal", "Science Experiment", "Coding", "Question Solve",
    "Trending Now", "New Song", "Mathematics", "Biology", "New AI Update",
    "Gaming News", "Treating Tips", "Market News",
]

LIVE_CATEGORIES = {
    "News", "Trending Now", "New Song", "New AI Update",
    "Gaming News", "Market News", "Finance", "Crime",
}

STYLE = {
    "Finance": "in-depth personal-finance or economy analysis with real examples, numbers and practical steps",
    "Health": "evidence-based wellness deep dive; explain mechanisms, research findings and limits; no miracle claims",
    "News": "neutral, well-sourced analysis of a current news story: background, what happened, why it matters, what comes next",
    "Education": "practical study, learning or career guide with frameworks, steps and examples",
    "History": "rich narrative analysis of a real historical event or person: causes, key moments with dates, consequences, legacy",
    "Crime": "factual true-crime or crime-prevention analysis based only on publicly reported facts; never accuse private individuals",
    "Paranormal": "storytelling deep dive into a famous legend or reported phenomenon; present as folklore/reports and include scientific explanations",
    "Science Experiment": "safe hands-on experiment: materials, step-by-step method, safety notes, the science behind it, variations",
    "Coding": "detailed tutorial with working code in <pre><code> blocks, explanations of each step, common mistakes, best practices",
    "Question Solve": "take one popular exam, school or logic question and solve it step by step, explain the concept, show alternate methods and common errors",
    "Trending Now": "analysis of a topic trending online right now: origin, why it spread, what people say, what it means",
    "New Song": "spotlight/review of a newly released song: artist background, sound and production, lyrical themes (describe, never quote lyrics), reception",
    "Mathematics": "concept taught from basics to advanced with worked examples, tricks, and practice problems with answers",
    "Biology": "clear, deep explainer with mechanisms, real-world examples, and recent discoveries",
    "New AI Update": "analysis of a recent AI tool/model/product update: what changed, how it works, who benefits, limits, alternatives",
    "Gaming News": "news and analysis of a recent game release, update or industry event with context and community reaction",
    "Treating Tips": "practical home-care and everyday-problem tips with reasons behind each; advise seeing a professional for medical issues",
    "Market News": "stock, crypto or commodity market analysis: drivers, key levels/figures, sector impact, risks; not financial advice",
}

DISCLAIMER_CATS = {"Health", "Finance", "Market News", "Treating Tips", "Crime"}

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
STATE_FILE = "state.json"


def env(name):
    v = os.getenv(name)
    if not v:
        sys.exit(f"Missing environment variable: {name}")
    return v


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"next": 0, "titles": {}}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def gemini_raw(prompt, search=False, schema=None, max_tokens=8192, temp=0.8):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temp, "maxOutputTokens": max_tokens},
    }
    if search:
        payload["tools"] = [{"google_search": {}}]
    if schema:
        payload["generationConfig"]["responseMimeType"] = "application/json"
        payload["generationConfig"]["responseSchema"] = schema
    headers = {"x-goog-api-key": env("GEMINI_API_KEY"), "Content-Type": "application/json"}
    last = None
    for attempt in range(4):
        r = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=240)
        if r.status_code == 200:
            cand = r.json()["candidates"][0]
            text = "".join(p.get("text", "") for p in cand["content"]["parts"]).strip()
            chunks = cand.get("groundingMetadata", {}).get("groundingChunks", [])
            sources = []
            for c in chunks:
                w = c.get("web")
                if w and w.get("uri"):
                    sources.append((w.get("title", "Source"), w["uri"]))
            return text, sources
        last = f"{r.status_code}: {r.text[:300]}"
        time.sleep(12 * (attempt + 1))
    sys.exit(f"Gemini failed: {last}")


def gemini(prompt, **kw):
    return gemini_raw(prompt, **kw)[0]


def clean_html(t):
    t = re.sub(r"^```(?:html)?\s*|\s*```$", "", t.strip())
    return t.strip()


def esc(s):
    return htmllib.escape(s, quote=True)


# ---------------------------------------------------------------- topic
def pick_topic(category, used):
    avoid = "\n".join(f"- {t}" for t in used[-50:]) or "(none yet)"
    live = (
        "Use Google Search to find a genuinely current topic from the last few days."
        if category in LIVE_CATEGORIES
        else "Pick a specific topic people actively search for; not too broad."
    )
    prompt = f"""You are the chief editor of a high-quality English blog. Category: "{category}".
Today is {time.strftime('%B %d, %Y')}.
Suggest ONE fresh, specific, search-friendly article topic with real reader demand and low-to-medium competition.
{live}
Do NOT repeat or closely resemble these earlier articles:
{avoid}
Reply with only the topic in one line, no quotes, no extra text."""
    topic = gemini(prompt, search=category in LIVE_CATEGORIES, max_tokens=300, temp=0.9)
    return topic.splitlines()[0].strip(" \"'*#-")


# ---------------------------------------------------------------- outline
OUTLINE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "meta_description": {"type": "STRING"},
        "primary_keyword": {"type": "STRING"},
        "tags": {"type": "ARRAY", "items": {"type": "STRING"}},
        "image_prompt": {"type": "STRING"},
        "image_alt": {"type": "STRING"},
        "sections": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {"heading": {"type": "STRING"}, "focus": {"type": "STRING"}},
                "required": ["heading", "focus"],
            },
        },
    },
    "required": ["title", "meta_description", "primary_keyword", "tags", "image_prompt", "image_alt", "sections"],
}

CLOSING_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "key_takeaways": {"type": "ARRAY", "items": {"type": "STRING"}},
        "faqs": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {"q": {"type": "STRING"}, "a": {"type": "STRING"}},
                "required": ["q", "a"],
            },
        },
        "conclusion_html": {"type": "STRING"},
    },
    "required": ["key_takeaways", "faqs", "conclusion_html"],
}

VOICE = """Writing rules (apply strictly):
- Write like a seasoned specialist journalist: confident, natural, conversational but authoritative. Vary sentence length. Use concrete examples, comparisons, numbers, cause-and-effect analysis and "why this matters" insight.
- Go deep: explain the background, the mechanics, different viewpoints, risks and practical takeaways. No fluff, no repetition, no generic filler, no clichés such as "in today's fast-paced world" or "delve".
- Never mention being an AI. Never invent quotes, statistics, studies or sources. If unsure of a fact, omit it or say it is reported/estimated.
- Short paragraphs (2-4 sentences). Clean HTML only: <p>, <h3>, <ul>/<ol>, <strong>, <table> (only when a comparison helps), <pre><code> for code. No <h1>/<h2>, no markdown, no inline styles."""


def make_outline(category, topic, research):
    prompt = f"""Plan a comprehensive, best-in-class blog article.
Category: {category}
Topic: {topic}
Style: {STYLE[category]}
{"Research notes:\n" + research if research else ""}

Return:
- title: under 60 characters, primary keyword near the start, compelling, not clickbait.
- meta_description: 140-155 characters, includes the primary keyword, promises clear value.
- primary_keyword: 2-4 words.
- tags: 3-5 short keyword tags (do not include the category name).
- image_prompt: vivid description of a clean illustrative featured image, no text/logos, no real people's faces.
- image_alt: SEO alt text for it.
- sections: 6 to 8 sections that together cover the topic completely, in logical order. Each has a search-friendly heading (use the primary keyword or close variants in at least two) and a 'focus' saying exactly what to cover. The last section should be practical (what to do next / how to apply this)."""
    return json.loads(gemini(prompt, schema=OUTLINE_SCHEMA, max_tokens=3000, temp=0.7))


def write_intro(category, topic, outline, research):
    heads = "; ".join(s["heading"] for s in outline["sections"])
    prompt = f"""{VOICE}

Write ONLY the introduction (180-250 words, 2-4 short paragraphs) for an article titled "{outline['title']}".
Category: {category}. Topic: {topic}. Primary keyword: {outline['primary_keyword']} (use it naturally in the first paragraph).
The article covers: {heads}.
Open with a hook that shows why the reader should care, state what they will learn, do not summarise every section.
{"Use these facts if helpful:\n" + research if research else ""}
Return only HTML paragraphs."""
    return clean_html(gemini(prompt, max_tokens=1200))


def write_sections(category, topic, outline, research):
    secs = outline["sections"]
    all_heads = "\n".join(f"{i+1}. {s['heading']}" for i, s in enumerate(secs))
    results = []
    for start in range(0, len(secs), 3):
        group = secs[start:start + 3]
        listing = "\n".join(f"- {s['heading']}: {s['focus']}" for s in group)
        prompt = f"""{VOICE}

You are writing part of the article "{outline['title']}" (category: {category}, topic: {topic}).
Full article structure:
{all_heads}

Write the body for ONLY these sections, in this order:
{listing}

For each section write 400-550 words of substantive, analytical content (explain, analyse, give examples and context). Do NOT include the section heading itself. Do not repeat points from other sections. Use <h3> sub-headings where they help.
{"Rely on these facts:\n" + research if research else ""}
Return a JSON array of strings: one HTML string per section, same order."""
        arr = json.loads(gemini(
            prompt,
            schema={"type": "ARRAY", "items": {"type": "STRING"}},
            max_tokens=8192, temp=0.75,
        ))
        if len(arr) != len(group):
            sys.exit("Section count mismatch, aborting this run.")
        results.extend(clean_html(a) for a in arr)
    return results


def write_closing(category, topic, outline):
    prompt = f"""{VOICE}

Article: "{outline['title']}" (category: {category}, topic: {topic}).
Sections covered: {'; '.join(s['heading'] for s in outline['sections'])}.
Return JSON with:
- key_takeaways: 5-6 specific, useful bullet points (plain text).
- faqs: 4 to 5 real questions people search about this topic, each with a helpful 50-90 word plain-text answer.
- conclusion_html: 120-180 words in <p> tags, uses the primary keyword "{outline['primary_keyword']}" once, ends with a clear, natural next step for the reader."""
    return json.loads(gemini(prompt, schema=CLOSING_SCHEMA, max_tokens=3500, temp=0.7))


# ---------------------------------------------------------------- blogger
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


def related_posts(token, category):
    try:
        r = requests.get(
            f"https://www.googleapis.com/blogger/v3/blogs/{env('BLOGGER_BLOG_ID')}/posts",
            params={"labels": category, "maxResults": 4, "fields": "items(title,url)"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )
        return r.json().get("items", []) if r.status_code == 200 else []
    except Exception:
        return []


def assemble(outline, intro, bodies, closing, category, topic, sources, related):
    img = (
        "https://image.pollinations.ai/prompt/" + quote(outline["image_prompt"])
        + f"?width=1200&height=630&nologo=true&seed={random.randint(1, 10**6)}"
    )
    secs = outline["sections"]
    toc = "".join(f'<li><a href="#s{i+1}">{esc(s["heading"])}</a></li>' for i, s in enumerate(secs))
    parts = [
        '<div style="line-height:1.8;">',
        f'<div style="text-align:center;margin-bottom:18px;"><img src="{img}" alt="{esc(outline["image_alt"])}" '
        f'style="max-width:100%;height:auto;border-radius:8px;" loading="lazy"/></div>',
        f'<p><em>Updated {time.strftime("%B %d, %Y")}</em></p>',
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


def publish(outline, content, category, token):
    labels = [category] + [t for t in outline["tags"] if t.lower() != category.lower()][:4]
    body = {"kind": "blogger#post", "title": outline["title"], "content": content, "labels": labels}
    if os.getenv("DRY_RUN") == "1":
        print(json.dumps(body, indent=2, ensure_ascii=False)[:4000])
        return
    r = requests.post(
        f"https://www.googleapis.com/blogger/v3/blogs/{env('BLOGGER_BLOG_ID')}/posts?isDraft=false",
        headers={"Authorization": f"Bearer {token}"},
        json=body,
        timeout=120,
    )
    if r.status_code not in (200, 201):
        sys.exit(f"Blogger publish failed: {r.status_code} {r.text}")
    print("Published:", r.json().get("url"))


def main():
    state = load_state()
    category = CATEGORIES[state["next"] % len(CATEGORIES)]
    used = state["titles"].setdefault(category, [])
    print("Category:", category)

    topic = pick_topic(category, used)
    print("Topic:", topic)

    research, sources = "", []
    if category in LIVE_CATEGORIES:
        research, sources = gemini_raw(
            f"Using Google Search, gather the latest accurate facts, figures, dates, context and differing viewpoints about: {topic}. "
            "Return detailed bullet notes. Include only verified information.",
            search=True, max_tokens=2500, temp=0.3,
        )

    outline = make_outline(category, topic, research)
    intro = write_intro(category, topic, outline, research)
    bodies = write_sections(category, topic, outline, research)
    closing = write_closing(category, topic, outline)

    words = len(re.sub(r"<[^>]+>", " ", intro + " ".join(bodies)).split())
    print("Words:", words)
    if words < 1800:
        sys.exit("Article too short, nothing posted.")

    token = access_token() if os.getenv("DRY_RUN") != "1" else ""
    related = related_posts(token, category) if token else []
    content = assemble(outline, intro, bodies, closing, category, topic, sources, related)
    publish(outline, content, category, token)

    used.append(outline["title"])
    state["titles"][category] = used[-60:]
    state["next"] = (state["next"] + 1) % len(CATEGORIES)
    save_state(state)


if __name__ == "__main__":
    main()
