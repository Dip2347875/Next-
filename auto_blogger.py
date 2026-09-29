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

DISCLAIMER_CATS = {
    "Health",
    "Finance",
    "Market News",
    "Treating Tips",
    "Crime",
}

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/"
    f"v1beta/models/{GEMINI_MODEL}:generateContent"
)

STATE_FILE = "state.json"


def env(name):
    v = os.getenv(name)

    if not v:
        sys.exit(
            f"Missing environment variable: {name}"
        )

    return v


def load_state():
    if os.path.exists(STATE_FILE):
        with open(
            STATE_FILE,
            encoding="utf-8"
        ) as f:
            return json.load(f)

    return {
        "next": 0,
        "titles": {}
    }


def save_state(state):
    with open(
        STATE_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            state,
            f,
            indent=2,
            ensure_ascii=False
        )


def gemini_raw(
    prompt,
    search=False,
    schema=None,
    max_tokens=8192,
    temp=0.8
):
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": temp,
            "maxOutputTokens": max_tokens
        },
    }

    if search:
        payload["tools"] = [
            {
                "google_search": {}
            }
        ]

    if schema:
        payload["generationConfig"][
            "responseMimeType"
        ] = "application/json"

        payload["generationConfig"][
            "responseSchema"
        ] = schema

    headers = {
        "x-goog-api-key": env(
            "GEMINI_API_KEY"
        ),
        "Content-Type": "application/json"
    }

    last = None

    for attempt in range(4):

        r = requests.post(
            GEMINI_URL,
            headers=headers,
            json=payload,
            timeout=240
        )

        if r.status_code == 200:

            cand = r.json()[
                "candidates"
            ][0]

            text = "".join(
                p.get("text", "")
                for p in cand[
                    "content"
                ]["parts"]
            ).strip()

            chunks = cand.get(
                "groundingMetadata",
                {}
            ).get(
                "groundingChunks",
                []
            )

            sources = []

            for c in chunks:

                w = c.get("web")

                if w and w.get("uri"):

                    sources.append(
                        (
                            w.get(
                                "title",
                                "Source"
                            ),
                            w["uri"]
                        )
                    )

            return text, sources

        last = (
            f"{r.status_code}: "
            f"{r.text[:300]}"
        )

        time.sleep(
            12 * (attempt + 1)
        )

    sys.exit(
        f"Gemini failed: {last}"
    )


def gemini(prompt, **kw):
    return gemini_raw(
        prompt,
        **kw
    )[0]


def clean_html(t):
    t = re.sub(
        r"^```(?:html)?\s*|\s*```$",
        "",
        t.strip()
    )

    return t.strip()


def esc(s):
    return htmllib.escape(
        s,
        quote=True
    )


# ---------------------------------------------------------------- topic

def pick_topic(category, used):

    avoid = "\n".join(
        f"- {t}"
        for t in used[-50:]
    ) or "(none yet)"

    live = (
        "Use Google Search to find a genuinely current topic from the last few days."
        if category in LIVE_CATEGORIES
        else
        "Pick a specific topic people actively search for; not too broad."
    )

    prompt = f"""You are the chief editor of a high-quality English blog. Category: "{category}".
Today is {time.strftime('%B %d, %Y')}.
Suggest ONE fresh, specific, search-friendly article topic with real reader demand and low-to-medium competition.
{live}
Do NOT repeat or closely resemble these earlier articles:
{avoid}
Reply with only the topic in one line, no quotes, no extra text."""

    topic = gemini(
        prompt,
        search=category in LIVE_CATEGORIES,
        max_tokens=300,
        temp=0.9
    )

    return topic.splitlines()[
        0
    ].strip(
        " \"'*#-"
    )


# ---------------------------------------------------------------- outline

OUTLINE_SCHEMA = {
    "type": "OBJECT",

    "properties": {

        "title": {
            "type": "STRING"
        },

        "meta_description": {
            "type": "STRING"
        },

        "primary_keyword": {
            "type": "STRING"
        },

        "tags": {
            "type": "ARRAY",
            "items": {
                "type": "STRING"
            }
        },

        "image_prompt": {
            "type": "STRING"
        },

        "image_alt": {
            "type": "STRING"
        },

        "sections": {
            "type": "ARRAY",

            "items": {
                "type": "OBJECT",

                "properties": {

                    "heading": {
                        "type": "STRING"
                    },

                    "focus": {
                        "type": "STRING"
                    },
                },

                "required": [
                    "heading",
                    "focus"
                ],
            },
        },
    },

    "required": [
        "title",
        "meta_description",
        "primary_keyword",
        "tags",
        "image_prompt",
        "image_alt",
        "sections"
    ],
}


CLOSING_SCHEMA = {
    "type": "OBJECT",

    "properties": {

        "key_takeaways": {
            "type": "ARRAY",
            "items": {
                "type": "STRING"
            }
        },

        "faqs": {
            "type": "ARRAY",

            "items": {
                "type": "OBJECT",

                "properties": {

                    "q": {
                        "type": "STRING"
                    },

                    "a": {
                        "type": "STRING"
                    },
                },

                "required": [
                    "q",
                    "a"
                ],
            },
        },

        "conclusion_html": {
            "type": "STRING"
        },
    },

    "required": [
        "key_takeaways",
        "faqs",
        "conclusion_html"
    ],
}


VOICE = """Writing rules (apply strictly):
- Write like a seasoned specialist journalist: confident, natural, conversational but authoritative. Vary sentence length. Use concrete examples, comparisons, numbers, cause-and-effect analysis and "why this matters" insight.
- Go deep: explain the background, the mechanics, different viewpoints, risks and practical takeaways. No fluff, no repetition, no generic filler, no clichés such as "in today's fast-paced world" or "delve".
- Never mention being an AI. Never invent quotes, statistics, studies or sources. If unsure of a fact, omit it or say it is reported/estimated.
- Short paragraphs (2-4 sentences). Clean HTML only: <p>, <h3>, <ul>/<ol>, <strong>, <table> (only when a comparison helps), <pre><code> for code. No <h1>/<h2>, no markdown, no inline styles."""


def make_outline(
    category,
    topic,
    research
):

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

    return json.loads(
        gemini(
            prompt,
            schema=OUTLINE_SCHEMA,
            max_tokens=3000,
            temp=0.7
        )
    )


def write_intro(
    category,
    topic,
    outline,
    research
):

    heads = "; ".join(
        s["heading"]
        for s in outline["sections"]
    )

    prompt = f"""{VOICE}

Write ONLY the introduction (180-250 words, 2-4 short paragraphs) for an article titled "{outline['title']}".
Category: {category}. Topic: {topic}. Primary keyword: {outline['primary_keyword']} (use it naturally in the first paragraph).
The article covers: {heads}.
Open with a hook that shows why the reader should care, state what they will learn, do not summarise every section.
{"Use these facts if helpful:\n" + research if research else ""}
Return only HTML paragraphs."""

    return clean_html(
        gemini(
            prompt,
            max_tokens=1200
        )
    )


def write_sections(
    category,
    topic,
    outline,
    research
):

    secs = outline["sections"]

    all_heads = "\n".join(
        f"{i+1}. {s['heading']}"
        for i, s in enumerate(secs)
    )

    results = []

    for start in range(
        0,
        len(secs),
        3
    ):

        group = secs[
            start:start + 3
        ]

        listing = "\n".join(
            f"- {s['heading']}: {s['focus']}"
            for s in group
        )

        prompt = f"""{VOICE}

You are writing part of the article "{outline['title']}" (category: {category}, topic: {topic}).
Full article structure:
{all_heads}

Write the body for ONLY these sections, in this order:
{listing}

For each section write 400-550 words of substantive, analytical content (explain, analyse, give examples and context). Do NOT include the section heading itself. Do not repeat points from other sections. Use <h3> sub-headings where they help.
{"Rely on these facts:\n" + research if research else ""}
Return a JSON array of strings: one HTML string per section, same order."""

        arr = json.loads(
            gemini(
                prompt,
                schema={
                    "type": "ARRAY",
                    "items": {
                        "type": "STRING"
                    }
                },
                max_tokens=8192,
                temp=0.75
            )
        )

        if len(arr) != len(group):
            sys.exit(
                "Section count mismatch, aborting this run."
            )

        results.extend(
            clean_html(a)
            for a in arr
        )

    return results
def write_closing(category, topic, outline):
    prompt = f"""{VOICE}

Article: "{outline['title']}" (category: {category}, topic: {topic}).
Sections covered: {'; '.join(s['heading'] for s in outline['sections'])}.

Return JSON with:
- key_takeaways: 5-6 specific, useful bullet points (plain text).
- faqs: 4 to 5 real questions people search about this topic, each with a helpful 50-90 word plain-text answer.
- conclusion_html: 120-180 words in <p> tags, uses the primary keyword "{outline['primary_keyword']}" once, ends with a clear, natural next step for the reader.
"""
    return gemini_raw(
        prompt,
        search=(category in LIVE_CATEGORIES),
        schema=CLOSING_SCHEMA,
        max_tokens=5000,
        temp=0.7,
    )[0]


def access_token():
    data = {
        "client_id": env("GOOGLE_CLIENT_ID"),
        "client_secret": env("GOOGLE_CLIENT_SECRET"),
        "refresh_token": env("GOOGLE_REFRESH_TOKEN"),
        "grant_type": "refresh_token",
    }

    r = requests.post(
        "https://oauth2.googleapis.com/token",
        data=data,
        timeout=30,
    )

    if r.status_code != 200:
        raise RuntimeError(
            f"OAuth token failed: {r.status_code}: {r.text[:1000]}"
        )

    return r.json()["access_token"]


def related_posts(token, category):
    blog_id = env("BLOGGER_BLOG_ID")

    url = (
        f"https://www.googleapis.com/blogger/v3/blogs/"
        f"{blog_id}/posts?labels={quote(category)}&maxResults=6"
    )

    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )

    if r.status_code != 200:
        return []

    items = r.json().get("items", [])

    return [
        {
            "title": x.get("title", ""),
            "url": x.get("url", ""),
        }
        for x in items
        if x.get("title") and x.get("url")
    ]


def assemble(category, topic, outline, intro, sections, closing, sources, related):
    title = esc(outline["title"])
    meta = esc(outline["meta_description"])
    keyword = esc(outline["primary_keyword"])
    alt = esc(outline["image_alt"])

    image_prompt = outline.get("image_prompt", "").strip()

    image_url = (
        "https://image.pollinations.ai/prompt/"
        + quote(image_prompt)
        + "?width=1280&height=720&nologo=true"
    )

    html = []

    html.append(
        f'<div class="article-meta" style="display:none">'
        f'<meta name="description" content="{meta}">'
        f'<meta name="keywords" content="{keyword}">'
        f'</div>'
    )

    html.append(
        f'<p><strong>Topic:</strong> {esc(topic)}</p>'
    )

    if image_prompt:
        html.append(
            f'<figure>'
            f'<img src="{image_url}" alt="{alt}" '
            f'style="width:100%;height:auto;" loading="lazy">'
            f'</figure>'
        )

    html.append("<h2>Table of Contents</h2>")
    html.append("<ol>")

    for section in outline["sections"]:
        heading = esc(section["heading"])
        anchor = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")

        html.append(
            f'<li><a href="#{anchor}">{heading}</a></li>'
        )

    html.append("</ol>")

    html.append("<h2>Introduction</h2>")
    html.append(intro)

    for section, body in zip(outline["sections"], sections):
        heading = esc(section["heading"])
        anchor = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")

        html.append(
            f'<h2 id="{anchor}">{heading}</h2>'
        )

        html.append(body)

    html.append("<h2>Key Takeaways</h2>")
    html.append("<ul>")

    for item in closing.get("key_takeaways", []):
        html.append(
            f"<li>{esc(str(item))}</li>"
        )

    html.append("</ul>")

    html.append("<h2>Frequently Asked Questions</h2>")

    for faq in closing.get("faqs", []):
        question = esc(faq.get("q", ""))
        answer = esc(faq.get("a", ""))

        if question and answer:
            html.append(
                f"<h3>{question}</h3>"
            )
            html.append(
                f"<p>{answer}</p>"
            )

    html.append("<h2>Final Thoughts</h2>")

    conclusion = closing.get("conclusion_html", "")
    html.append(conclusion)

    if category in DISCLAIMER_CATS:
        html.append(
            "<div style=\"margin-top:24px;padding:15px;"
            "border-left:4px solid #999;background:#f7f7f7;\">"
            "<strong>Disclaimer:</strong> "
            "This article is for general informational purposes only "
            "and should not be considered professional advice."
            "</div>"
        )

    if category == "New Song":
        search_url = (
            "https://www.youtube.com/results?search_query="
            + quote(topic)
        )

        html.append(
            '<p><strong>Watch or search for the song:</strong> '
            f'<a href="{search_url}" target="_blank" '
            'rel="noopener">YouTube Search</a></p>'
        )

    if sources:
        html.append("<h2>Sources</h2>")
        html.append("<ul>")

        for source in sources:
            if isinstance(source, dict):
                name = source.get("title") or source.get("name") or "Source"
                url = source.get("url") or source.get("link")

                if url:
                    html.append(
                        f'<li><a href="{esc(url)}" '
                        f'target="_blank" rel="noopener">'
                        f'{esc(name)}</a></li>'
                    )
                else:
                    html.append(
                        f"<li>{esc(name)}</li>"
                    )

            elif isinstance(source, str):
                html.append(
                    f"<li>{esc(source)}</li>"
                )

        html.append("</ul>")

    if related:
        html.append("<h2>Related Posts</h2>")
        html.append("<ul>")

        for post in related:
            html.append(
                f'<li><a href="{esc(post["url"])}" '
                f'target="_blank" rel="noopener">'
                f'{esc(post["title"])}</a></li>'
            )

        html.append("</ul>")

    return "\n".join(html)


def publish(token, title, content, labels):
    blog_id = env("BLOGGER_BLOG_ID")

    if os.getenv("DRY_RUN", "0") == "1":
        print("DRY_RUN=1")
        print("Title:", title)
        print("Labels:", labels)
        print("Content length:", len(content))
        return None

    url = (
        f"https://www.googleapis.com/blogger/v3/blogs/"
        f"{blog_id}/posts/"
    )

    payload = {
        "kind": "blogger#post",
        "title": title,
        "content": content,
        "labels": labels,
    }

    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    if r.status_code not in (200, 201):
        raise RuntimeError(
            f"Blogger publish failed: {r.status_code}: "
            f"{r.text[:2000]}"
        )

    data = r.json()

    print("Published successfully:")
    print(data.get("url", ""))

    return data


def main():
    state = load_state()

    next_index = int(state.get("next", 0)) % len(CATEGORIES)
    category = CATEGORIES[next_index]

    used_titles = list(state.get("titles", {}).values())

    print(f"Category: {category}")

    topic = pick_topic(category, used_titles)

    print(f"Topic: {topic}")

    research = ""

    if category in LIVE_CATEGORIES:
        research = gemini(
            f"""
Research this topic carefully using current web information:

Topic: {topic}
Category: {category}

Give concise factual research notes with dates, important facts,
relevant developments, and source information.

Do not invent facts.
""",
            search=True,
            max_tokens=6000,
            temp=0.3,
        )

    outline_raw = make_outline(
        category,
        topic,
        research,
    )

    if isinstance(outline_raw, str):
        outline = json.loads(outline_raw)
    else:
        outline = outline_raw

    print("Outline created.")

    intro = write_intro(
        category,
        topic,
        outline,
        research,
    )

    print("Introduction created.")

    sections = write_sections(
        category,
        topic,
        outline,
        research,
    )

    print(
        f"Sections created: {len(sections)}"
    )

    closing_raw = write_closing(
        category,
        topic,
        outline,
    )

    if isinstance(closing_raw, str):
        closing = json.loads(closing_raw)
    else:
        closing = closing_raw

    print("Closing created.")

    total_text = (
        str(intro)
        + "\n"
        + "\n".join(sections)
        + "\n"
        + str(closing)
    )

    word_count = len(
        re.findall(r"\b[\w'-]+\b", total_text)
    )

    print(f"Approx word count: {word_count}")

    if word_count < 1800:
        raise RuntimeError(
            f"Generated article is too short: {word_count} words"
        )

    token = access_token()

    related = related_posts(
        token,
        category,
    )

    sources = []

    article = assemble(
        category,
        topic,
        outline,
        intro,
        sections,
        closing,
        sources,
        related,
    )

    title = outline["title"]

    publish(
        token,
        title,
        article,
        [category],
    )

    state.setdefault("titles", {})

    state["titles"][category] = title

    if len(state["titles"]) > 50:
        old_items = list(state["titles"].items())

        state["titles"] = dict(
            old_items[-50:]
        )

    state["next"] = (
        next_index + 1
    ) % len(CATEGORIES)

    save_state(state)

    print("State saved.")
    print("Done.")


if __name__ == "__main__":
    main()
