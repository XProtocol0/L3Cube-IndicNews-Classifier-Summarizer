import html
import re
from datetime import datetime
from typing import Dict, List

import feedparser
import streamlit as st
from transformers import pipeline

try:
    import google.generativeai as genai
except ImportError:
    genai = None


st.set_page_config(page_title="India Daily Briefing", page_icon="🗞️", layout="wide")

FEED_URLS = {
    "The Hindu - National": "https://www.thehindu.com/news/national/feeder/default.rss",
    "The Hindu - Sport": "https://www.thehindu.com/sport/feeder/default.rss",
    "The Hindu - Tech": "https://www.thehindu.com/sci-tech/technology/feeder/default.rss",
    "The Hindu - Entertainment": "https://www.thehindu.com/entertainment/feeder/default.rss",
    "NDTV - Top Stories": "https://feeds.feedburner.com/ndtvnews-top-stories",
    "NDTV - Sports": "https://feeds.feedburner.com/ndtvsports-latest",
    "NDTV - Tech": "https://feeds.feedburner.com/gadgets360-latest",
    "NDTV - Entertainment": "https://feeds.feedburner.com/ndtvmovies-latest",
    "Indian Express - India": "https://indianexpress.com/section/india/feed/",
    "Indian Express - Sports": "https://indianexpress.com/section/sports/feed/",
    "Indian Express - Tech": "https://indianexpress.com/section/technology/feed/",
    "Indian Express - Entertainment": "https://indianexpress.com/section/entertainment/feed/",
}

CATEGORY_LABELS = ["Politics", "Sports", "Technology", "Business", "Entertainment"]
CATEGORY_MAP = {
    "Politics": "Politics",
    "Sports": "Sports",
    "Technology": "Tech",
    "Business": "Business",
    "Entertainment": "Entertainment",
}
TAB_ORDER = ["Politics", "Sports", "Tech", "Business", "Entertainment"]


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@st.cache_resource(show_spinner=True)
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


@st.cache_data(ttl=900, show_spinner=False)
def fetch_latest_articles(limit: int = 30) -> List[Dict]:
    items: List[Dict] = []
    seen = set()

    for source, url in FEED_URLS.items():
        parsed = feedparser.parse(url)
        for entry in parsed.entries:
            link = entry.get("link", "").strip()
            title = _clean_text(entry.get("title", "Untitled"))
            key = link or title
            if not key or key in seen:
                continue
            seen.add(key)

            description = _clean_text(
                entry.get("summary", "") or entry.get("description", "")
            )
            published = entry.get("published", "") or entry.get("updated", "")

            items.append(
                {
                    "title": title,
                    "link": link,
                    "description": description,
                    "published": published,
                    "source": source,
                }
            )

    items = sorted(items, key=lambda x: x.get("published", ""), reverse=True)
    return items[:limit]


def classify_articles(articles: List[Dict], classifier) -> List[Dict]:
    if not articles:
        return articles

    texts = [
        f"{a['title']}. {a.get('description', '')}".strip()
        for a in articles
    ]
    outputs = classifier(
        texts,
        candidate_labels=CATEGORY_LABELS,
        multi_label=False,
        hypothesis_template="This news article is about {}.",
    )

    if isinstance(outputs, dict):
        outputs = [outputs]

    enriched = []
    for article, output in zip(articles, outputs):
        predicted = output["labels"][0]
        confidence = float(output["scores"][0])
        article = dict(article)
        article["raw_category"] = predicted
        article["category"] = CATEGORY_MAP.get(predicted, "Politics")
        article["confidence"] = confidence
        enriched.append(article)

    return enriched


def _fallback_summary(text: str) -> str:
    text = text.strip()
    if not text:
        return "Summary unavailable."
    chunks = re.split(r"(?<=[.!?])\s+", text)
    chunks = [c.strip() for c in chunks if c.strip()]
    lines = chunks[:3]
    while len(lines) < 3:
        lines.append("Additional details are awaited.")
    return "\n".join(lines)


@st.cache_data(ttl=3600, show_spinner=False)
def summarize_article(title: str, description: str, api_key: str) -> str:
    context = f"Title: {title}\nDetails: {description}"
    if not api_key or genai is None:
        return _fallback_summary(description or title)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are an editor for an Indian daily briefing app. "
            "Write exactly 3 concise lines in plain text, no bullets, no numbering. "
            "Each line should be <= 22 words and fact-focused.\n\n"
            f"{context}"
        )
        response = model.generate_content(prompt)
        text = _clean_text(response.text if response and response.text else "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        if not lines:
            return _fallback_summary(description or title)

        lines = lines[:3]
        while len(lines) < 3:
            lines.append("More details are expected as the story develops.")
        return "\n".join(lines)
    except Exception:
        return _fallback_summary(description or title)


def build_newsletter_fallback(recipient_name: str, selected_articles: List[Dict]) -> str:
    lines = [
        f"Subject: Your India Briefing ({datetime.now().strftime('%d %b %Y')})",
        "",
        f"Hi {recipient_name},",
        "",
        "Here are your top updates today:",
        "",
    ]
    for idx, article in enumerate(selected_articles[:10], start=1):
        lines.append(f"{idx}. {article['title']} ({article['category']}, {article['source']})")
    lines.append("")
    lines.append("Regards,")
    lines.append("India Daily Briefing")
    return "\n".join(lines)


@st.cache_data(ttl=1800, show_spinner=False)
def build_newsletter_with_llm(
    selected_articles: List[Dict],
    api_key: str,
    recipient_name: str,
    tone: str,
) -> str:
    if not api_key or genai is None:
        return build_newsletter_fallback(recipient_name, selected_articles)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        compact_articles = [
            {
                "title": a["title"],
                "category": a["category"],
                "source": a["source"],
                "summary": a.get("summary", ""),
            }
            for a in selected_articles[:12]
        ]
        prompt = (
            "Create an email-style personalized newsletter for Indian news. "
            f"Recipient: {recipient_name}. Tone: {tone}. "
            "Return plain text with Subject line, greeting, 5-8 bullet points, and sign-off. "
            "Use only facts from the provided items.\n\n"
            f"Items: {compact_articles}"
        )
        response = model.generate_content(prompt)
        text = response.text if response and response.text else ""
        text = text.strip()
        return text if text else build_newsletter_fallback(recipient_name, selected_articles)
    except Exception:
        return build_newsletter_fallback(recipient_name, selected_articles)


def render_article_card(article: Dict):
    st.markdown(f"### [{article['title']}]({article['link']})")
    st.caption(
        f"Source: {article['source']} | Published: {article.get('published', 'N/A')} | "
        f"Zero-shot confidence: {article['confidence']:.2f}"
    )
    st.write(article.get("summary", "Summary unavailable."))


def main():
    st.title("India Daily Briefing")
    st.write(
        "Live RSS news from The Hindu, NDTV, and Indian Express with zero-shot category tagging and 3-line summaries."
    )

    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "Gemini API Key (optional)",
            type="password",
            help="Used for 3-line summaries and personalized newsletter generation.",
        )
        article_limit = st.slider("Articles per refresh", min_value=10, max_value=80, value=50, step=5)
        if st.button("Refresh Feed"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Fetching latest RSS articles..."):
        articles = fetch_latest_articles(limit=article_limit)

    if not articles:
        st.error("No articles fetched from RSS feeds. Please try again.")
        return

    classifier = load_classifier()

    with st.spinner("Classifying articles with BART MNLI..."):
        articles = classify_articles(articles, classifier)

    with st.spinner("Generating 3-line summaries..."):
        for article in articles:
            article["summary"] = summarize_article(
                article["title"],
                article.get("description", ""),
                api_key,
            )

    grouped = {k: [] for k in TAB_ORDER}
    for article in articles:
        grouped.setdefault(article["category"], []).append(article)

    tab_widgets = st.tabs(TAB_ORDER)
    for tab_name, tab in zip(TAB_ORDER, tab_widgets):
        with tab:
            bucket = grouped.get(tab_name, [])
            if not bucket:
                st.info(f"No articles in {tab_name} right now. Try Refresh Feed for the latest RSS items.")
            for article in bucket:
                render_article_card(article)
                st.divider()

    st.subheader("Personalized Newsletter Generator")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        recipient_name = st.text_input("Recipient name", value="Reader")
    with col2:
        selected_categories = st.multiselect(
            "Choose categories",
            options=TAB_ORDER,
            default=TAB_ORDER,
        )
    with col3:
        tone = st.selectbox("Tone", options=["Professional", "Friendly", "Concise"], index=0)

    selected_articles = [a for a in articles if a["category"] in selected_categories]

    if st.button("Generate Email Digest"):
        digest = build_newsletter_with_llm(selected_articles, api_key, recipient_name, tone)
        st.text_area("Generated Digest", value=digest, height=320)


if __name__ == "__main__":
    main()
