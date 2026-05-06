import html
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, List, Optional

import feedparser
import pandas as pd
import streamlit as st
from transformers import pipeline
from dotenv import load_dotenv

try:
    import requests
except ImportError:
    requests = None

load_dotenv()


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

DEFAULT_CATEGORIES = ["Politics", "Sports", "Technology", "Business", "Entertainment"]
DEFAULT_CATEGORY_MAP = {
    "Politics": "Politics",
    "Sports": "Sports",
    "Technology": "Tech",
    "Business": "Business",
    "Entertainment": "Entertainment",
}
DEFAULT_TAB_ORDER = ["Politics", "Sports", "Tech", "Business", "Entertainment"]
LABELS_PATH = Path(__file__).resolve().parent / "labels.txt"
MISTRAL_API_URL = os.getenv("MISTRAL_API_URL", "https://api.mistral.ai/v1/chat/completions")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_RPM = int(os.getenv("MISTRAL_RPM", "10"))
MISTRAL_RPD = int(os.getenv("MISTRAL_RPD", "1000"))
MISTRAL_MIN_INTERVAL = 60.0 / max(MISTRAL_RPM, 1)
_MISTRAL_LAST_CALL = 0.0
_MISTRAL_DAY_KEY = ""
_MISTRAL_DAY_COUNT = 0


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_summary_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text.strip()


def _parse_entry_datetime(entry: Dict) -> datetime:
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            try:
                return datetime(*parsed[:6], tzinfo=timezone.utc)
            except Exception:
                continue
    for key in ("published", "updated"):
        raw = entry.get(key, "")
        if raw:
            try:
                parsed = parsedate_to_datetime(raw)
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except Exception:
                continue
    return datetime.min.replace(tzinfo=timezone.utc)


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lowered = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lowered:
            return lowered[key]
    return None


def _normalize_label(label: str) -> str:
    cleaned = re.sub(r"[_-]+", " ", label or "").strip()
    return cleaned.title() if cleaned else "Unknown"


@st.cache_data(ttl=3600, show_spinner=False)
def load_labels_file(path: str) -> List[str]:
    if not path:
        return []
    label_path = Path(path)
    if not label_path.exists():
        return []
    lines = [line.strip() for line in label_path.read_text().splitlines()]
    return [line for line in lines if line]


def resolve_categories(labels_path: str) -> Dict[str, object]:
    file_labels = load_labels_file(labels_path)
    if file_labels:
        label_map = {label: _normalize_label(label) for label in file_labels}
        tab_order = [label_map[label] for label in file_labels]
        return {
            "labels": file_labels,
            "label_map": label_map,
            "tab_order": tab_order,
            "source": "labels",
        }
    return {
        "labels": DEFAULT_CATEGORIES,
        "label_map": DEFAULT_CATEGORY_MAP,
        "tab_order": DEFAULT_TAB_ORDER,
        "source": "default",
    }


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
            published_dt = _parse_entry_datetime(entry)

            items.append(
                {
                    "title": title,
                    "link": link,
                    "description": description,
                    "published": published,
                    "published_dt": published_dt,
                    "source": source,
                }
            )

    items = sorted(
        items,
        key=lambda x: x.get("published_dt", datetime.min.replace(tzinfo=timezone.utc)),
        reverse=True,
    )
    return items[:limit]


def load_articles_from_csv(uploaded_file, limit: int = 30) -> List[Dict]:
    if uploaded_file is None:
        return []

    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)

    title_col = _find_column(
        df,
        [
            "title",
            "headline",
            "headline_text",
            "news_title",
            "article_title",
        ],
    )
    if title_col is None:
        raise ValueError(
            "CSV must include a title or headline column (e.g., title, headline_text)."
        )

    desc_col = _find_column(
        df,
        ["description", "summary", "text", "details", "content", "article"],
    )
    link_col = _find_column(df, ["link", "url", "article_url", "news_url"])
    published_col = _find_column(
        df,
        ["published", "date", "datetime", "timestamp", "time", "published_at", "updated_at"],
    )
    source_col = _find_column(
        df,
        ["source", "publisher", "news_source", "outlet"],
    )

    work = df.copy()
    work[title_col] = work[title_col].astype(str).str.strip()
    work = work[work[title_col] != ""]

    if desc_col:
        work[desc_col] = work[desc_col].fillna("").astype(str).str.strip()
    if link_col:
        work[link_col] = work[link_col].fillna("").astype(str).str.strip()
    if source_col:
        work[source_col] = work[source_col].fillna("").astype(str).str.strip()

    if published_col:
        published_series = pd.to_datetime(
            work[published_col], errors="coerce", utc=True
        )
    else:
        published_series = None

    items: List[Dict] = []
    for row_idx, row in work.iterrows():
        published_dt = datetime.min.replace(tzinfo=timezone.utc)
        if published_series is not None:
            parsed_dt = published_series.loc[row_idx]
            if pd.notna(parsed_dt):
                published_dt = parsed_dt.to_pydatetime()

        published_raw = ""
        if published_col:
            published_raw = row.get(published_col, "")

        items.append(
            {
                "title": _clean_text(row.get(title_col, "")),
                "link": row.get(link_col, "") if link_col else "",
                "description": _clean_text(row.get(desc_col, "")) if desc_col else "",
                "published": str(published_raw).strip() if published_raw is not None else "",
                "published_dt": published_dt,
                "source": row.get(source_col, "") if source_col else "Local CSV",
            }
        )

    if published_series is not None:
        items = sorted(
            items,
            key=lambda x: x.get("published_dt", datetime.min.replace(tzinfo=timezone.utc)),
            reverse=True,
        )

    return items[:limit]


def classify_articles(
    articles: List[Dict],
    classifier,
    category_labels: List[str],
    label_map: Dict[str, str],
) -> List[Dict]:
    if not articles:
        return articles

    texts = [
        f"{a['title']}. {a.get('description', '')}".strip()
        for a in articles
    ]
    outputs = classifier(
        texts,
        candidate_labels=category_labels,
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
        article["category"] = label_map.get(predicted, "Unknown")
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


def _extract_json_payload(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if match:
        return match.group(0).strip()
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        return match.group(0).strip()
    return cleaned


def _normalize_json_payload(text: str) -> str:
    payload = _extract_json_payload(text)
    if not payload:
        return ""
    payload = re.sub(r",\s*([}\]])", r"\1", payload)
    out = []
    in_string = False
    escape = False
    for ch in payload:
        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ch in ("\n", "\r"):
                out.append("\\n")
                continue
            out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_string = True
    return "".join(out)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "429",
            "rate limit",
            "resource exhausted",
            "too many requests",
            "quota",
        )
    )


def _check_mistral_rpd_limit():
    global _MISTRAL_DAY_KEY, _MISTRAL_DAY_COUNT
    if MISTRAL_RPD <= 0:
        return
    day_key = datetime.now(tz=timezone.utc).date().isoformat()
    if _MISTRAL_DAY_KEY != day_key:
        _MISTRAL_DAY_KEY = day_key
        _MISTRAL_DAY_COUNT = 0
    if _MISTRAL_DAY_COUNT >= MISTRAL_RPD:
        raise RuntimeError(
            "Mistral daily quota reached. Reduce requests or set MISTRAL_RPD for your tier."
        )
    _MISTRAL_DAY_COUNT += 1


def _throttle_mistral():
    global _MISTRAL_LAST_CALL
    if MISTRAL_RPM <= 0:
        raise RuntimeError(
            "Mistral RPM is set to 0. Set MISTRAL_RPM or update MISTRAL_MODEL."
        )
    _check_mistral_rpd_limit()
    now = time.monotonic()
    wait_for = MISTRAL_MIN_INTERVAL - (now - _MISTRAL_LAST_CALL)
    if wait_for > 0:
        time.sleep(wait_for)
    _MISTRAL_LAST_CALL = time.monotonic()


def _mistral_chat_completion(api_key: str, prompt: str) -> str:
    if requests is None:
        raise RuntimeError("Mistral summaries are not available. Install requests.")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an editor for an Indian daily news briefing app.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    response = requests.post(
        MISTRAL_API_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=90,
    )
    if not response.ok:
        raise RuntimeError(f"Mistral API error {response.status_code}: {response.text}")
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Mistral API response missing content.") from exc


def _mistral_generate_with_backoff(api_key: str, prompt: str, max_retries: int = 5):
    for attempt in range(max_retries):
        _throttle_mistral()
        try:
            return _mistral_chat_completion(api_key, prompt)
        except Exception as exc:
            if not _is_rate_limit_error(exc):
                raise
            if attempt == max_retries - 1:
                raise
            backoff = (2 ** attempt) + random.uniform(0.2, 0.8)
            time.sleep(backoff)


@st.cache_data(ttl=3600, show_spinner=False)
def summarize_articles_batch(
    articles: List[Dict],
    api_key: str,
    require_llm: bool = False,
) -> List[str]:
    if not articles:
        return []
    if not api_key or requests is None:
        if require_llm:
            raise RuntimeError(
                "Mistral summaries are not available. Check your API key and "
                "ensure requests is installed."
            )
        return [
            _fallback_summary(article.get("description", "") or article.get("title", ""))
            for article in articles
        ]

    items = []
    for idx, article in enumerate(articles, start=1):
        details = article.get("description", "") or article.get("title", "")
        items.append(
            {
                "id": idx,
                "title": article.get("title", ""),
                "details": details,
            }
        )

    try:
        payload = json.dumps(items, ensure_ascii=True)
        prompt = (
            "For each item, summarize the news article in exactly 3 concise lines in plain text, "
            "no bullets, no numbering. Each line should be <= 22 words and fact-focused. "
            "Return ONLY a JSON array of objects with keys 'id' and 'summary'. "
            "The 'summary' value must be a single string with line breaks as \\n separators. "
            "Keep the same order and ids.\n\n"
            f"Items: {payload}"
        )
        text = _mistral_generate_with_backoff(api_key, prompt)
        json_text = _normalize_json_payload(text)
        if not json_text:
            raise ValueError("Empty JSON payload from Mistral response.")
        data = json.loads(json_text)

        summaries_by_id = {}
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                entry_id = entry.get("id")
                summary = entry.get("summary", "")
                if isinstance(entry_id, int):
                    summaries_by_id[entry_id] = _clean_summary_text(summary)

        summaries = []
        for idx in range(1, len(articles) + 1):
            summary = summaries_by_id.get(idx, "").strip()
            if not summary:
                article = articles[idx - 1]
                summary = _fallback_summary(
                    article.get("description", "") or article.get("title", "")
                )
            summaries.append(summary)
        return summaries
    except Exception as exc:
        if require_llm:
            raise RuntimeError(
                f"Mistral batch summary failed. Please verify your API key and quota. ({exc})"
            ) from exc
        return [
            _fallback_summary(article.get("description", "") or article.get("title", ""))
            for article in articles
        ]


@st.cache_data(ttl=3600, show_spinner=False)
def summarize_article(
    title: str,
    description: str,
    api_key: str,
    require_llm: bool = False,
) -> str:
    details = description or title
    context = f"Title: {title}\nDetails: {details}"
    if not api_key or requests is None:
        if require_llm:
            raise RuntimeError(
                "Mistral summaries are not available. Check your API key and "
                "ensure requests is installed."
            )
        return _fallback_summary(description or title)

    try:
        prompt = (
            "Write exactly 3 concise lines in plain text, no bullets, no numbering. "
            "Each line should be <= 22 words and fact-focused.\n\n"
            f"{context}"
        )
        text = _mistral_generate_with_backoff(api_key, prompt)
        text = _clean_summary_text(text)
        return text if text else _fallback_summary(description or title)
    except Exception as exc:
        if require_llm:
            raise RuntimeError(
                f"Mistral summary failed. Please verify your API key and quota. ({exc})"
            ) from exc
        return _fallback_summary(description or title)


def build_newsletter_fallback(
    recipient_name: str,
    selected_articles: List[Dict],
    max_articles: int,
) -> str:
    lines = [
        f"Subject: Your India Briefing ({datetime.now().strftime('%d %b %Y')})",
        "",
        f"Hi {recipient_name},",
        "",
        "Here are your top updates today:",
        "",
    ]
    for idx, article in enumerate(selected_articles[:max_articles], start=1):
        lines.append(f"{idx}. {article['title']} ({article['category']}, {article['source']})")
        summary = (article.get("summary") or "").strip()
        if not summary:
            summary = _fallback_summary(
                article.get("description", "") or article.get("title", "")
            )
        for line in summary.splitlines():
            lines.append(f"   {line}")
        lines.append("")
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
    max_articles: int,
    force_llm: bool = True,
) -> str:
    if not api_key or requests is None:
        return build_newsletter_fallback(recipient_name, selected_articles, max_articles)

    try:
        batch_articles = selected_articles[:max_articles]
        summaries = summarize_articles_batch(
            batch_articles,
            api_key,
            require_llm=True,
        )
        enriched_articles = []
        for article, summary in zip(batch_articles, summaries):
            enriched = dict(article)
            enriched["summary"] = summary
            enriched_articles.append(enriched)
        return build_newsletter_fallback(recipient_name, enriched_articles, max_articles)
    except RuntimeError:
        raise
    except Exception:
        return build_newsletter_fallback(recipient_name, selected_articles, max_articles)


def render_article_card(article: Dict):
    st.markdown(f"### [{article['title']}]({article['link']})")
    st.caption(
        f"Source: {article['source']} | Published: {article.get('published', 'N/A')} | "
        f"Zero-shot confidence: {article['confidence']:.2f}"
    )
    summary = (article.get("summary") or "").strip()
    if not summary:
        summary = _fallback_summary(
            article.get("description", "") or article.get("title", "")
        )
    st.write(summary)


def main():
    st.title("India Daily Briefing")
    st.write(
        "Live RSS news from The Hindu, NDTV, and Indian Express with zero-shot category tagging and 3-line summaries."
    )

    with st.sidebar:
        st.header("Configuration")
        if "article_source" not in st.session_state:
            st.session_state.article_source = "rss"
        user_api_key = st.text_input(
            "Mistral API Key (optional)",
            value="",
            type="password",
            help="Optional. If left blank, the server environment key is used (if set).",
        ).strip()
        api_key = user_api_key or os.getenv("MISTRAL_AI_API_KEY", "").strip()
        article_limit = st.slider("Articles per refresh", min_value=10, max_value=50, value=30, step=5)
        st.subheader("News Source")
        uploaded_csv = st.file_uploader(
            "Local CSV file",
            type=["csv"],
            help="Choose a CSV file with headline/title columns to load local news.",
        )
        source_cols = st.columns(2)
        with source_cols[0]:
            if st.button("Use Live RSS"):
                st.session_state.article_source = "rss"
        with source_cols[1]:
            if st.button("Load Local CSV"):
                if uploaded_csv is None:
                    st.warning("Please choose a CSV file before loading.")
                else:
                    st.session_state.article_source = "csv"
        st.caption(
            f"Current source: {'Local CSV' if st.session_state.article_source == 'csv' else 'Live RSS'}"
        )
        if st.button("Refresh Feed"):
            st.cache_data.clear()
            st.rerun()

    if st.session_state.article_source == "csv":
        if uploaded_csv is None:
            st.error("Local CSV source selected but no file uploaded.")
            return
        with st.spinner("Loading articles from local CSV..."):
            try:
                articles = load_articles_from_csv(uploaded_csv, limit=article_limit)
            except Exception as exc:
                st.error(f"Failed to load CSV: {exc}")
                return
    else:
        with st.spinner("Fetching latest RSS articles..."):
            articles = fetch_latest_articles(limit=article_limit)

    if not articles:
        st.error("No articles fetched from RSS feeds. Please try again.")
        return

    classifier = load_classifier()
    category_config = resolve_categories(str(LABELS_PATH))
    category_labels = category_config["labels"]
    label_map = category_config["label_map"]
    tab_order = category_config["tab_order"]

    if category_config.get("source") == "default":
        st.sidebar.warning(
            "Labels file not found and L3Cube categories not available. "
            "Using default categories instead."
        )

    with st.spinner("Classifying articles with BART MNLI..."):
        articles = classify_articles(articles, classifier, category_labels, label_map)

    grouped = {k: [] for k in tab_order}
    for article in articles:
        grouped.setdefault(article["category"], []).append(article)

    tab_widgets = st.tabs(tab_order)
    for tab_name, tab in zip(tab_order, tab_widgets):
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
            options=tab_order,
            default=tab_order,
        )
    with col3:
        tone = st.selectbox("Tone", options=["Professional", "Friendly", "Concise"], index=0)

    selected_articles = [a for a in articles if a["category"] in selected_categories]

    if st.button("Generate Email Digest"):
        try:
            digest = build_newsletter_with_llm(
                selected_articles,
                api_key,
                recipient_name,
                tone,
                max_articles=article_limit,
                force_llm=True,
            )
        except RuntimeError as exc:
            st.warning(str(exc))
            digest = build_newsletter_fallback(
                recipient_name,
                selected_articles,
                max_articles=article_limit,
            )
        st.text_area("Generated Digest", value=digest, height=320)


if __name__ == "__main__":
    main()
