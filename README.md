# Daily Briefing App (India News)

Streamlit app that fetches live Indian news from RSS, classifies each article into one of 5 categories with zero-shot NLP, and generates a 3-line summary per article.

## Features

- Live RSS aggregation from:
  - The Hindu
  - NDTV
  - Indian Express
- Zero-shot category classification using `facebook/bart-large-mnli`
- Category tabs:
  - Politics
  - Sports
  - Tech
  - Business
  - Entertainment
- 3-line article summaries using Mistral API (with fallback summarizer)
- Personalized newsletter generator (email style digest)
- Streamlit caching for faster repeated loads

## Project Structure

- `app.py`: Streamlit app (RSS + classifier + summary + newsletter)
- `requirements.txt`: Python dependencies
- `notebooks/training_and_ablation.ipynb`: training/evaluation notebook
- `report/report_template.md`: technical report draft scaffold
- `.streamlit/config.toml`: Streamlit runtime + theme config

## Setup

### 1) Create and activate environment

Use your preferred Python environment. If you already use a conda environment, activate it first.

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure Mistral API key (optional but recommended)

- Get an API key from the Mistral console.
- In the app sidebar, paste the key in `Mistral API Key`.
- Or set `MISTRAL_AI_API_KEY` in your environment.

Without a key, the app still works and uses a lightweight fallback summarizer.

### 4) Run app

```bash
streamlit run app.py
```

## Model Details

### Classification

- Model: `facebook/bart-large-mnli`
- Task: Zero-shot single-label classification
- Candidate labels:
  - Defaults to `Politics`, `Sports`, `Technology`, `Business`, `Entertainment`
  - If `labels.txt` exists, labels are loaded from that file
- Display mapping: `Technology -> Tech` (defaults only)

### Summarization

- Model/API: Mistral chat completions
- Prompt constraint: exactly 3 concise lines
- Fallback: sentence-based heuristic summarizer
- Batch mode: all selected articles are summarized in a single request for the email digest

## Caching

Implemented using Streamlit cache decorators:

- `@st.cache_data(ttl=900)`: RSS fetch
- `@st.cache_resource`: zero-shot model loading
- `@st.cache_data(ttl=3600)`: article summaries
- `@st.cache_data(ttl=1800)`: generated newsletter digest

## Training / Evaluation Notebook

Open `notebooks/training_and_ablation.ipynb` and run all cells.

Note: The Streamlit app does not use this dataset at runtime. It is only for offline evaluation and analysis.

Notebook includes:

- Dataset loading from Kaggle India headlines corpus
- Label mapping to app taxonomy
- Zero-shot evaluation (`classification_report`, confusion matrix)
- Ablation section (hypothesis template sensitivity)

Dataset source:

- https://www.kaggle.com/datasets/therohk/india-headlines-news-dataset

Place CSV at:

- `data/india_news_headlines.csv`

(adjust path/column names if your copy differs)

## Environment Variables

- `MISTRAL_AI_API_KEY`: API key for Mistral (used for summaries)
- `MISTRAL_MODEL`: model name (default: `mistral-small-latest`)
- `MISTRAL_API_URL`: override API base URL if needed
- `MISTRAL_RPM`: requests per minute limit for throttling (default: 10)
- `MISTRAL_RPD`: requests per day limit for throttling (default: 1000)

If you are using a different plan, update the RPM/RPD values to match your console limits.

## Deployment (Free)

## Option A: Streamlit Community Cloud

1. Push this project to GitHub.
2. Go to Streamlit Community Cloud.
3. Deploy from repo with main file `app.py`.
4. Add secret for API key if needed.

## Option B: HuggingFace Spaces (Streamlit)

1. Create new Streamlit Space.
2. Upload project files.
3. Ensure `requirements.txt` is present.
4. Set API key in Space secrets.

If hosting fails, provide a screen recording (<= 2 minutes) showing:

- App startup
- RSS article fetch
- Category tabs
- 3-line summaries
- Newsletter generation

## Deliverables Checklist

- [x] Streamlit app code
- [x] `requirements.txt`
- [x] Training/evaluation notebook
- [x] Report template
- [ ] Live deployed link OR <=2 min demo recording
- [ ] Final PDF report (6-8 pages, ICVGIP style)

## Notes and Limitations

- RSS feeds can occasionally be delayed or unavailable.
- Published timestamps from RSS providers may have inconsistent formats.
- Zero-shot classification is flexible but not as precise as task-specific fine-tuning.
- LLM summaries depend on API availability and prompt compliance.
