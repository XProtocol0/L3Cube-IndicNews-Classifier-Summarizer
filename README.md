# Daily Briefing App (India News)

Streamlit app that fetches live Indian news from RSS, classifies each article into one of six categories with zero-shot NLP, and generates a 3-line summary per article.

## Features

- Live RSS aggregation from:
  - The Hindu
  - NDTV
  - Indian Express
- Zero-shot category classification using `facebook/bart-large-mnli`
- Category tabs:
  - Business
  - Crime
  - Entertainment
  - Politics
  - Sports
  - Technology
- (6 Major Common Classes included if want to run for L3Cube Categories provided in L3Cube_labels.txt)
- 3-line article summaries using Mistral API (with fallback summarizer)
- Personalized newsletter generator (email style digest)
- Local CSV upload mode with flexible column mapping
- Streamlit caching for faster repeated loads

## Project Structure

- `app.py`: Streamlit app (RSS + classifier + summary + newsletter)
- `requirements.txt`: Python dependencies
- `notebooks/training_and_ablation.ipynb`: training/evaluation notebook
- `report/report_final.pdf`: final report
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
  - If `labels.txt` exists, labels are loaded from that file (default: Business, Crime, Entertainment, Politics, Sports, Technology)
  - If `labels.txt` is missing, the app falls back to 5 default categories
- Display mapping: `Technology -> Tech` (defaults only)

### Summarization

- Model/API: Mistral chat completions
- Prompt constraint: exactly 3 concise lines
- Fallback: sentence-based heuristic summarizer
- Batch mode: all selected articles are summarized in a single request for the email digest
- Rate limiting: per-minute throttling and per-day quota guards with exponential backoff

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

## Results (from report_final.pdf)

- Evaluation set: 5,471 headlines filtered to six categories
- Baseline (template: "This headline is about {}.")
  - Accuracy: 0.7044
  - Macro-F1: 0.6903
- Best prompt template: "This Indian news headline covers {}." (accuracy 0.7428)

## Environment Variables

- `MISTRAL_AI_API_KEY`: API key for Mistral (used for summaries)
- `MISTRAL_MODEL`: model name (default: `mistral-small-latest`)
- `MISTRAL_API_URL`: override API base URL if needed
- `MISTRAL_RPM`: requests per minute limit for throttling (default: 10)
- `MISTRAL_RPD`: requests per day limit for throttling (default: 1000)

If you are using a different plan, update the RPM/RPD values to match your console limits.



## Notes and Limitations

- RSS feeds can occasionally be delayed or unavailable.
- Published timestamps from RSS providers may have inconsistent formats.
- Zero-shot classification is flexible but not as precise as task-specific fine-tuning.
- LLM summaries depend on API availability and prompt compliance.

## NOTE:
The L3Cube-IndicNews-style taxonomy contains 36 category labels. Because the app fetches only ~30 articles per refresh, most categories are empty; for clarity, the screenshots show only the 6 most common labels. To classify against all 36 labels, add the full label list to `labels.txt`. The demo video includes the full label set.

## Assumptions

- Single-label classification is used (argmax over the label set), not multi-label.
- Headlines are classified using title + short RSS/CSV description, not full article bodies.
- Articles outside the label set are forced into the closest available category.

