# Deploying the dashboard to Hugging Face Spaces

The Streamlit dashboard (`streamlit_app.py`) is self-contained — it reads pre-generated CSVs and
a saved XGBoost model directly, with no separate API server needed, which makes it the simplest
thing to host live. `deploy/huggingface_space/` holds the two files a Space needs that this repo
doesn't otherwise have (a frontmatter-tagged `README.md` and a Streamlit-specific
`requirements.txt` — the main repo's `requirements.txt` is intentionally the lean FastAPI-only
set, not this one).

## Steps

1. **Create the Space** (huggingface.co account required):
   - Go to https://huggingface.co/new-space
   - Name it (e.g. `delivery-risk-forecasting`), SDK: **Streamlit**, hardware: free CPU, visibility: Public
   - This gives you a git remote: `https://huggingface.co/spaces/<your-username>/<space-name>`

2. **Clone the new (empty) Space repo** somewhere separate from this project:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name> hf-space-deploy
   cd hf-space-deploy
   ```

3. **Copy in the files the Space needs** (from this repo):
   ```bash
   cp <this-repo>/streamlit_app.py .
   cp -r <this-repo>/src .
   mkdir -p data/raw
   cp <this-repo>/data/raw/delay_timeseries.csv data/raw/
   cp <this-repo>/data/raw/baseline_predictions.csv data/raw/
   cp <this-repo>/data/raw/tft_predictions.csv data/raw/
   mkdir -p models
   cp <this-repo>/models/*.json models/
   cp <this-repo>/deploy/huggingface_space/README.md .
   cp <this-repo>/deploy/huggingface_space/requirements.txt .
   ```
   (If any of `data/raw/*.csv` or `models/*.json` don't exist locally, regenerate them first —
   see the main README's Setup section.)

4. **Push:**
   ```bash
   git add .
   git commit -m "Initial deploy"
   git push
   ```
   The Space builds automatically (a couple minutes) and gives you a public URL:
   `https://huggingface.co/spaces/<your-username>/<space-name>`

5. Add that URL to `case_study.md` section 9 (Deployment) once it's live.

## Why the generated files are committed here but not in the main repo

`data/raw/` and `models/` are gitignored in the main repo on purpose (they're deterministically
regenerable, and the point of that repo is the generation/training code, not the output files).
The Space repo is a different thing — a deployment target, not a portfolio source repo — so
committing the small pre-computed artifacts there (a few MB total) avoids the Space needing
torch/prophet installed just to regenerate them on every cold start.
