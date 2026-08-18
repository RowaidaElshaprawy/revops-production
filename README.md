# RevOps Lead Scoring — Production Rebuild

A real, working RevOps lead-scoring pipeline: discover companies → scrape
real signals from their public website → label with a human judgment →
train a model on that real data → score new companies through a real API,
with auth, a persistent database, and monitoring.

This replaces the earlier version, which had three critical gaps: manual
fixed numbers instead of real scraped features, a model trained on nothing,
and simulated output where a real result should be. Every one of those is
fixed here — see "What's real now" below.

## Architecture

```
Discovery (Clearbit API) → Company (DB) → Scraper (real HTTP + parsing)
    → ScrapeResult (DB) → Human label (DB) → Training (PyTorch, real data)
    → Model (.pt file) → Score endpoint (real inference on new scrapes)
```

- **API**: FastAPI (`src/api/main.py`) — auth-gated, logs every request to the DB
- **Dashboard**: Streamlit (`streamlit_app.py`) — same underlying DB/model
- **Database**: SQLAlchemy, SQLite by default, `DATABASE_URL` env var swaps to Postgres
- **Model**: small PyTorch MLP trained on real labeled scrapes, not synthetic data

## What's real now (vs. the old version)

| Before | Now |
|---|---|
| `raw_features=[0.85, 0.90, 0.80, 0.70]` manual | Features computed from actual page content: word count, pricing/careers links found, keyword density, tech-stack signals |
| Model trained on nothing / random weights | Trains only when >= 15 real labeled+scraped companies exist in the DB; refuses to run otherwise |
| `simulate_production.py` as the only "test" | 7 pytest tests hitting real code paths: scraper error handling, feature vector shape, discovery failure fallback, full API auth flow, end-to-end scrape→label→train |
| No persistence | Every scrape, label, and score is a row in a real database with timestamps |
| No auth | `X-API-Key` header required on all write endpoints once `REVOPS_API_KEY` is set |
| No deployment story beyond "push and hope" | Dockerfile + docker-compose for a real containerized deploy |

## What's still a placeholder — and why, honestly

- **`scripts/seed_companies.py`** ships with 8 well-known SaaS domains as a
  runnable example. These are NOT your real pipeline — replace them with
  companies you actually know the outcome for before training a model you'll
  trust.
- **LinkedIn was intentionally NOT scraped.** LinkedIn's ToS prohibits
  automated scraping and they actively enforce it — shipping that in
  something shown to clients/employers is a real legal liability, not a
  minor detail. `discovery_agent.py` uses Clearbit's free public Autocomplete
  API instead, which is legitimate for company discovery. For serious volume,
  swap in a paid compliant provider (Apollo.io, ZoomInfo, Crunchbase — all
  have real APIs meant for this).
- **Training data size.** 15-20 labeled companies proves the pipeline is
  real and mechanically correct (confirmed: loss drops from ~0.70 to ~0.09
  over 150 epochs on seeded data). It is not enough data for a model you'd
  trust to make real sales decisions — treat early scores as directional
  until you've labeled 100+ real companies.
- **Auth is a single shared API key**, appropriate for an internal tool or a
  portfolio demo you control access to. Before opening this to multiple
  external users, add per-user keys or OAuth.

## Setup

```bash
cp .env.example .env      # edit REVOPS_API_KEY before any public deploy
pip install -r requirements.txt
python -m pytest tests/ -v          # confirm the 7 tests pass in your environment
```

## Building real training data

```bash
python -m scripts.seed_companies      # or add your own companies via the API/dashboard
python -m scripts.label_companies     # interactive: scrapes + asks you to label each
python -m src.ml.train_scoring_model  # trains on whatever you've labeled (needs >= 15)
```

## Running

Locally (two processes):
```bash
uvicorn src.api.main:app --reload --port 8000
streamlit run streamlit_app.py
```

With Docker (closer to real production):
```bash
docker compose up --build
```
API on `:8000`, dashboard on `:8501`, both sharing one persistent SQLite volume.

## Deploying

- **Streamlit Cloud**: point it at `streamlit_app.py`, set `REVOPS_API_KEY`
  and `DATABASE_URL` (ideally a real Postgres instance — Streamlit Cloud's
  filesystem is ephemeral, so SQLite there will lose data on redeploy) as
  secrets in the app settings.
- **API**: any container host that takes a Dockerfile — Railway, Render,
  Fly.io, or a VM. Set the same env vars there.
- For real production, replace the SQLite default with a managed Postgres
  (`DATABASE_URL=postgresql://...`) so both services share real persistent
  state instead of a local file.

## API reference

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | none | uptime check |
| `POST /discover` | API key | find companies via Clearbit, add to DB |
| `POST /scrape` | API key | scrape one domain, store real features |
| `POST /label` | API key | attach a human 1/0 label to a company |
| `POST /score` | API key | scrape + run the trained model, store result |
