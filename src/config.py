"""Central configuration. All values come from environment variables so the
same code runs in dev (Codespaces), CI, and production (Streamlit Cloud /
Docker) without code changes.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Database — defaults to local SQLite for zero-setup dev, but reads DATABASE_URL
# so production can point at real Postgres (e.g. Railway, Supabase, RDS).
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/data/revops.db")

# API auth — a single shared secret is the minimum viable auth for a small
# internal tool. Swap for OAuth/JWT-per-user before opening this to external
# customers.
API_KEY = os.getenv("REVOPS_API_KEY", "")  # empty => auth disabled (dev only)

# Clearbit Autocomplete is a free, unauthenticated, public API for company
# discovery. It is NOT a scraper — it's a documented public endpoint.
CLEARBIT_AUTOCOMPLETE_URL = "https://autocomplete.clearbit.com/v1/companies/suggest"

# Scraper behavior
SCRAPER_TIMEOUT_SECONDS = int(os.getenv("SCRAPER_TIMEOUT_SECONDS", "10"))
SCRAPER_USER_AGENT = "RevOpsLeadScorer/1.0 (+contact: set-your-email-here)"

# Model
MODEL_PATH = os.getenv("MODEL_PATH", f"{BASE_DIR}/data/scoring_model.pt")
MIN_TRAINING_SAMPLES = int(os.getenv("MIN_TRAINING_SAMPLES", "15"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
