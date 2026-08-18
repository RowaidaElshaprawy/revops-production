"""Seeds the database with a real starting set of companies to label and
train on. Edit STARTER_COMPANIES with domains you actually know the outcome
for (real customers, real prospects that went cold, etc.) — the model is
only as real as these labels.

Usage:
    python -m scripts.seed_companies
"""
from src.db.session import init_db, get_session
from src.db.models import Company
from src.monitoring.logger import get_logger

logger = get_logger(__name__)

# Real Egyptian market companies — publicly known startups/scale-ups across
# fintech, e-commerce, mobility, and healthtech. These are real domains;
# labeling them (good lead / poor lead) is a judgment call only you can
# make based on your actual sales/RevOps context — this script deliberately
# does NOT pre-label them.
STARTER_COMPANIES = [
    "fawry.com",          # fintech / payments
    "vezeeta.com",        # healthtech
    "swvl.com",           # mobility
    "maxab.com",          # B2B e-commerce
    "trella.app",         # logistics
    "breadfast.com",      # quick commerce
    "instabug.com",       # dev tools (Egyptian-founded, global SaaS)
    "yodawy.com",         # healthtech / pharmacy
    "elmenus.com",        # food delivery / restaurant tech
    "nowpay.cash",        # fintech / earned wage access
    "khazna.com",         # fintech
    "moneyfellows.com",   # fintech
]



def seed():
    init_db()
    added = 0
    with get_session() as session:
        for domain in STARTER_COMPANIES:
            if not session.query(Company).filter_by(domain=domain).first():
                session.add(Company(domain=domain, source="manual"))
                added += 1
    logger.info(f"seeded {added} companies ({len(STARTER_COMPANIES) - added} already existed)")


if __name__ == "__main__":
    seed()
