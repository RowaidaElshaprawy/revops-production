"""Interactive CLI: scrapes every un-scraped company, shows you the real
extracted features, and asks you to label it (1 = good lead, 0 = poor lead).
This is the human-in-the-loop step that makes training data real.

Usage:
    python -m scripts.label_companies
"""
from src.db.session import init_db, get_session
from src.db.models import Company, ScrapeResult, TrainingLabel
from src.agents.scraper_agent import scrape_company
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


def run():
    init_db()
    with get_session() as session:
        companies = session.query(Company).all()

    for company in companies:
        with get_session() as session:
            already_labeled = (
                session.query(TrainingLabel).filter_by(company_id=company.id).first()
            )
            if already_labeled:
                continue

            existing_scrape = (
                session.query(ScrapeResult)
                .filter_by(company_id=company.id, success=True)
                .first()
            )
            if not existing_scrape:
                print(f"\nScraping {company.domain} ...")
                result = scrape_company(company.domain)
                session.add(ScrapeResult(company_id=company.id, **result))
                session.flush()
                if not result["success"]:
                    print(f"  scrape failed: {result['error']} — skipping")
                    continue
                features = result
            else:
                features = {
                    "word_count": existing_scrape.word_count,
                    "pricing_page_found": existing_scrape.pricing_page_found,
                    "careers_page_found": existing_scrape.careers_page_found,
                    "keyword_density": existing_scrape.keyword_density,
                    "tech_signals": existing_scrape.tech_signals,
                }

            print(f"\n{company.domain}")
            print(f"  word_count={features['word_count']}  "
                  f"pricing_page={features['pricing_page_found']}  "
                  f"careers_page={features['careers_page_found']}  "
                  f"keyword_density={features['keyword_density']}  "
                  f"tech_signals={features['tech_signals']}")

            answer = input("  Label as good lead? [y/n/skip]: ").strip().lower()
            if answer == "y":
                session.add(TrainingLabel(company_id=company.id, label=1))
            elif answer == "n":
                session.add(TrainingLabel(company_id=company.id, label=0))
            # 'skip' or anything else: leave unlabeled


if __name__ == "__main__":
    run()
