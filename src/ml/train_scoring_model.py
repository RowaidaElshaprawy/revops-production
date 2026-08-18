"""Trains the scoring model on REAL labeled data pulled from the database —
not synthetic/heuristic numbers. Run this after you've:
  1. Added companies (scripts/seed_companies.py or the API)
  2. Scraped them (src/agents/scraper_agent.py, called via the API or a batch script)
  3. Labeled them (scripts/label_companies.py) — a human says 1 (good lead) or 0 (poor lead)

Usage:
    python -m src.ml.train_scoring_model
"""
import sys

import torch
import torch.nn as nn

from src.config import MODEL_PATH, MIN_TRAINING_SAMPLES
from src.db.session import get_session
from src.db.models import Company, ScrapeResult, TrainingLabel
from src.agents.scraper_agent import features_to_vector
from src.ml.model import LeadScoringModel
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


def load_training_data():
    """Join each labeled company to its most recent successful scrape and
    build (X, y). Companies with no successful scrape are skipped — you
    can't train on a feature vector that doesn't exist.
    """
    X, y, domains = [], [], []
    with get_session() as session:
        labels = session.query(TrainingLabel).all()
        for label in labels:
            company = session.get(Company, label.company_id)
            scrape = (
                session.query(ScrapeResult)
                .filter_by(company_id=company.id, success=True)
                .order_by(ScrapeResult.scraped_at.desc())
                .first()
            )
            if not scrape:
                logger.warning(f"skipping {company.domain}: no successful scrape yet")
                continue
            X.append(features_to_vector({
                "word_count": scrape.word_count,
                "pricing_page_found": scrape.pricing_page_found,
                "careers_page_found": scrape.careers_page_found,
                "has_https": scrape.has_https,
                "keyword_density": scrape.keyword_density,
                "tech_signals": scrape.tech_signals,
            }))
            y.append(float(label.label))
            domains.append(company.domain)
    return X, y, domains


def train(epochs: int = 150, lr: float = 0.01):
    X, y, domains = load_training_data()

    if len(X) < MIN_TRAINING_SAMPLES:
        logger.error(
            f"Only {len(X)} labeled+scraped companies found "
            f"(need >= {MIN_TRAINING_SAMPLES}). "
            f"Run scripts/seed_companies.py and scripts/label_companies.py first."
        )
        sys.exit(1)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    model = LeadScoringModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    logger.info(f"training on {len(X)} real labeled companies: {domains}")

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = criterion(preds, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0 or epoch == epochs - 1:
            logger.info(f"epoch {epoch}: loss={loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"saved model to {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train()
