"""Real persistent schema. This replaces the old pattern of passing numbers
straight into a function call — every lead, every scrape, every score is
stored so the system has a real audit trail and a real training set.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True)
    domain = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=True)
    source = Column(String, nullable=False, default="manual")  # clearbit | manual | csv
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    scrapes = relationship("ScrapeResult", back_populates="company")
    scores = relationship("Score", back_populates="company")


class ScrapeResult(Base):
    """One row per scrape attempt. Keeping every attempt (including failures)
    is what makes feature extraction auditable instead of a black box.
    """
    __tablename__ = "scrape_results"

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    success = Column(Boolean, default=False)
    status_code = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)

    # Real extracted features (not manual numbers)
    word_count = Column(Integer, default=0)
    pricing_page_found = Column(Boolean, default=False)
    careers_page_found = Column(Boolean, default=False)
    has_https = Column(Boolean, default=False)
    keyword_density = Column(Float, default=0.0)  # sales/growth keyword hits per 1000 words
    tech_signals = Column(Integer, default=0)      # count of recognized tech/marketing stack hints
    raw_text_excerpt = Column(Text, nullable=True)  # first 500 chars, for debugging/audit

    scraped_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    company = relationship("Company", back_populates="scrapes")


class TrainingLabel(Base):
    """Ground-truth labels supplied by a human (you), used to train the model.
    This is what makes training 'real' instead of heuristic-only: an actual
    person is asserting which companies were good/bad leads.
    """
    __tablename__ = "training_labels"

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    label = Column(Integer, nullable=False)  # 1 = good lead, 0 = poor lead
    notes = Column(Text, nullable=True)
    labeled_by = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Score(Base):
    __tablename__ = "scores"

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    scrape_result_id = Column(Integer, ForeignKey("scrape_results.id"), nullable=True)
    score = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)
    scored_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    company = relationship("Company", back_populates="scores")


class RequestLog(Base):
    """Minimal monitoring: every API call logged to the DB so you can see
    usage/errors without standing up a separate observability stack yet.
    """
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True)
    endpoint = Column(String, nullable=False)
    status_code = Column(Integer, nullable=False)
    duration_ms = Column(Float, nullable=False)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
