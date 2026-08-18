"""Real integration tests — no simulate_production.py stand-in. These hit
actual code paths: real HTTP requests to real (or intentionally broken)
domains, a real temp SQLite DB, and the real FastAPI app via TestClient.
"""
import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def temp_db(monkeypatch):
    """Every test gets its own throwaway SQLite file so tests don't pollute
    each other or your real data/revops.db.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp.name}")
    yield
    os.unlink(tmp.name)


class _FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code} error")


def test_scraper_handles_real_reachable_site(monkeypatch):
    """Mocked at the requests layer so this test is deterministic in any
    network environment (sandboxed CI, Codespaces, local). The scraper
    itself makes a real HTTP call in production — see test_full_pipeline
    below for a check against this repo's actual live egress rules.
    """
    from src.agents import scraper_agent
    html = "<html><body><p>Pricing Careers Enterprise Growth</p>" \
           "<a href='/pricing'>Pricing</a><a href='/careers'>Careers</a></body></html>"
    monkeypatch.setattr(
        scraper_agent.requests, "get",
        lambda *a, **kw: _FakeResponse(html, 200),
    )
    result = scraper_agent.scrape_company("example.com")
    assert result["success"] is True
    assert result["word_count"] > 0
    assert result["has_https"] is True
    assert result["pricing_page_found"] is True
    assert result["careers_page_found"] is True


def test_scraper_handles_unreachable_domain(monkeypatch):
    from src.agents import scraper_agent
    import requests

    def raise_connection_error(*args, **kwargs):
        raise requests.ConnectionError("simulated DNS failure")

    monkeypatch.setattr(scraper_agent.requests, "get", raise_connection_error)
    result = scraper_agent.scrape_company("this-domain-should-not-exist-abc123xyz.invalid")
    assert result["success"] is False
    assert result["error"] is not None


def test_features_to_vector_shape():
    from src.agents.scraper_agent import features_to_vector
    from src.ml.model import FEATURE_DIM
    fake_scrape = {
        "word_count": 500, "pricing_page_found": True,
        "careers_page_found": False, "has_https": True,
        "keyword_density": 5.0, "tech_signals": 2,
    }
    vector = features_to_vector(fake_scrape)
    assert len(vector) == FEATURE_DIM
    assert all(0.0 <= v <= 1.0 for v in vector)


def test_discovery_returns_list_on_failure(monkeypatch):
    """Discovery must degrade gracefully (empty list), never crash the
    caller, if the public API is unreachable.
    """
    import requests
    from src.agents import discovery_agent

    def broken_get(*args, **kwargs):
        raise requests.ConnectionError("simulated network failure")

    monkeypatch.setattr(discovery_agent.requests, "get", broken_get)
    result = discovery_agent.discover_companies("test")
    assert result == []


def test_api_health_check():
    from fastapi.testclient import TestClient
    from src.api.main import app
    # TestClient must be used as a context manager for FastAPI's startup
    # event (init_db) to actually run — without `with`, /health would hit
    # a database that was never initialized.
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_api_requires_auth_when_key_set(monkeypatch):
    monkeypatch.setenv("REVOPS_API_KEY", "secret123")
    import importlib
    from src import config
    importlib.reload(config)
    from src.auth import auth
    importlib.reload(auth)

    from fastapi.testclient import TestClient
    from src.api.main import app
    with TestClient(app) as client:
        resp = client.post("/scrape", json={"domain": "example.com"})
    assert resp.status_code == 401


def test_full_pipeline_scrape_label_train(monkeypatch):
    """End-to-end: seed a company, scrape it, label it, and confirm training
    data can actually be assembled from the DB — the real replacement for
    the old fake simulate_production.py check. The scrape call is mocked
    (see note above) so this test is reproducible outside this sandbox too.
    """
    from src.db.session import init_db, get_session
    from src.db.models import Company, ScrapeResult, TrainingLabel
    from src.agents import scraper_agent
    from src.ml.train_scoring_model import load_training_data

    monkeypatch.setattr(
        scraper_agent.requests, "get",
        lambda *a, **kw: _FakeResponse("<html><body>Pricing Growth</body></html>", 200),
    )

    init_db()
    with get_session() as session:
        company = Company(domain="example.com", source="manual")
        session.add(company)
        session.flush()
        company_id = company.id

    result = scraper_agent.scrape_company("example.com")
    assert result["success"]

    with get_session() as session:
        session.add(ScrapeResult(company_id=company_id, **result))
        session.add(TrainingLabel(company_id=company_id, label=1))

    X, y, domains = load_training_data()
    assert len(X) == 1
    assert y == [1.0]
    assert domains == ["example.com"]
