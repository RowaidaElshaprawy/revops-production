import time

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel

from src.auth.auth import require_api_key
from src.db.session import init_db, get_session
from src.db.models import Company, ScrapeResult, TrainingLabel, Score, RequestLog
from src.agents.discovery_agent import discover_companies
from src.agents.scraper_agent import scrape_company, features_to_vector
from src.config import MODEL_PATH
from src.monitoring.logger import get_logger

logger = get_logger(__name__)
app = FastAPI(title="RevOps Lead Scoring API", version="1.0.0")

MODEL_VERSION = "v1-real-data"
_model_cache = {}


@app.on_event("startup")
def startup():
    init_db()
    logger.info("database initialized")


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        with get_session() as session:
            session.add(RequestLog(
                endpoint=str(request.url.path),
                status_code=response.status_code,
                duration_ms=duration_ms,
            ))
        return response
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        with get_session() as session:
            session.add(RequestLog(
                endpoint=str(request.url.path), status_code=500,
                duration_ms=duration_ms, error=str(e),
            ))
        raise


@app.get("/health")
def health():
    """No auth required — used by uptime monitors / Streamlit Cloud health checks."""
    return {"status": "ok"}


class DiscoverRequest(BaseModel):
    query: str
    limit: int = 10


@app.post("/discover", dependencies=[Depends(require_api_key)])
def discover(req: DiscoverRequest):
    companies = discover_companies(req.query, req.limit)
    with get_session() as session:
        for c in companies:
            existing = session.query(Company).filter_by(domain=c["domain"]).first()
            if not existing:
                session.add(Company(domain=c["domain"], name=c["name"], source="clearbit"))
    return {"found": len(companies), "companies": companies}


class ScrapeRequest(BaseModel):
    domain: str


@app.post("/scrape", dependencies=[Depends(require_api_key)])
def scrape(req: ScrapeRequest):
    result = scrape_company(req.domain)
    with get_session() as session:
        company = session.query(Company).filter_by(domain=req.domain).first()
        if not company:
            company = Company(domain=req.domain, source="manual")
            session.add(company)
            session.flush()
        scrape_row = ScrapeResult(company_id=company.id, **result)
        session.add(scrape_row)
        session.flush()
        scrape_id = scrape_row.id
    return {"domain": req.domain, "scrape_id": scrape_id, **result}


class LabelRequest(BaseModel):
    domain: str
    label: int  # 1 or 0
    notes: str = ""


@app.post("/label", dependencies=[Depends(require_api_key)])
def label(req: LabelRequest):
    if req.label not in (0, 1):
        raise HTTPException(400, "label must be 0 or 1")
    with get_session() as session:
        company = session.query(Company).filter_by(domain=req.domain).first()
        if not company:
            raise HTTPException(404, f"company {req.domain} not found — scrape it first")
        session.add(TrainingLabel(company_id=company.id, label=req.label, notes=req.notes))
    return {"status": "labeled", "domain": req.domain, "label": req.label}


class ScoreRequest(BaseModel):
    domain: str


@app.post("/score", dependencies=[Depends(require_api_key)])
def score(req: ScoreRequest):
    if "model" not in _model_cache:
        try:
            from src.ml.model import load_model
            _model_cache["model"] = load_model(MODEL_PATH)
        except FileNotFoundError:
            raise HTTPException(
                503,
                "Model not trained yet. Run: python -m src.ml.train_scoring_model",
            )

    import torch

    result = scrape_company(req.domain)
    if not result["success"]:
        raise HTTPException(422, f"could not scrape {req.domain}: {result['error']}")

    vector = features_to_vector(result)
    with torch.no_grad():
        pred = _model_cache["model"](torch.tensor([vector], dtype=torch.float32))
        lead_score = float(pred.item())

    with get_session() as session:
        company = session.query(Company).filter_by(domain=req.domain).first()
        if not company:
            company = Company(domain=req.domain, source="manual")
            session.add(company)
            session.flush()
        scrape_row = ScrapeResult(company_id=company.id, **result)
        session.add(scrape_row)
        session.flush()
        session.add(Score(
            company_id=company.id, scrape_result_id=scrape_row.id,
            score=lead_score, model_version=MODEL_VERSION,
        ))

    return {"domain": req.domain, "score": round(lead_score, 4), "features": result}
