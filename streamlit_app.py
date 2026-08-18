import streamlit as st
import pandas as pd

from src.db.session import init_db, get_session
from src.db.models import Company, ScrapeResult, TrainingLabel, Score
from src.agents.discovery_agent import discover_companies
from src.agents.scraper_agent import scrape_company, features_to_vector
from src.config import MODEL_PATH

st.set_page_config(page_title="RevOps Lead Scoring", layout="wide")
init_db()

st.title("RevOps Lead Scoring — Production Dashboard")
st.caption(
    "Every score below comes from a real scrape of the company's live "
    "website, run through a model trained on real labeled examples — "
    "not fixed demo numbers."
)

tab_score, tab_discover, tab_label, tab_data = st.tabs(
    ["Score a company", "Discover companies", "Label training data", "Training data"]
)

with tab_score:
    domain = st.text_input("Company domain", placeholder="e.g. stripe.com")
    if st.button("Scrape + Score", type="primary") and domain:
        with st.spinner(f"Scraping {domain}..."):
            result = scrape_company(domain)

        if not result["success"]:
            st.error(f"Could not scrape {domain}: {result['error']}")
        else:
            st.success("Scraped successfully — real features below")
            cols = st.columns(4)
            cols[0].metric("Word count", result["word_count"])
            cols[1].metric("Pricing page", "Yes" if result["pricing_page_found"] else "No")
            cols[2].metric("Careers page", "Yes" if result["careers_page_found"] else "No")
            cols[3].metric("Tech signals", result["tech_signals"])

            try:
                import torch
                from src.ml.model import load_model
                model = load_model(MODEL_PATH)
                vector = features_to_vector(result)
                with torch.no_grad():
                    score = float(model(torch.tensor([vector], dtype=torch.float32)).item())
                st.metric("Lead score", f"{score:.2%}")
            except FileNotFoundError:
                st.warning(
                    "Model not trained yet. Label at least "
                    f"{15} companies in the 'Label training data' tab, then run: "
                    "`python -m src.ml.train_scoring_model`"
                )

with tab_discover:
    query = st.text_input("Search companies (name or keyword)", placeholder="e.g. project management")
    if st.button("Discover") and query:
        with st.spinner("Querying company database..."):
            companies = discover_companies(query, limit=10)
        if not companies:
            st.info("No results — try a different query.")
        else:
            with get_session() as session:
                for c in companies:
                    if not session.query(Company).filter_by(domain=c["domain"]).first():
                        session.add(Company(domain=c["domain"], name=c["name"], source="clearbit"))
            st.dataframe(pd.DataFrame(companies))
            st.success(f"Added {len(companies)} companies to the database for labeling.")

with tab_label:
    with get_session() as session:
        unlabeled = (
            session.query(Company)
            .outerjoin(TrainingLabel, Company.id == TrainingLabel.company_id)
            .filter(TrainingLabel.id.is_(None))
            .limit(20)
            .all()
        )
        unlabeled_domains = [c.domain for c in unlabeled]

    if not unlabeled_domains:
        st.info("No unlabeled companies. Discover some first, or all are labeled.")
    else:
        pick = st.selectbox("Company to label", unlabeled_domains)
        if st.button("Fetch features for labeling"):
            st.session_state["label_preview"] = scrape_company(pick)

        preview = st.session_state.get("label_preview")
        if preview:
            st.json(preview)
            c1, c2 = st.columns(2)
            if c1.button("Good lead (1)", type="primary"):
                with get_session() as session:
                    company = session.query(Company).filter_by(domain=pick).first()
                    session.add(ScrapeResult(company_id=company.id, **preview))
                    session.add(TrainingLabel(company_id=company.id, label=1))
                st.success(f"Labeled {pick} as good lead")
                del st.session_state["label_preview"]
                st.rerun()
            if c2.button("Poor lead (0)"):
                with get_session() as session:
                    company = session.query(Company).filter_by(domain=pick).first()
                    session.add(ScrapeResult(company_id=company.id, **preview))
                    session.add(TrainingLabel(company_id=company.id, label=0))
                st.success(f"Labeled {pick} as poor lead")
                del st.session_state["label_preview"]
                st.rerun()

with tab_data:
    with get_session() as session:
        labels = session.query(TrainingLabel).count()
        scrapes = session.query(ScrapeResult).filter_by(success=True).count()
        scores = session.query(Score).count()
    c1, c2, c3 = st.columns(3)
    c1.metric("Labeled companies", labels)
    c2.metric("Successful scrapes", scrapes)
    c3.metric("Scores computed", scores)
    st.caption(
        "Need at least 15 labeled companies before training the model. "
        "Run `python -m src.ml.train_scoring_model` from the terminal once you do."
    )
