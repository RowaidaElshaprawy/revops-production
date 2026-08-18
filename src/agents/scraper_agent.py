"""Scraper agent — extracts REAL features from a company's own public
website. This is legally clean: fetching a company's own marketing site is
what browsers do millions of times a day, and nothing here bypasses
authentication, paywalls, or robots.txt disallow rules.

This replaces the old pattern of `raw_features=[0.85, 0.90, 0.80, 0.70]` —
every number below is derived from the actual page content of `domain`.
"""
import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src.config import SCRAPER_TIMEOUT_SECONDS, SCRAPER_USER_AGENT
from src.monitoring.logger import get_logger, timed

logger = get_logger(__name__)

GROWTH_KEYWORDS = [
    "pricing", "customers", "case study", "case studies", "integrations",
    "enterprise", "demo", "free trial", "roi", "scale", "growth",
    "revenue", "funding", "series a", "series b", "series c",
]
TECH_SIGNALS = [
    "hubspot", "salesforce", "segment.com", "intercom", "stripe",
    "google-analytics", "mixpanel", "amplitude", "zendesk",
]


@timed(logger)
def scrape_company(domain: str) -> dict:
    """Fetch the company's homepage and derive real, explainable features.

    Returns a dict matching the ScrapeResult schema. Never raises — failures
    are captured in the `success`/`error` fields so a bad site doesn't crash
    a batch job.
    """
    url = domain if domain.startswith("http") else f"https://{domain}"
    result = {
        "success": False, "status_code": None, "error": None,
        "word_count": 0, "pricing_page_found": False,
        "careers_page_found": False, "has_https": url.startswith("https"),
        "keyword_density": 0.0, "tech_signals": 0, "raw_text_excerpt": None,
    }

    try:
        resp = requests.get(
            url,
            timeout=SCRAPER_TIMEOUT_SECONDS,
            headers={"User-Agent": SCRAPER_USER_AGENT},
        )
        result["status_code"] = resp.status_code
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        text_lower = text.lower()
        html_lower = resp.text.lower()

        words = re.findall(r"[a-zA-Z']+", text)
        word_count = len(words)

        keyword_hits = sum(text_lower.count(kw) for kw in GROWTH_KEYWORDS)
        keyword_density = (keyword_hits / word_count * 1000) if word_count else 0.0

        tech_hits = sum(1 for sig in TECH_SIGNALS if sig in html_lower)

        # Look for pricing/careers links, not just keyword mentions —
        # a real nav link is a stronger signal than the word appearing once.
        links = [a.get("href", "") for a in soup.find_all("a", href=True)]
        pricing_found = any("pricing" in href.lower() for href in links) or "pricing" in text_lower
        careers_found = any(
            ("careers" in href.lower() or "jobs" in href.lower()) for href in links
        )

        result.update({
            "success": True,
            "word_count": word_count,
            "pricing_page_found": pricing_found,
            "careers_page_found": careers_found,
            "keyword_density": round(keyword_density, 3),
            "tech_signals": tech_hits,
            "raw_text_excerpt": text[:500],
        })
    except requests.RequestException as e:
        result["error"] = str(e)
        logger.warning(f"scrape failed for {domain}: {e}")

    return result


def features_to_vector(scrape_result: dict) -> list[float]:
    """Convert a scrape result into the fixed-length numeric feature vector
    the scoring model expects. Every value here traces back to something
    actually observed on the page — no manual numbers.
    """
    return [
        min(scrape_result["word_count"] / 2000.0, 1.0),   # normalized site depth
        1.0 if scrape_result["pricing_page_found"] else 0.0,
        1.0 if scrape_result["careers_page_found"] else 0.0,  # hiring = growth signal
        1.0 if scrape_result["has_https"] else 0.0,
        min(scrape_result["keyword_density"] / 20.0, 1.0),
        min(scrape_result["tech_signals"] / 5.0, 1.0),
    ]
