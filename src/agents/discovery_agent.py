"""Company discovery agent.

IMPORTANT — why this doesn't scrape LinkedIn:
LinkedIn's Terms of Service explicitly prohibit automated scraping, and they
actively detect and block it (legal precedent: hiQ v. LinkedIn was about
public data, but LinkedIn has since won related cases and aggressively bans
scraper IPs/accounts). Shipping a LinkedIn scraper in something you show to
clients or employers is a real legal/reputational liability, not just a
technical inconvenience.

Instead this uses Clearbit's Autocomplete API — a free, public, unauthenticated,
documented endpoint designed exactly for "find companies by name/keyword".
It's the legitimate equivalent for this use case. For serious production use,
swap this for a paid, ToS-compliant provider with a real leads API:
Apollo.io, Clearbit Enrichment, ZoomInfo, or Crunchbase — all have proper
APIs meant to be called programmatically.
"""
import requests

from src.config import CLEARBIT_AUTOCOMPLETE_URL
from src.monitoring.logger import get_logger, timed

logger = get_logger(__name__)


@timed(logger)
def discover_companies(query: str, limit: int = 10) -> list[dict]:
    """Return a list of {name, domain, logo} dicts matching `query`.

    `query` can be a company name fragment or an industry-ish keyword —
    Clearbit's autocomplete matches loosely on name, so for real industry
    search you'd typically seed this with a list of known players and let
    discovery expand from there.
    """
    try:
        resp = requests.get(
            CLEARBIT_AUTOCOMPLETE_URL,
            params={"query": query},
            timeout=8,
        )
        resp.raise_for_status()
        results = resp.json()[:limit]
        return [
            {"name": r.get("name"), "domain": r.get("domain"), "logo": r.get("logo")}
            for r in results
            if r.get("domain")
        ]
    except requests.RequestException as e:
        logger.warning(f"discovery failed for query={query!r}: {e}")
        return []
