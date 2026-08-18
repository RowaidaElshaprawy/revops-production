"""Minimal but real auth: a shared API key checked via FastAPI dependency
injection. This is intentionally simple — it's the right amount of auth for
a small internal/portfolio tool. Before selling this as multi-tenant SaaS,
replace with per-user API keys (a `users` table + hashed keys) or OAuth.
"""
from fastapi import Header, HTTPException, status

from src.config import API_KEY


async def require_api_key(x_api_key: str = Header(default="")):
    if not API_KEY:
        # Auth disabled — dev mode only. Fails loudly in logs so it's never
        # silently left off in production.
        return True
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )
    return True
