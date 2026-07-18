"""API-key authentication and per-caller rate limiting.

Auth is opt-in: with no keys configured the service runs open, which is what
the test suite and the local/Colab demo use. Setting PID_API_KEYS turns
enforcement on for every route depending on ``require_client``.
"""

from __future__ import annotations

import hmac

from fastapi import Depends, Header, HTTPException, Request, status

from ..config import Settings, get_settings
from .ratelimit import FixedWindowRateLimiter, RateLimitDecision


def _identity(api_key: str | None, request: Request) -> str:
    if api_key:
        # Never key limiter state on the raw secret.
        digest = hmac.new(b"pid-rl", api_key.encode(), "sha256").hexdigest()
        return f"key:{digest[:16]}"
    client = request.client.host if request.client else "unknown"
    return f"ip:{client}"


def _match(candidate: str, allowed: set[str]) -> bool:
    # Constant-time comparison so a valid prefix is not discoverable by timing.
    return any(hmac.compare_digest(candidate, k) for k in allowed)


class ClientContext:
    def __init__(self, identity: str, decision: RateLimitDecision) -> None:
        self.identity = identity
        self.decision = decision


def require_client(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> ClientContext:
    allowed = settings.parsed_api_keys()
    if allowed and (not x_api_key or not _match(x_api_key, allowed)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "invalid_api_key",
                "message": "Missing or invalid X-API-Key.",
            },
        )

    limiter: FixedWindowRateLimiter = request.app.state.limiter
    identity = _identity(x_api_key, request)
    decision = limiter.check(identity)
    if not decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "rate_limit_exceeded",
                "message": "Rate limit exceeded.",
                "retry_after_seconds": decision.reset_after,
            },
            headers={"Retry-After": str(decision.reset_after)},
        )
    return ClientContext(identity, decision)
