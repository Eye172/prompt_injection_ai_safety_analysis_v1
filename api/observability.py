"""Structured request logging (spec 2.4 step 6).

Deliberately records only the decision — score, label, category, timestamp,
latency — and never the submitted text, which may contain a client's PII. That
is enough for drift monitoring; retraining on live traffic would need a
separate, consented collection path, not this log.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from .config import Settings
from .schemas import DetectionData

logger = logging.getLogger("api.detections")


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:20]}"


def log_detection(
    request_id: str,
    results: list[DetectionData],
    latency_ms: int,
    settings: Settings,
    *,
    endpoint: str,
    texts: list[str] | None = None,
) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "endpoint": endpoint,
        "latency_ms": latency_ms,
        "count": len(results),
        "detections": [
            {
                "score": r.score,
                "label": r.label.value,
                "language": r.language_used,
                "attack_type": r.attack_type.value if r.attack_type else None,
            }
            for r in results
        ],
    }

    if settings.log_request_text and texts is not None:
        # Off by default; only ever enabled knowingly, on traffic you own.
        record["texts"] = texts

    logger.info(json.dumps(record, ensure_ascii=False))
