from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

logger = logging.getLogger("squat_counter_api.request")

MAX_REQUEST_ID_LEN = 64


def _safe_request_id(raw: str | None) -> str:
    if not raw:
        return str(uuid.uuid4())
    candidate = raw[:MAX_REQUEST_ID_LEN]
    if not candidate.isascii() or not candidate.isprintable():
        return str(uuid.uuid4())
    return candidate


async def request_id_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    request_id = _safe_request_id(request.headers.get("x-request-id"))
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    duration_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "request completed",
        extra={
            "request_id": request_id,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response
