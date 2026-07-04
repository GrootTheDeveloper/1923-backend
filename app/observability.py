"""Request metrics (Prometheus) + structured access logging.

Exposes counters and a latency histogram at /metrics (Prometheus text format)
and emits one JSON log line per request for structured log aggregation. Uses the
matched route template (e.g. /api/cvs/{cv_id}) as the label to avoid unbounded
cardinality from path parameters.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUESTS = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency (s)", ["method", "path"])


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", None) or request.url.path


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - start
            path = _route_template(request)
            if path != "/metrics":  # don't self-instrument scrapes
                REQUESTS.labels(request.method, path, status_code).inc()
                LATENCY.labels(request.method, path).observe(elapsed)
                print(json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "method": request.method,
                    "path": path,
                    "status": status_code,
                    "duration_ms": round(elapsed * 1000, 1),
                    "client": request.client.host if request.client else None,
                }))


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
