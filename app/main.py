import asyncio

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config import (
    CORS_ALLOW_ORIGIN_REGEX,
    ENABLE_DEMO_MODE,
    FRONTEND_URLS,
    IS_PRODUCTION,
    MAX_REQUEST_BODY_BYTES,
    RATE_LIMIT_STATUS,
    validate_runtime_config,
)
from app.observability import ObservabilityMiddleware, metrics_response
from app.rate_limit import limiter
from app.services.guest_session import attach_guest_cookie_if_needed
from app.routes import analytics, auth, cvs, demo, documents, jobs, matches, platform, skills

app = FastAPI(
    title="Lattice Recruitment Matching API",
    description="Hybrid search, evidence-based ranking, fairness monitoring, and human review for CV/JD matching.",
    version="2.0.0",
)

# Exact origins and any optional preview-origin regex come from the environment.
origins = sorted(set(FRONTEND_URLS))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=CORS_ALLOW_ORIGIN_REGEX or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-IP rate limiting (protects Gemini/embedding-backed endpoints).
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}. Please slow down."},
    )


app.add_middleware(SlowAPIMiddleware)
# Request metrics + structured access logging.
app.add_middleware(ObservabilityMiddleware)


@app.middleware("http")
async def request_size_guard(request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
        if size > MAX_REQUEST_BODY_BYTES:
            max_mb = MAX_REQUEST_BODY_BYTES / (1024 * 1024)
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body is too large. Maximum allowed size is {max_mb:.0f} MB."},
            )
    return await call_next(request)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
    if IS_PRODUCTION:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    attach_guest_cookie_if_needed(request, response)
    return response

# Register routes
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(cvs.router, prefix="/api/cvs", tags=["CVs"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["Jobs"])
app.include_router(matches.router, prefix="/api/matches", tags=["Matches"])
app.include_router(platform.router, prefix="/api", tags=["Recruitment Platform"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(skills.router, prefix="/api/skills", tags=["Skills"])
app.include_router(demo.router, prefix="/api/demo", tags=["Demo"])


@app.get("/")
@limiter.limit(RATE_LIMIT_STATUS)
async def root(request: Request):
    return {"message": "FARM CV-JD PDF Reader API is running"}


@app.get("/api/health")
@limiter.limit(RATE_LIMIT_STATUS)
async def health_check(request: Request):
    return {"status": "healthy"}


@app.get("/metrics")
@limiter.limit(RATE_LIMIT_STATUS)
async def metrics(request: Request):
    return metrics_response()


@app.get("/api/ready")
@limiter.limit(RATE_LIMIT_STATUS)
async def readiness(request):
    """Deep readiness: verify each backing service is actually reachable.
    Returns 503 if the core store (Mongo) is down."""
    from app.database import database
    from app.services import object_storage, vector_store

    deps: dict[str, str] = {}

    async def check(name: str, coro) -> None:
        try:
            await asyncio.wait_for(coro, timeout=2.5)
            deps[name] = "ok"
        except Exception as exc:  # noqa: BLE001
            deps[name] = f"down: {type(exc).__name__}"

    async def redis_ping() -> None:
        from app.queue import get_arq_pool

        pool = await get_arq_pool()
        if pool is None:
            raise RuntimeError("redis unavailable")
        await pool.ping()

    await check("mongo", database.client.admin.command("ping"))
    await check("redis", redis_ping())
    await check("qdrant", vector_store.ping())
    await check("minio", object_storage.ping())

    ready = deps.get("mongo") == "ok"
    return JSONResponse({"ready": ready, "dependencies": deps}, status_code=200 if ready else 503)


@app.on_event("startup")
async def startup_event():
    from app.database import create_indexes
    from app.services.skill_service import load_skill_aliases_from_db

    # 0. Fail fast on insecure production configuration.
    problems = validate_runtime_config()
    if problems:
        raise RuntimeError(
            "Insecure production configuration:\n- " + "\n- ".join(problems)
        )
    if not IS_PRODUCTION and ENABLE_DEMO_MODE:
        print("[WARN] ENABLE_DEMO_MODE is on: authentication is bypassed. Do not use in production.")

    # 1. Create database indexes
    await create_indexes()

    # 2. Preload skill aliases cache from DB
    await load_skill_aliases_from_db()


@app.on_event("shutdown")
async def shutdown_event():
    from app.queue import close_arq_pool

    await close_arq_pool()