import os
from dotenv import load_dotenv

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
IS_PRODUCTION = ENVIRONMENT in {"production", "prod"}

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "project_management")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
# Max attempts for a match job before it is marked permanently failed.
MATCH_JOB_MAX_TRIES = int(os.getenv("MATCH_JOB_MAX_TRIES", "3"))
# Default recall depth shared by sync matching and async matching jobs.
MATCH_RECALL_TOP_K = max(1, min(int(os.getenv("MATCH_RECALL_TOP_K", "1000")), 5000))
# Debounce owner-level learning-to-rank retraining after recruiter feedback/status changes.
RANKER_RETRAIN_DEBOUNCE_SECONDS = max(0, int(os.getenv("RANKER_RETRAIN_DEBOUNCE_SECONDS", "30")))

# Object storage (MinIO / S3) for raw CV PDFs. When unset, uploads still work but
# the raw PDF is not persisted (re-parse from raw becomes unavailable).
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "")
S3_BUCKET = os.getenv("S3_BUCKET", "lattice-cv-raw")
S3_SECURE = os.getenv("S3_SECURE", "false").strip().lower() in {"1", "true", "yes", "on"}
OBJECT_STORAGE_ENABLED = bool(S3_ENDPOINT and S3_ACCESS_KEY and S3_SECRET_KEY)
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
ENABLE_DEMO_MODE = os.getenv("ENABLE_DEMO_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_ANON_GUEST_MODE = os.getenv("ENABLE_ANON_GUEST_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
GUEST_SESSION_COOKIE_NAME = os.getenv("GUEST_SESSION_COOKIE_NAME", "lattice_guest")
GUEST_SESSION_DAYS = max(1, int(os.getenv("GUEST_SESSION_DAYS", "30")))
TURNSTILE_REQUIRED = os.getenv("TURNSTILE_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"}
TURNSTILE_SECRET_KEY = os.getenv("TURNSTILE_SECRET_KEY", "")

# Insecure default secrets that must never be used in production.
_INSECURE_SECRETS = {
    "",
    "dev-secret-key-change-in-production",
    "local-development-secret-change-me",
    "change-this-to-a-long-random-secret",
    "change-me",
}


def validate_runtime_config() -> list[str]:
    """Return fatal misconfigurations for the current environment.

    In development everything is permissive so the local demo keeps working.
    In production, insecure defaults are rejected so the app fails fast at startup
    instead of silently running with a bypassable auth layer.
    """
    problems: list[str] = []
    if IS_PRODUCTION:
        if JWT_SECRET_KEY in _INSECURE_SECRETS:
            problems.append(
                "JWT_SECRET_KEY is an insecure default. Set a long random secret in production."
            )
        elif len(JWT_SECRET_KEY) < 32:
            problems.append(
                "JWT_SECRET_KEY is too short (<32 chars). Use a long random secret in production."
            )
        if ENABLE_DEMO_MODE:
            problems.append(
                "ENABLE_DEMO_MODE must be false in production (demo mode bypasses authentication)."
            )
        if ENABLE_ANON_GUEST_MODE and TURNSTILE_REQUIRED and not TURNSTILE_SECRET_KEY:
            problems.append(
                "TURNSTILE_SECRET_KEY is required when anonymous scans require Turnstile."
            )
        if ENABLE_ANON_GUEST_MODE and not TURNSTILE_REQUIRED:
            problems.append(
                "TURNSTILE_REQUIRED must be true in production when anonymous guest scans are enabled."
            )
    return problems
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "25"))
MAX_REQUEST_BODY_BYTES = int(os.getenv("MAX_REQUEST_BODY_BYTES", str(MAX_UPLOAD_BYTES + 1024 * 1024)))
MAX_TEXT_INPUT_CHARS = int(os.getenv("MAX_TEXT_INPUT_CHARS", "30000"))


def comma_separated_env(name: str) -> list[str]:
    return [
        value.strip().rstrip("/")
        for value in os.getenv(name, "").split(",")
        if value.strip()
    ]


# CORS is deployment configuration: no frontend hostname is embedded in code.
FRONTEND_URLS = comma_separated_env("FRONTEND_URLS")
CORS_ALLOW_ORIGIN_REGEX = os.getenv("CORS_ALLOW_ORIGIN_REGEX", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL = os.getenv(
    "GEMINI_API_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
)

# --- Embeddings & vector search (Phase 3) ---
# Provider is swappable via env so the pipeline is not locked to one vendor.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini").strip().lower()  # gemini | hashing
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBEDDING_MODEL_VERSION = os.getenv(
    "EMBEDDING_MODEL_VERSION",
    GEMINI_EMBED_MODEL if EMBEDDING_PROVIDER == "gemini" else f"{EMBEDDING_PROVIDER}-v1",
).strip()
GEMINI_EMBED_URL = os.getenv(
    "GEMINI_EMBED_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent",
)
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "cv_embeddings")
QDRANT_COLLECTION_VERSIONED = os.getenv("QDRANT_COLLECTION_VERSIONED", "true").strip().lower() in {"1", "true", "yes", "on"}
VECTOR_SEARCH_ENABLED = bool(QDRANT_URL)

# --- Rate limiting (per client IP) ---
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
RATE_LIMIT_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "120/minute")
RATE_LIMIT_LLM = os.getenv("RATE_LIMIT_LLM", "20/minute")  # endpoints that call Gemini (cost)
RATE_LIMIT_MATCH = os.getenv("RATE_LIMIT_MATCH", "30/minute")  # embedding + heavy compute
RATE_LIMIT_AUTH = os.getenv("RATE_LIMIT_AUTH", "10/minute")  # brute-force protection on login/register
RATE_LIMIT_STATUS = os.getenv("RATE_LIMIT_STATUS", "60/minute")  # health/readiness/metrics probes

# Only trust forwarded client IP headers from known reverse proxies. Leave empty
# in local development so clients cannot spoof their rate-limit key with XFF.
TRUSTED_PROXY_IPS = set(comma_separated_env("TRUSTED_PROXY_IPS"))

# Distributed abuse controls. These are account/resource limits, so they still
# apply when traffic comes through many VPS/proxies.
MAX_CONCURRENT_UPLOADS_PER_USER = max(1, int(os.getenv("MAX_CONCURRENT_UPLOADS_PER_USER", "3")))
MAX_UPLOADS_PER_USER_PER_DAY = max(1, int(os.getenv("MAX_UPLOADS_PER_USER_PER_DAY", "100")))
MAX_CONCURRENT_UPLOADS_PER_GUEST = max(1, int(os.getenv("MAX_CONCURRENT_UPLOADS_PER_GUEST", "1")))
MAX_UPLOADS_PER_GUEST_PER_DAY = max(1, int(os.getenv("MAX_UPLOADS_PER_GUEST_PER_DAY", "3")))
MAX_UPLOADS_PER_IP_PER_DAY = max(1, int(os.getenv("MAX_UPLOADS_PER_IP_PER_DAY", "20")))
MAX_UPLOADS_PER_SUBNET_PER_DAY = max(1, int(os.getenv("MAX_UPLOADS_PER_SUBNET_PER_DAY", "60")))
MAX_GLOBAL_CONCURRENT_UPLOADS = max(1, int(os.getenv("MAX_GLOBAL_CONCURRENT_UPLOADS", "20")))
MAX_ANON_GLOBAL_CONCURRENT_UPLOADS = max(1, int(os.getenv("MAX_ANON_GLOBAL_CONCURRENT_UPLOADS", "5")))
UPLOAD_SLOT_LEASE_SECONDS = max(60, int(os.getenv("UPLOAD_SLOT_LEASE_SECONDS", "900")))
