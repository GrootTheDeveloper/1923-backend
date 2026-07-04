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
ENABLE_DEMO_MODE = os.getenv("ENABLE_DEMO_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}

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
        if ENABLE_DEMO_MODE:
            problems.append(
                "ENABLE_DEMO_MODE must be false in production (demo mode bypasses authentication)."
            )
    return problems
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "25"))
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
FRONTEND_URLS = [
    url.strip()
    for url in os.getenv("FRONTEND_URLS", FRONTEND_URL).split(",")
    if url.strip()
]
CORS_ALLOW_ORIGIN_REGEX = os.getenv(
    "CORS_ALLOW_ORIGIN_REGEX",
    r"^https://1923-frontend(?:-[a-z0-9]+)?(?:\.[a-z0-9-]+)?\.azurewebsites\.net$|^https://[a-z0-9-]+\.azurestaticapps\.net$",
)
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
GEMINI_EMBED_URL = os.getenv(
    "GEMINI_EMBED_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent",
)
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "cv_embeddings")
VECTOR_SEARCH_ENABLED = bool(QDRANT_URL)

# --- Rate limiting (per client IP) ---
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
RATE_LIMIT_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "120/minute")
RATE_LIMIT_LLM = os.getenv("RATE_LIMIT_LLM", "20/minute")  # endpoints that call Gemini (cost)
RATE_LIMIT_MATCH = os.getenv("RATE_LIMIT_MATCH", "30/minute")  # embedding + heavy compute
