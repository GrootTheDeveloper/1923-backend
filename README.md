# Lattice Recruitment Matching API

FastAPI backend for CV ingestion, JD extraction, hybrid candidate retrieval, evidence-backed scoring, asynchronous match jobs, recruiter feedback, and fairness monitoring.

## Run locally

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Configuration is loaded from the environment. Start from the repository-level `.env.example`. Important controls include `MONGO_URI`, `DATABASE_NAME`, `JWT_SECRET_KEY`, `ENABLE_DEMO_MODE`, `FRONTEND_URLS`, `CORS_ALLOW_ORIGIN_REGEX`, provider credentials, and storage/vector URLs.

`FRONTEND_URLS` is the complete exact-origin CORS allowlist. The application contains no deployed frontend hostname. In demo mode, an omitted bearer token maps to the demo owner; a supplied invalid or expired token is always rejected with HTTP 401.

## Main API groups

- `/api/cvs`: upload, inspect, reparse, update, and remove CVs.
- `/api/jobs`: create and version job requirements.
- `/api/matches`: synchronous matching and recruiter status changes.
- `/api/match-jobs`: asynchronous matching lifecycle.
- `/api/feedback` and `/api/analytics`: learning signals and monitoring.
- `/api/skills`: normalized skill aliases.
- `/api/auth`: owner authentication.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`tests/test_matching_characterization.py` locks representative current outputs. `tests/test_ranking_service.py` covers the learning-to-rank hold-out gate.

## Learning-to-rank guardrails

LTR models use a versioned feature schema. Active models with an incompatible schema are skipped, so matching falls back to the heuristic priority proxy instead of returning a misleading zero score. The heuristic proxy is not blended into `final_score`; only a compatible learned model can contribute the `ml_rank` term.

Training reserves a deterministic stratified hold-out set and activates a candidate model only when out-of-sample NDCG or MAP lift is positive. Feedback can include `displayed_rank`; this is stored for position-bias audits because labels collected from ranked lists may reflect exposure, not only candidate quality.

## Vector and semantic scoring guardrails

Qdrant CV vectors are isolated by embedding provider, model version, and dimension. By default the active collection name is suffixed with that index version, and each point payload stores the same metadata. Changing `EMBEDDING_DIM`, `EMBEDDING_PROVIDER`, or `EMBEDDING_MODEL_VERSION` therefore creates a fresh index path instead of mixing incompatible vectors; re-index CVs after such changes to populate the new collection.

Semantic scores are calibrated before blending. Raw embedding cosine and lexical TF-cosine have different natural distributions, so `score_breakdown` stores both the raw semantic score/source and the calibrated `semantic_score` used by final scoring.

## CV hash migration

Before enabling the unique `(owner_id, file_hash)` index on an existing database, inspect duplicates:

```powershell
.\.venv\Scripts\python.exe scripts/migrate_cv_file_hash_index.py
```

After reviewing the report, apply the non-destructive migration:

```powershell
.\.venv\Scripts\python.exe scripts/migrate_cv_file_hash_index.py --apply
```

Older duplicate records are preserved, linked to the newest canonical CV, and have only `file_hash` unset. Startup then replaces the legacy non-unique index with the partial unique index.
