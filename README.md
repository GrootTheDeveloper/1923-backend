# CVMatch AI Backend

FastAPI backend for the CV-JD matching MVP. It keeps the existing auth endpoints and adds CV, JD, skill alias, and matching APIs.

## Local Run

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

Required environment values:

```text
MONGO_URI=mongodb://localhost:27017
DATABASE_NAME=project_management
JWT_SECRET_KEY=dev-secret-key-change-in-production
ENABLE_DEMO_MODE=true
MAX_UPLOAD_BYTES=10485760
MAX_PDF_PAGES=25
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
```

When `GEMINI_API_KEY` is set, CV/JD extraction uses Gemini structured JSON output first. If Gemini is unavailable or returns invalid data, the app falls back to the local rule-based extractor.

## Main API

- `POST /api/jobs`
- `GET /api/jobs`
- `PUT /api/jobs/{job_id}`
- `POST /api/cvs/upload`
- `GET /api/cvs`
- `PUT /api/cvs/{cv_id}/extracted-data`
- `POST /api/matches/run`
- `GET /api/matches?job_id=...`
- `PUT /api/matches/{match_id}/status`
- `GET /api/skills/aliases`

## Tests

```bash
python -m unittest discover -s tests
```
