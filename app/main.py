from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ALLOW_ORIGIN_REGEX, FRONTEND_URLS
from app.routes import auth, cvs, demo, documents, jobs, matches, projects, skills, tasks

app = FastAPI(
    title="FARM CV-JD PDF Reader API",
    description="FastAPI service for extracting CV/JD text from PDF files and storing results in MongoDB.",
    version="1.0.0",
)

# CORS origins for local dev, local production preview, and deployed frontend.
default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "https://1923-frontend-eya6exfrhebxftgc.southeastasia-01.azurewebsites.net",
]
origins = sorted(set(default_origins + FRONTEND_URLS))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=CORS_ALLOW_ORIGIN_REGEX or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(cvs.router, prefix="/api/cvs", tags=["CVs"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["Jobs"])
app.include_router(matches.router, prefix="/api/matches", tags=["Matches"])
app.include_router(skills.router, prefix="/api/skills", tags=["Skills"])
app.include_router(demo.router, prefix="/api/demo", tags=["Demo"])
app.include_router(projects.router, prefix="/api/projects", tags=["Projects"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["Tasks"])


@app.get("/")
async def root():
    return {"message": "FARM CV-JD PDF Reader API is running"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


@app.on_event("startup")
async def startup_event():
    from app.database import create_indexes
    from app.services.skill_service import load_skill_aliases_from_db
    
    # 1. Create database indexes
    await create_indexes()
    
    # 2. Preload skill aliases cache from DB
    await load_skill_aliases_from_db()
