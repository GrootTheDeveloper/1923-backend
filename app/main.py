from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ALLOW_ORIGIN_REGEX, FRONTEND_URLS
from app.routes import auth, documents, projects, tasks

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
app.include_router(projects.router, prefix="/api/projects", tags=["Projects"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["Tasks"])


@app.get("/")
async def root():
    return {"message": "FARM CV-JD PDF Reader API is running"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}
