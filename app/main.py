from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import CORS_ALLOW_ORIGIN_REGEX, FRONTEND_URLS
from app.routes import auth, projects, tasks

app = FastAPI(
    title="Project Management API",
    description="API quản lý dự án và công việc",
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
app.include_router(projects.router, prefix="/api/projects", tags=["Projects"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["Tasks"])


@app.get("/")
async def root():
    return {"message": "Project Management API is running"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}
