from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import FRONTEND_URL
from app.routes import auth, projects, tasks

app = FastAPI(
    title="Project Management API",
    description="API quản lý dự án và công việc",
    version="1.0.0",
)

# CORS — cho phép frontend gọi API
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    FRONTEND_URL,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
