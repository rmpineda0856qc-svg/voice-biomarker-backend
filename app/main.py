"""FastAPI entry point for the Voice Biomarker AI backend."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import auth, baseline, assessment

settings = get_settings()

app = FastAPI(
    title="Voice Biomarker AI",
    description="Respiratory risk screening from voice + air quality data.",
    version="1.0.0",
)

# CORS - allow the mobile app (and browser during dev) to call the API
origins = settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(baseline.router)
app.include_router(assessment.router)


@app.get("/")
def root():
    return {
        "service": "Voice Biomarker AI",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
