"""Baseline enrollment endpoint."""
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from datetime import datetime

from app import database, schemas
from app.services.auth import get_current_user
from app.services.voice_analysis import extract_biomarkers

router = APIRouter(prefix="/baseline", tags=["baseline"])


@router.post("", response_model=schemas.BaselineResponse)
async def create_baseline(
    audio: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """Upload a baseline voice recording. Extracts biomarkers and stores them."""
    audio_bytes = await audio.read()
    if len(audio_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Audio file too small")

    try:
        biomarkers = extract_biomarkers(audio_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {e}")

    baseline_id = database.save_baseline(user["id"], biomarkers)
    return schemas.BaselineResponse(
        baseline_id=baseline_id,
        biomarkers=schemas.Biomarkers(**biomarkers),
        created_at=datetime.utcnow(),
    )


@router.get("", response_model=schemas.BaselineResponse)
def get_baseline(user: dict = Depends(get_current_user)):
    baseline = database.get_baseline(user["id"])
    if not baseline:
        raise HTTPException(status_code=404, detail="No baseline recorded yet")
    return schemas.BaselineResponse(
        baseline_id=baseline["id"],
        biomarkers=schemas.Biomarkers(
            f0_hz=baseline["f0_hz"],
            jitter_pct=baseline["jitter_pct"],
            shimmer_pct=baseline["shimmer_pct"],
            hnr_db=baseline["hnr_db"],
        ),
        created_at=baseline["created_at"],
    )
