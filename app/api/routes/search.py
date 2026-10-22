"""Face-similarity search over one album."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi import status as http_status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_joined_event
from app.config import settings
from app.controllers import search_controller
from app.models.event import Event
from app.schemas.search import SearchResponse
from app.services import search_service
from app.utils import file_utils

router = APIRouter(tags=["search"])


@router.post("/events/{event_id}/search", response_model=SearchResponse)
async def search_event(
    selfie: UploadFile = File(..., description="Reference selfie (one person)."),
    threshold: float | None = Query(
        None, ge=-1.0, le=1.0,
        description="Minimum cosine similarity; defaults to the configured value.",
    ),
    top_k: int | None = Query(
        None, ge=1, le=settings.SEARCH_MAX_TOP_K,
        description="Maximum number of photos to return.",
    ),
    db: Session = Depends(get_db),
    event: Event = Depends(get_joined_event),
):
    """Detect + embed the selfie and rank this event's photos against it.

    Available to the owner and every joined member.  The selfie is
    deleted from public/search/ right after embedding.
    """
    try:
        return await search_controller.search(
            db, event, selfie, threshold=threshold, top_k=top_k
        )
    except file_utils.UnsupportedFileType as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except file_utils.UploadTooLarge as exc:
        raise HTTPException(
            status_code=http_status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        )
    except search_service.UnreadableSelfie:
        raise HTTPException(status_code=400, detail="The uploaded selfie could not be read as an image.")
    except search_service.NoFaceInSelfie:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No face detected in the uploaded selfie. Please upload a clear, front-facing photo.",
        )
