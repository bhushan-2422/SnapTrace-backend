"""Photo upload, listing and deletion."""

from __future__ import annotations

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.models.event import Event
from app.models.image import Image, ImageStatus
from app.schemas.image import ImageResponse, ImageUploadResponse
from app.schemas.processing import FaceRecord
from app.repositories.face_repository import FaceRepository
from app.services import image_service


async def upload(
    db: Session, event: Event, files: list[UploadFile], *, uploaded_by: int
) -> ImageUploadResponse:
    """Store the photos on disk, register them as `uploaded`, queue them."""
    images = await image_service.store_uploads(db, event, files, uploaded_by=uploaded_by)
    return ImageUploadResponse(
        event_id=event.id,
        accepted=len(images),
        image_ids=[image.id for image in images],
    )


def list_by_event(
    db: Session,
    event: Event,
    *,
    status: ImageStatus | None,
    limit: int,
    offset: int,
) -> list[ImageResponse]:
    images = image_service.list_images(
        db, event.id, status=status, limit=limit, offset=offset
    )
    return [_to_response(image) for image in images]


def _to_response(image: Image) -> ImageResponse:
    return ImageResponse(
        id=image.id,
        event_id=image.event_id,
        s3_url=image_service.public_url(image),
        original_filename=image.original_filename,
        status=ImageStatus(image.status),
        created_at=image.created_at,
    )


def detail(image: Image) -> ImageResponse:
    return _to_response(image)


def faces(db: Session, image: Image) -> list[FaceRecord]:
    rows = FaceRepository(db).list_by_image(image.id)
    return [FaceRecord.model_validate(row) for row in rows]


def delete(db: Session, image: Image) -> None:
    image_service.delete_image(db, image)
