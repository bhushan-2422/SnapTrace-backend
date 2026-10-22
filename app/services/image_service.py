"""Saving uploads, queueing them, serving them and deleting them.

Storage locations
-----------------
Local mode (no AWS keys yet): the file lands in
`public/upload/<event_id>/<uuid>.<ext>` and `images.s3_key` holds that
relative path, so the file is served from `/media/<s3_key>`.

S3 mode (keys present): the worker uploads the file, rewrites
`images.s3_key` to the object key `events/<event_id>/<uuid>.<ext>` and
clients only ever receive a presigned URL generated on the fly.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import UploadFile
from redis.exceptions import RedisError
from sqlalchemy.orm import Session

from app.config import PUBLIC_DIR, settings
from app.integrations import redis_client, s3_client
from app.models.event import Event
from app.models.image import Image, ImageStatus
from app.repositories.image_repository import ImageRepository
from app.utils import file_utils

logger = logging.getLogger(__name__)


class TooManyFiles(Exception):
    """More files in one request than MAX_FILES_PER_UPLOAD."""


class QueueUnavailable(Exception):
    """Redis is down - the upload cannot be queued for processing."""


async def store_uploads(
    db: Session, event: Event, uploads: list[UploadFile], *, uploaded_by: int
) -> list[Image]:
    """Validate, save and register uploaded photos, then queue them.

    The files and their rows are committed first; queueing happens
    afterwards.  If Redis is down the rows still exist with status
    "uploaded" (so nothing is lost) and QueueUnavailable tells the client
    the job was not scheduled.
    """
    if len(uploads) > settings.MAX_FILES_PER_UPLOAD:
        raise TooManyFiles(f"at most {settings.MAX_FILES_PER_UPLOAD} files per request")
    if not uploads:
        raise ValueError("no files provided")

    repo = ImageRepository(db)
    images: list[Image] = []

    for upload in uploads:
        # Extension check happens before a single byte is written.
        key = file_utils.unique_key(event.id, upload.filename)
        local_path = PUBLIC_DIR / key  # public/upload/<event_id>/<uuid>.<ext>

        try:
            await file_utils.save_upload(upload, local_path)
        finally:
            await upload.close()

        images.append(
            Image(
                event_id=event.id,
                s3_key=key,
                original_filename=file_utils.safe_original_name(upload.filename),
                status=ImageStatus.UPLOADED.value,
                uploaded_by=uploaded_by,
            )
        )

    repo.add_many(images)
    db.commit()
    for image in images:
        db.refresh(image)

    try:
        for image in images:
            redis_client.enqueue_image(image.id)
    except RedisError as exc:
        logger.error("queue unavailable after saving %d images: %s", len(images), exc)
        raise QueueUnavailable(str(exc)) from exc

    return images


def list_images(
    db: Session,
    event_id: int,
    *,
    status: ImageStatus | None,
    limit: int,
    offset: int,
) -> list[Image]:
    return ImageRepository(db).list_by_event(event_id, status=status, limit=limit, offset=offset)


def get_image(db: Session, image_id: int) -> Image | None:
    return ImageRepository(db).get_by_id(image_id)


def public_url(image: Image) -> str | None:
    """URL a client can use to open this photo.

    Presigned S3 link when S3 is configured, `/media/...` otherwise.
    """
    if settings.s3_enabled():
        return s3_client.presigned_url(image.s3_key)
    return f"/media/{image.s3_key}"


def resolve_source(image: Image) -> Path:
    """Where the worker can read the photo from.

    Prefers the local copy; falls back to fetching it back from S3 when
    the local file has already been cleaned up.
    """
    local_path = PUBLIC_DIR / image.s3_key
    if local_path.is_file():
        return local_path

    if settings.s3_enabled():
        temp_path = settings.TEMP_DIR / Path(image.s3_key).name
        s3_client.download_file(image.s3_key, temp_path)
        return temp_path

    raise FileNotFoundError(f"no local file for image {image.id} ({image.s3_key})")


def delete_image(db: Session, image: Image) -> None:
    """Delete the row (faces/embeddings cascade) and every stored copy."""
    local_path = PUBLIC_DIR / image.s3_key
    ImageRepository(db).delete(image)
    db.commit()

    local_path.unlink(missing_ok=True)
    if settings.s3_enabled():
        s3_client.delete_object(image.s3_key)
