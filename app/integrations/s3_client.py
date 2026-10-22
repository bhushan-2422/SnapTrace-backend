"""AWS S3: object upload, download and presigned URL generation.

Until AWS credentials are pasted into .env, `settings.s3_enabled()` is
False and the rest of the app stores/serves files from disk instead
(see app/services/image_service.py).  Nothing else in the code base
talks to boto3 directly.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings

logger = logging.getLogger(__name__)

# How long a presigned download link stays valid.
PRESIGNED_EXPIRES_SECONDS = 3600


class StorageError(Exception):
    """Raised when an S3 operation fails."""


@lru_cache(maxsize=1)
def _client():
    """Shared boto3 client pinned to the bucket's regional endpoint.

    Presigned URLs are signed against this endpoint, and the signature
    covers the Host header.  Without an explicit endpoint botocore signs
    them as `bucket.s3.amazonaws.com`, which makes in-region buckets
    answer 307 TemporaryRedirect - and the redirect invalidates the
    signature (403 SignatureDoesNotMatch / 400).
    """
    return boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
        endpoint_url=f"https://s3.{settings.AWS_REGION}.amazonaws.com",
    )


def upload_file(local_path: Path, key: str, content_type: str) -> None:
    """Upload one local file to the configured bucket."""
    try:
        _client().upload_file(
            str(local_path),
            settings.S3_BUCKET,
            key,
            ExtraArgs={"ContentType": content_type},
        )
    except (BotoCoreError, ClientError) as exc:
        raise StorageError(f"s3 upload failed for {key}: {exc}") from exc


def download_file(key: str, local_path: Path) -> None:
    """Fetch an object back to disk (used when reprocessing)."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _client().download_file(settings.S3_BUCKET, key, str(local_path))
    except (BotoCoreError, ClientError) as exc:
        raise StorageError(f"s3 download failed for {key}: {exc}") from exc


def presigned_url(key: str, *, download_filename: str | None = None) -> str | None:
    """Short-lived download URL for a private object.

    The raw object key is never handed to clients - they only ever see
    this signed link, which is what keeps other people's photos private.
    Pass `download_filename` to force a browser download (attachment)
    with that filename instead of inline display.
    """
    params: dict = {"Bucket": settings.S3_BUCKET, "Key": key}
    if download_filename:
        safe = download_filename.replace('"', "").replace("\n", "")
        params["ResponseContentDisposition"] = f'attachment; filename="{safe}"'
    try:
        return _client().generate_presigned_url(
            "get_object",
            Params=params,
            ExpiresIn=PRESIGNED_EXPIRES_SECONDS,
        )
    except (BotoCoreError, ClientError) as exc:
        logger.warning("presigned url failed for %s: %s", key, exc)
        return None


def delete_object(key: str) -> None:
    try:
        _client().delete_object(Bucket=settings.S3_BUCKET, Key=key)
    except (BotoCoreError, ClientError) as exc:
        logger.warning("s3 delete failed for %s: %s", key, exc)
