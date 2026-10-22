"""`images` table: metadata for every uploaded photograph.

The image file itself lives in S3; PostgreSQL only stores references.
`images.status` is the single source of truth for background processing state
(there is deliberately no ProcessingJob table - Redis is the temporary queue).
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base, utcnow


class ImageStatus(str, enum.Enum):
    """Processing states an image moves through (see project spec section 6)."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Image(Base):
    __tablename__ = "images"
    __table_args__ = (
        # Covers both "images of event X" and "count images of event X by status"
        # (event processing status endpoint).
        Index("ix_images_event_id_status", "event_id", "status"),
        # Guard the state machine at the database level.
        CheckConstraint(
            "status IN ('uploaded', 'processing', 'completed', 'failed')",
            name="ck_images_status",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("events.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Object key inside the S3 bucket - the stable identifier for the file.
    s3_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    # Optional stored/CDN URL; a presigned URL can be generated from s3_key.
    s3_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)

    # Who uploaded this photo. NULL only for rows created before this
    # column existed; used by the delete rule (uploader or event owner).
    uploaded_by: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Stored as a plain string value ("uploaded", ...) with the enum used as
    # the application-level vocabulary.
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=ImageStatus.UPLOADED.value,
        server_default=ImageStatus.UPLOADED.value,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    event: Mapped["Event"] = relationship(back_populates="images")
    # One image -> zero or more detected faces.
    faces: Mapped[list["Face"]] = relationship(
        back_populates="image",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Image id={self.id} event_id={self.event_id} status={self.status!r}>"
