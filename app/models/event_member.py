"""`event_members` table: which users are present in which event album.

The owner row is written automatically when the event is created; other
users join through the shared event link (POST /events/{id}/join) and
can leave again.  This table is what the home page (GET /events) reads
to list "my albums".
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base, utcnow


class EventMember(Base):
    __tablename__ = "event_members"

    # Composite primary key: one row per (event, user) - joining twice
    # is rejected by the database itself.
    event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("events.id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    joined_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    def __repr__(self) -> str:
        return f"<EventMember event_id={self.event_id} user_id={self.user_id}>"
