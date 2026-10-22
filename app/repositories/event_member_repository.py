"""Data access for the `event_members` table. No business logic."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.event_member import EventMember


class EventMemberRepository:
    """Queries for event membership.  Commits are owned by the calling service."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get(self, event_id: int, user_id: int) -> EventMember | None:
        return self.db.get(EventMember, (event_id, user_id))

    def is_member(self, event_id: int, user_id: int) -> bool:
        stmt = select(EventMember.event_id).where(
            EventMember.event_id == event_id,
            EventMember.user_id == user_id,
        )
        return self.db.scalar(stmt) is not None

    def add(self, member: EventMember) -> EventMember:
        self.db.add(member)
        self.db.flush()
        return member

    def delete(self, member: EventMember) -> None:
        self.db.delete(member)
        self.db.flush()
