"""Data access for the `events` table. Contains no business logic."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.event import Event
from app.models.event_member import EventMember


class EventRepository:
    """Queries for events.  Commits are owned by the calling service."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_id(self, event_id: int) -> Event | None:
        return self.db.get(Event, event_id)

    def list(
        self,
        *,
        owner_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Event]:
        """All events, optionally filtered to those owned by one user."""
        stmt = select(Event)
        if owner_id is not None:
            stmt = stmt.where(Event.owner_id == owner_id)
        stmt = stmt.order_by(Event.created_at.desc(), Event.id.desc())
        stmt = stmt.limit(limit).offset(offset)
        return list(self.db.scalars(stmt).all())

    def list_by_owner(self, owner_id: int, *, limit: int = 100, offset: int = 0) -> list[Event]:
        return self.list(owner_id=owner_id, limit=limit, offset=offset)

    def list_for_user(
        self, user_id: int, *, limit: int = 100, offset: int = 0
    ) -> list[Event]:
        """Home page: every event this user owns or has joined."""
        stmt = (
            select(Event)
            .where(
                (Event.owner_id == user_id)
                | Event.id.in_(select(EventMember.event_id).where(EventMember.user_id == user_id))
            )
            .order_by(Event.created_at.desc(), Event.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.db.scalars(stmt).all())

    def add(self, event: Event) -> Event:
        self.db.add(event)
        self.db.flush()
        return event

    def delete(self, event: Event) -> None:
        """Removes the event; images/faces follow via relationship cascade."""
        self.db.delete(event)
        self.db.flush()
