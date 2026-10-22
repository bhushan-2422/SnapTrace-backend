"""Event (album) CRUD plus membership (join / leave)."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.event import Event
from app.models.event_member import EventMember
from app.models.user import User
from app.repositories.event_member_repository import EventMemberRepository
from app.repositories.event_repository import EventRepository


class NotEventOwner(Exception):
    """The acting user does not own this event."""


class AlreadyMember(Exception):
    """The user already belongs to this event."""


class NotAMember(Exception):
    """The user has not joined this event (and does not own it)."""


class OwnerCannotLeave(Exception):
    """The owner must delete the event instead of leaving it."""


def create_event(db: Session, *, owner_id: int, name: str, description: str | None) -> Event:
    event = Event(name=name.strip(), description=description, owner_id=owner_id)
    EventRepository(db).add(event)
    # The owner is a member of their own album from the start, so the
    # home page and member checks work uniformly.
    EventMemberRepository(db).add(EventMember(event_id=event.id, user_id=owner_id))
    db.commit()
    db.refresh(event)
    return event


def list_events(db: Session, *, user_id: int, limit: int, offset: int) -> list[Event]:
    """Home page: owned + joined albums."""
    return EventRepository(db).list_for_user(user_id, limit=limit, offset=offset)


def get_event(db: Session, event_id: int) -> Event | None:
    return EventRepository(db).get_by_id(event_id)


def is_member(db: Session, event_id: int, user_id: int) -> bool:
    return EventMemberRepository(db).is_member(event_id, user_id)


def require_owner(event: Event, user: User) -> None:
    """Raise NotEventOwner unless `user` created `event`."""
    if event.owner_id != user.id:
        raise NotEventOwner(event.id)


def join_event(db: Session, event: Event, user: User) -> None:
    if event.owner_id == user.id or is_member(db, event.id, user.id):
        raise AlreadyMember(event.id)
    EventMemberRepository(db).add(EventMember(event_id=event.id, user_id=user.id))
    db.commit()


def leave_event(db: Session, event: Event, user: User) -> None:
    if event.owner_id == user.id:
        raise OwnerCannotLeave(event.id)
    repo = EventMemberRepository(db)
    member = repo.get(event.id, user.id)
    if member is None:
        raise NotAMember(event.id)
    repo.delete(member)
    db.commit()


def update_event(
    db: Session,
    event: Event,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Event:
    if name is not None:
        event.name = name.strip()
    if description is not None:
        event.description = description
    db.commit()
    db.refresh(event)
    return event


def delete_event(db: Session, event: Event) -> None:
    """Delete the event; memberships, images, faces cascade away."""
    EventRepository(db).delete(event)
    db.commit()
