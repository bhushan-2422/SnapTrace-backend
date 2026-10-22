"""Event CRUD plus the per-event processing status summary."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.event import Event
from app.models.image import ImageStatus
from app.models.user import User
from app.repositories.image_repository import ImageRepository
from app.schemas.event import EventCreate, EventResponse, EventUpdate
from app.schemas.processing import EventProcessingStatusResponse
from app.services import event_service


def create(db: Session, user: User, payload: EventCreate) -> EventResponse:
    event = event_service.create_event(
        db, owner_id=user.id, name=payload.name, description=payload.description
    )
    return EventResponse.model_validate(event)


def list_mine(db: Session, user: User, *, limit: int, offset: int) -> list[EventResponse]:
    events = event_service.list_events(db, user_id=user.id, limit=limit, offset=offset)
    return [EventResponse.model_validate(event) for event in events]


def join(db: Session, user: User, event: Event) -> None:
    event_service.join_event(db, event, user)


def leave(db: Session, user: User, event: Event) -> None:
    event_service.leave_event(db, event, user)


def get(event: Event) -> EventResponse:
    return EventResponse.model_validate(event)


def update(db: Session, event: Event, payload: EventUpdate) -> EventResponse:
    updated = event_service.update_event(
        db, event, name=payload.name, description=payload.description
    )
    return EventResponse.model_validate(updated)


def delete(db: Session, event: Event) -> None:
    event_service.delete_event(db, event)


def processing_status(db: Session, event: Event) -> EventProcessingStatusResponse:
    repo = ImageRepository(db)
    counts = repo.status_counts_by_event(event.id)
    return EventProcessingStatusResponse(
        event_id=event.id,
        total=repo.count_by_event(event.id),
        uploaded=counts[ImageStatus.UPLOADED.value],
        processing=counts[ImageStatus.PROCESSING.value],
        completed=counts[ImageStatus.COMPLETED.value],
        failed=counts[ImageStatus.FAILED.value],
    )
