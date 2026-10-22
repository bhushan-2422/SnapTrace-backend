"""Event albums: create, join/leave, list, view, update, delete, status."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import (
    PageParams,
    get_current_user,
    get_db,
    get_joined_event,
    get_owned_event,
    get_page_params,
)
from app.controllers import event_controller
from app.models.event import Event
from app.models.user import User
from app.schemas.event import EventCreate, EventResponse, EventUpdate
from app.schemas.processing import EventProcessingStatusResponse
from app.services import event_service

router = APIRouter(prefix="/events", tags=["events"])


@router.post("", response_model=EventResponse, status_code=status.HTTP_201_CREATED)
def create_event(
    payload: EventCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return event_controller.create(db, user, payload)


@router.get("", response_model=list[EventResponse])
def list_my_events(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    page: PageParams = Depends(get_page_params),
):
    return event_controller.list_mine(db, user, limit=page.limit, offset=page.offset)


@router.get("/{event_id}", response_model=EventResponse)
def read_event(event: Event = Depends(get_joined_event)):
    return event_controller.get(event)


@router.post("/{event_id}/join", status_code=status.HTTP_200_OK)
def join_event(
    event_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Join via the shared event link; the album then shows on your home page."""
    event = event_service.get_event(db, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    try:
        event_controller.join(db, user, event)
    except event_service.AlreadyMember:
        raise HTTPException(status_code=409, detail="You are already a member of this event")
    return {"detail": "Joined", "event_id": event.id}


@router.post("/{event_id}/leave", status_code=status.HTTP_200_OK)
def leave_event(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    event: Event = Depends(get_joined_event),
):
    try:
        event_controller.leave(db, user, event)
    except event_service.OwnerCannotLeave:
        raise HTTPException(
            status_code=400,
            detail="The event owner cannot leave; delete the event instead",
        )
    except event_service.NotAMember:
        raise HTTPException(status_code=403, detail="You are not a member of this event")
    return {"detail": "Left", "event_id": event.id}


@router.patch("/{event_id}", response_model=EventResponse)
def update_event(
    payload: EventUpdate,
    db: Session = Depends(get_db),
    event: Event = Depends(get_owned_event),
):
    return event_controller.update(db, event, payload)


@router.delete("/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_event(db: Session = Depends(get_db), event: Event = Depends(get_owned_event)):
    event_controller.delete(db, event)


@router.get("/{event_id}/status", response_model=EventProcessingStatusResponse)
def processing_status(
    db: Session = Depends(get_db),
    event: Event = Depends(get_owned_event),
):
    """Organizer view: how many photos are uploaded/processing/done/failed."""
    return event_controller.processing_status(db, event)
