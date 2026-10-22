"""Shared FastAPI dependencies: database session, current user, event
access, pagination.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import get_db
from app.models.event import Event
from app.models.user import User
from app.services import event_service, user_service
from app.utils import security

__all__ = [
    "PageParams",
    "get_db",
    "get_current_user",
    "get_owned_event",
    "get_joined_event",
    "get_page_params",
]

# Extracts the bearer token; tokenUrl is where a client would get one.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme),
) -> User:
    """Resolve `Authorization: Bearer <access token>` to a live user."""
    try:
        claims = security.decode_token(token, security.ACCESS)
    except security.TokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = user_service.get_user(db, claims["user_id"])
    if user is None:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return user


def get_owned_event(
    event_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Event:
    """404 when the event is unknown, 403 when the caller does not own it.

    Used by every mutating endpoint (upload, update, delete).
    """
    event = event_service.get_event(db, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    if event.owner_id != user.id:
        raise HTTPException(status_code=403, detail="You do not own this event")
    return event


def get_joined_event(
    event_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Event:
    """The event, but only for its owner or members.

    Used by every endpoint that touches album content (upload, list,
    read, faces, search, processing status).  Non-members get 403 even
    when the event id is known - joining happens via the shared link.
    """
    event = event_service.get_event(db, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    if event.owner_id != user.id and not event_service.is_member(db, event.id, user.id):
        raise HTTPException(
            status_code=403,
            detail="Join this event first (POST /events/{event_id}/join)",
        )
    return event


@dataclass(frozen=True)
class PageParams:
    limit: int
    offset: int


def get_page_params(
    limit: int = Query(
        settings.DEFAULT_PAGE_SIZE, ge=1, le=settings.MAX_PAGE_SIZE,
        description="Maximum number of rows to return.",
    ),
    offset: int = Query(0, ge=0, description="Rows to skip (pagination)."),
) -> PageParams:
    """Caps every list endpoint: no unbounded SELECTs ever leave the DB."""
    return PageParams(limit=limit, offset=offset)
