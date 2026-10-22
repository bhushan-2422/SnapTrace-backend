"""SQLAlchemy models.

Importing this package registers every table on `Base.metadata`, which is what
`scripts/initialize_db.py` relies on to create the schema.
"""

from app.models.event import Event
from app.models.event_member import EventMember
from app.models.face import Face
from app.models.face_embedding import FaceEmbedding
from app.models.image import Image, ImageStatus
from app.models.user import User

__all__ = [
    "Event",
    "EventMember",
    "Face",
    "FaceEmbedding",
    "Image",
    "ImageStatus",
    "User",
]
