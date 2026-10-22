"""Redis: connection helper + the background image queue.

The queue is a plain Redis list (LPUSH producer / BRPOP consumer).  That
is enough for "process these images eventually, don't block the HTTP
request" and needs no extra broker or worker framework.  Job payload is
just the image id as a string.
"""

from __future__ import annotations

import logging

import redis
from redis.exceptions import RedisError

from app.config import constants, settings

logger = logging.getLogger(__name__)

_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    """Process-wide Redis connection (created lazily, reused afterwards)."""
    global _client
    if _client is None:
        _client = redis.Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _client


def ping() -> bool:
    """True when Redis answers - used by the health endpoint and startup."""
    try:
        return bool(get_redis().ping())
    except RedisError as exc:
        logger.warning("redis unavailable: %s", exc)
        return False


def enqueue_image(image_id: int) -> None:
    """Producer side: push one image id onto the processing queue.

    Raises RedisError when the queue is down; the API turns that into a
    503 so the client knows the upload was *not* accepted for processing.
    """
    get_redis().lpush(constants.IMAGE_QUEUE, str(image_id))


def dequeue_image(timeout: int = 5) -> int | None:
    """Consumer side: block up to `timeout` seconds for the next image id.

    Returns None when the queue stayed quiet (worker loop sleeps again).

    Note: the client's socket timeout is 2s (fail-fast for API calls),
    which is shorter than this blocking wait - an idle BRPOP therefore
    raises TimeoutError before the server replies.  That is treated
    exactly like an empty queue instead of killing the worker.
    """
    try:
        item = get_redis().brpop(constants.IMAGE_QUEUE, timeout=timeout)
    except redis.exceptions.TimeoutError:
        return None
    if item is None:
        return None
    _queue_name, payload = item  # BRPOP -> (queue, value)
    return int(payload)


def queue_length() -> int:
    try:
        return int(get_redis().llen(constants.IMAGE_QUEUE))
    except RedisError:
        return 0
