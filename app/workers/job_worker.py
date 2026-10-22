"""Background worker: consumes the Redis image queue.

Run it from the repository root (separate terminal, API can stay up):

    python -m app.workers.job_worker           # run forever
    python -m app.workers.job_worker --once    # process one job, exit
    python -m app.workers.job_worker --drain   # finish what is queued, exit

`--drain` is what CI / a demo script wants; `--once` is for debugging.
"""

from __future__ import annotations

import argparse
import logging
import time

from redis.exceptions import RedisError

from app.db.session import SessionLocal
from app.integrations import redis_client
from app.workers import image_processor

logger = logging.getLogger("snaptrace.worker")


def handle(image_id: int) -> bool:
    """One job = one DB session = one image."""
    db = SessionLocal()
    try:
        return image_processor.process_image(db, image_id)
    finally:
        db.close()


def run_forever() -> None:
    logger.info("worker started, waiting for jobs...")
    while True:
        try:
            image_id = redis_client.dequeue_image(timeout=5)
        except RedisError as exc:
            logger.warning("redis error while waiting for jobs: %s", exc)
            time.sleep(1)
            continue
        if image_id is None:
            continue  # quiet queue, loop again (allows clean Ctrl+C)
        try:
            started = time.perf_counter()
            handle(image_id)
            logger.info("finished image %s in %.1fs", image_id, time.perf_counter() - started)
        except Exception:
            # process_image already contains per-image failures; this
            # guards against unexpected ones (DB down, disk error) so
            # one bad job can never stop the worker permanently.
            logger.exception("unexpected error while processing image %s", image_id)


def drain() -> None:
    """Process everything currently queued, then stop."""
    processed = 0
    while True:
        image_id = redis_client.dequeue_image(timeout=1)
        if image_id is None:
            break
        handle(image_id)
        processed += 1
    logger.info("drained %d job(s)", processed)


def main() -> int:
    parser = argparse.ArgumentParser(description="SnapTrace image processing worker")
    parser.add_argument("--once", action="store_true", help="process one job and exit")
    parser.add_argument("--drain", action="store_true", help="process queued jobs and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not redis_client.ping():
        logger.error("cannot reach Redis at the configured REDIS_URL - is redis-server running?")
        return 1

    try:
        if args.once:
            image_id = redis_client.dequeue_image(timeout=5)
            if image_id is None:
                logger.info("queue empty, nothing to do")
            else:
                handle(image_id)
        elif args.drain:
            drain()
        else:
            run_forever()
    except KeyboardInterrupt:
        logger.info("worker stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
