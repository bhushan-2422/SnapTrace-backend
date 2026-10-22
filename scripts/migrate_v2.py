"""Upgrade an existing database to the membership model (v2).

Adds:
  * `event_members` table (who is present in which album)
  * `images.uploaded_by` column (who uploaded each photo)
  * one membership row per existing event for its owner

Idempotent: safe to run more than once.  Run from the repository root:

    python scripts/migrate_v2.py

Fresh databases do not need this - `scripts/initialize_db.py` already
creates the new schema from the models.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.db.database import engine  # noqa: E402


def main() -> int:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS event_members (
            event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            joined_at TIMESTAMP WITH TIME ZONE NOT NULL,
            PRIMARY KEY (event_id, user_id)
        )
        """,
        """
        ALTER TABLE images
            ADD COLUMN IF NOT EXISTS uploaded_by INTEGER
            REFERENCES users(id) ON DELETE SET NULL
        """,
        """
        INSERT INTO event_members (event_id, user_id, joined_at)
        SELECT id, owner_id, NOW() FROM events
        ON CONFLICT DO NOTHING
        """,
    ]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    print("Migration v2 applied: event_members table, images.uploaded_by, owner memberships.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
