"""
DB Sink - Database Models & Operations

Handles batch inserts into TimescaleDB.
"""
import logging
from typing import List, Dict, Any
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import execute_values

from config import Config

logger = logging.getLogger("DBModels")

INSERT_QUERY = """
    INSERT INTO equipment_events (
        time, frame_id, equipment_id, equipment_class, video_timestamp,
        current_state, current_activity, motion_source,
        bbox_x, bbox_y, bbox_w, bbox_h, confidence,
        total_tracked_seconds, total_active_seconds,
        total_idle_seconds, utilization_percent
    ) VALUES %s
"""


def create_connection():
    """Create a PostgreSQL connection with retry logic."""
    import time

    max_retries = 30
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(Config.get_dsn())
            conn.autocommit = False
            logger.info(f"Connected to TimescaleDB at {Config.POSTGRES_HOST}")
            return conn
        except psycopg2.OperationalError as e:
            logger.warning(
                f"DB connection attempt {attempt + 1}/{max_retries}: {e}"
            )
            time.sleep(2)

    raise RuntimeError("Failed to connect to TimescaleDB after all retries")


def batch_insert(conn, events: List[Dict[str, Any]]) -> int:
    """
    Batch insert equipment events into TimescaleDB.

    Returns the number of rows inserted.
    """
    if not events:
        return 0

    rows = []
    for event in events:
        utilization = event.get("utilization", {})
        time_analytics = event.get("time_analytics", {})
        bbox = event.get("bbox", {})

        row = (
            datetime.now(timezone.utc),
            event.get("frame_id", 0),
            event.get("equipment_id", "UNK"),
            event.get("equipment_class", "unknown"),
            event.get("timestamp", "00:00:00.000"),
            utilization.get("current_state", "INACTIVE"),
            utilization.get("current_activity", "WAITING"),
            utilization.get("motion_source", "none"),
            bbox.get("x", 0),
            bbox.get("y", 0),
            bbox.get("w", 0),
            bbox.get("h", 0),
            event.get("confidence", 0),
            time_analytics.get("total_tracked_seconds", 0),
            time_analytics.get("total_active_seconds", 0),
            time_analytics.get("total_idle_seconds", 0),
            time_analytics.get("utilization_percent", 0),
        )
        rows.append(row)

    try:
        with conn.cursor() as cur:
            execute_values(cur, INSERT_QUERY, rows, page_size=100)
        conn.commit()
        return len(rows)
    except Exception as e:
        conn.rollback()
        logger.error(f"Batch insert failed: {e}")
        raise
