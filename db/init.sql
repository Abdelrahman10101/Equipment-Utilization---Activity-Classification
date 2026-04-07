-- ============================================
-- Eagle Vision - TimescaleDB Schema
-- ============================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Main events table: stores per-frame, per-equipment analysis results
CREATE TABLE IF NOT EXISTS equipment_events (
    time                    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    frame_id                INTEGER         NOT NULL,
    equipment_id            TEXT            NOT NULL,
    equipment_class         TEXT            NOT NULL,
    video_timestamp         TEXT,
    current_state           TEXT            NOT NULL,
    current_activity        TEXT,
    motion_source           TEXT,
    bbox_x                  REAL,
    bbox_y                  REAL,
    bbox_w                  REAL,
    bbox_h                  REAL,
    confidence              REAL,
    total_tracked_seconds   REAL            DEFAULT 0,
    total_active_seconds    REAL            DEFAULT 0,
    total_idle_seconds      REAL            DEFAULT 0,
    utilization_percent     REAL            DEFAULT 0
);

-- Convert to hypertable for time-series optimizations
SELECT create_hypertable('equipment_events', 'time', if_not_exists => TRUE);

-- Index for fast lookups by equipment
CREATE INDEX IF NOT EXISTS idx_equipment_id ON equipment_events (equipment_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_equipment_state ON equipment_events (current_state, time DESC);

-- Summary view: latest state per equipment
CREATE OR REPLACE VIEW equipment_latest AS
SELECT DISTINCT ON (equipment_id)
    equipment_id,
    equipment_class,
    current_state,
    current_activity,
    motion_source,
    total_tracked_seconds,
    total_active_seconds,
    total_idle_seconds,
    utilization_percent,
    time as last_updated
FROM equipment_events
ORDER BY equipment_id, time DESC;

-- Utilization summary view
CREATE OR REPLACE VIEW utilization_summary AS
SELECT
    equipment_id,
    equipment_class,
    MAX(total_tracked_seconds) as total_tracked_seconds,
    MAX(total_active_seconds) as total_active_seconds,
    MAX(total_idle_seconds) as total_idle_seconds,
    MAX(utilization_percent) as utilization_percent,
    COUNT(*) as total_events,
    MAX(time) as last_updated
FROM equipment_events
GROUP BY equipment_id, equipment_class;
