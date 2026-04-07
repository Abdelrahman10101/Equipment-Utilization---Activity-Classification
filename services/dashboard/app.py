"""
Eagle Vision — Streamlit Dashboard

Real-time equipment utilization monitoring dashboard.
Displays:
  - Live annotated video feed
  - Equipment status cards (ACTIVE/INACTIVE + current activity)
  - Utilization metrics and charts
"""
import sys
import json
import time
import logging
import threading
from collections import OrderedDict

import streamlit as st
import pandas as pd
import psycopg2
from confluent_kafka import Consumer, KafkaError

from config import Config

# Add components to path
sys.path.insert(0, ".")
from components.video_feed import render_video_feed, render_placeholder
from components.status_panel import render_status_cards
from components.utilization import (
    render_utilization_metrics,
    render_per_equipment_breakdown,
    render_utilization_chart,
    render_activity_breakdown,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Dashboard")

# ── Page Config ──
st.set_page_config(
    page_title="Eagle Vision — Equipment Monitor",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown(
    """
    <style>
    /* Dark professional theme */
    .stApp {
        background-color: #0e1117;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 20px 30px;
        border-radius: 16px;
        margin-bottom: 24px;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2.2em;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: #a0a0b0;
        margin: 4px 0 0 0;
        font-size: 1.05em;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e, #1e1e3a);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricLabel"] {
        color: #a0a0c0 !important;
        font-size: 0.9em !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.6em !important;
    }

    /* Section headers */
    .section-header {
        color: #e94560;
        border-bottom: 2px solid #e94560;
        padding-bottom: 6px;
        margin: 20px 0 12px 0;
    }

    /* Status indicator pulse */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .status-live {
        animation: pulse 2s infinite;
        color: #00ff88;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session State Init ──
if "latest_frame" not in st.session_state:
    st.session_state.latest_frame = None
if "equipment_states" not in st.session_state:
    st.session_state.equipment_states = {}
if "full_states" not in st.session_state:
    st.session_state.full_states = {}
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "consumer_started" not in st.session_state:
    st.session_state.consumer_started = False


def fetch_latest_from_kafka():
    """Fetch latest messages from Kafka (non-blocking)."""
    try:
        if "kafka_consumer" not in st.session_state:
            consumer_conf = {
                "bootstrap.servers": Config.KAFKA_BROKER,
                "group.id": f"dashboard-{id(st.session_state)}",
                "auto.offset.reset": "latest",
                "max.partition.fetch.bytes": 2 * 1024 * 1024,
                "session.timeout.ms": 10000,
            }
            consumer = Consumer(consumer_conf)
            consumer.subscribe([Config.ANNOTATED_FRAMES_TOPIC, Config.EQUIPMENT_EVENTS_TOPIC])
            st.session_state.kafka_consumer = consumer
            # Give it a tiny bit of time to get initial assignment
            time.sleep(0.5)

        consumer = st.session_state.kafka_consumer
        # Poll for a batch of messages
        latest_frame = None
        equipment_states = {}
        full_states = {}

        deadline = time.time() + 1.5
        while time.time() < deadline:
            msg = consumer.poll(timeout=0.2)
            if msg is None:
                continue
            if msg.error():
                continue

            topic = msg.topic()
            try:
                data = json.loads(msg.value().decode("utf-8"))
            except Exception:
                continue

            if topic == Config.ANNOTATED_FRAMES_TOPIC:
                latest_frame = data
                if "states" in data:
                    equipment_states.update(data["states"])

            elif topic == Config.EQUIPMENT_EVENTS_TOPIC:
                eid = data.get("equipment_id", "UNK")
                equipment_states[eid] = {
                    "state": data.get("utilization", {}).get("current_state", "INACTIVE"),
                    "activity": data.get("utilization", {}).get("current_activity", "WAITING"),
                    "utilization": data.get("time_analytics", {}).get("utilization_percent", 0),
                    "equipment_class": data.get("equipment_class", "unknown"),
                }
                full_states[eid] = {
                    "equipment_class": data.get("equipment_class", "unknown"),
                    "current_state": data.get("utilization", {}).get("current_state", "INACTIVE"),
                    "current_activity": data.get("utilization", {}).get("current_activity", "WAITING"),
                    "motion_source": data.get("utilization", {}).get("motion_source", "none"),
                    "total_tracked_seconds": data.get("time_analytics", {}).get("total_tracked_seconds", 0),
                    "total_active_seconds": data.get("time_analytics", {}).get("total_active_seconds", 0),
                    "total_idle_seconds": data.get("time_analytics", {}).get("total_idle_seconds", 0),
                    "utilization_percent": data.get("time_analytics", {}).get("utilization_percent", 0),
                }

        # Do not close the consumer here, keep it alive in session_state!

        return latest_frame, equipment_states, full_states

    except Exception as e:
        logger.warning(f"Kafka fetch error: {e}")
        return None, {}, {}


def fetch_db_data() -> pd.DataFrame:
    """Fetch recent data from TimescaleDB."""
    try:
        conn = psycopg2.connect(Config.get_dsn())
        query = """
            SELECT
                time, equipment_id, equipment_class,
                current_state, current_activity,
                utilization_percent,
                total_active_seconds, total_idle_seconds,
                total_tracked_seconds
            FROM equipment_events
            WHERE time > NOW() - INTERVAL '10 minutes'
            ORDER BY time DESC
            LIMIT 1000
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.warning(f"DB query error: {e}")
        return pd.DataFrame()


def fetch_utilization_summary() -> dict:
    """Fetch the latest utilization summary per equipment from DB."""
    try:
        conn = psycopg2.connect(Config.get_dsn())
        query = """
            SELECT * FROM utilization_summary
        """
        df = pd.read_sql(query, conn)
        conn.close()

        result = {}
        for _, row in df.iterrows():
            result[row["equipment_id"]] = row.to_dict()
        return result
    except Exception as e:
        logger.warning(f"Summary query error: {e}")
        return {}


# ── Main Layout ──

# Header
st.markdown(
    """
    <div class="main-header">
        <h1>🦅 Eagle Vision</h1>
        <p>Real-Time Construction Equipment Utilization Monitor</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_rate = st.slider("Refresh interval (sec)", 1, 10, Config.REFRESH_INTERVAL)
    data_source = st.radio("Data source", ["Kafka (Live)", "Database (Historical)"])

    st.markdown("---")
    st.markdown("### 📋 System Status")

    # Connection status
    try:
        conn_test = psycopg2.connect(Config.get_dsn())
        conn_test.close()
        st.success("✅ TimescaleDB: Connected")
    except Exception:
        st.error("❌ TimescaleDB: Disconnected")

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.85em;">
            Eagle Vision v1.0<br/>
            Equipment Utilization Tracker
        </div>
        """,
        unsafe_allow_html=True,
    )

# Fetch data
if data_source == "Kafka (Live)":
    latest_frame, equipment_states, full_states = fetch_latest_from_kafka()
    if latest_frame:
        st.session_state.latest_frame = latest_frame
    if equipment_states:
        st.session_state.equipment_states.update(equipment_states)
    if full_states:
        st.session_state.full_states.update(full_states)
else:
    full_states = fetch_utilization_summary()
    if full_states:
        st.session_state.full_states = full_states
        st.session_state.equipment_states = {
            eid: {
                "state": s.get("current_state", "INACTIVE"),
                "activity": s.get("current_activity", "WAITING"),
                "utilization": s.get("utilization_percent", 0),
                "equipment_class": s.get("equipment_class", "unknown"),
            }
            for eid, s in full_states.items()
        }


# ── Row 1: Video Feed + Status Cards ──
col_video, col_status = st.columns([2, 1])

with col_video:
    st.markdown('<h3 class="section-header">📹 Live Video Feed</h3>', unsafe_allow_html=True)
    if st.session_state.latest_frame:
        render_video_feed(st.session_state.latest_frame)
    else:
        render_placeholder()

with col_status:
    st.markdown('<h3 class="section-header">🔧 Equipment Status</h3>', unsafe_allow_html=True)
    if st.session_state.equipment_states:
        render_status_cards(st.session_state.equipment_states)
    else:
        st.info("Waiting for equipment detection...")


# ── Row 2: Utilization Metrics ──
st.markdown('<h3 class="section-header">📊 Utilization Dashboard</h3>', unsafe_allow_html=True)

if st.session_state.full_states:
    render_utilization_metrics(st.session_state.full_states)
    st.markdown("---")
    render_per_equipment_breakdown(st.session_state.full_states)
else:
    st.info("⏳ Utilization data will appear once equipment is tracked...")


# ── Row 3: Charts (from DB) ──
st.markdown('<h3 class="section-header">📈 Historical Analytics</h3>', unsafe_allow_html=True)

db_data = fetch_db_data()
if not db_data.empty:
    tab1, tab2 = st.tabs(["Utilization Over Time", "Activity Distribution"])
    with tab1:
        render_utilization_chart(db_data)
    with tab2:
        render_activity_breakdown(db_data)
else:
    st.info("📈 Historical data will appear once events are stored in TimescaleDB.")


# ── Auto-refresh ──
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
