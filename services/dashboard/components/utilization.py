"""
Dashboard - Utilization Component

Displays utilization metrics, time breakdowns, and charts.
"""
import logging

import streamlit as st
import pandas as pd

logger = logging.getLogger("Utilization")


def render_utilization_metrics(equipment_states: dict):
    """
    Render utilization metrics: Total Working Time, Idle Time, Utilization %.

    Args:
        equipment_states: Dict of equipment_id -> full state info
    """
    if not equipment_states:
        st.info("📊 Waiting for utilization data...")
        return

    # Calculate totals
    total_active = 0
    total_idle = 0
    total_tracked = 0

    for state in equipment_states.values():
        total_active += state.get("total_active_seconds", 0)
        total_idle += state.get("total_idle_seconds", 0)
        total_tracked += state.get("total_tracked_seconds", 0)

    overall_util = (total_active / total_tracked * 100) if total_tracked > 0 else 0

    # Display global metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "⏱️ Total Tracked",
            _format_time(total_tracked),
            help="Total time all equipment has been tracked",
        )
    with col2:
        st.metric(
            "🟢 Working Time",
            _format_time(total_active),
            delta=f"{overall_util:.1f}%",
            delta_color="normal",
        )
    with col3:
        st.metric(
            "🔴 Idle Time",
            _format_time(total_idle),
            delta=f"{100 - overall_util:.1f}%",
            delta_color="inverse",
        )
    with col4:
        st.metric(
            "📊 Overall Utilization",
            f"{overall_util:.1f}%",
            help="Total Active Time / Total Tracked Time across all equipment",
        )


def render_per_equipment_breakdown(equipment_states: dict):
    """Render per-equipment utilization breakdown."""
    if not equipment_states:
        return

    st.subheader("Per-Equipment Breakdown")

    for eid, state in equipment_states.items():
        eq_class = state.get("equipment_class", "unknown")
        utilization = state.get("utilization_percent", 0)
        active = state.get("total_active_seconds", 0)
        idle = state.get("total_idle_seconds", 0)
        tracked = state.get("total_tracked_seconds", 0)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"**{eid}** ({eq_class})")
            st.markdown(
                f"Active: `{_format_time(active)}` | "
                f"Idle: `{_format_time(idle)}` | "
                f"Total: `{_format_time(tracked)}`"
            )

        with col2:
            st.progress(
                min(utilization / 100, 1.0),
                text=f"{utilization:.1f}% Utilization",
            )


def render_utilization_chart(db_data: pd.DataFrame):
    """
    Render time-series utilization chart from TimescaleDB data.

    Args:
        db_data: DataFrame with columns: time, equipment_id, utilization_percent, current_state
    """
    if db_data is None or db_data.empty:
        st.info("📈 Chart data will appear once events are recorded in the database.")
        return

    st.subheader("📈 Utilization Over Time")

    # Pivot data for multi-line chart
    if "equipment_id" in db_data.columns and "utilization_percent" in db_data.columns:
        try:
            chart_data = db_data.pivot_table(
                index="time",
                columns="equipment_id",
                values="utilization_percent",
                aggfunc="last",
            ).ffill()

            st.line_chart(chart_data, use_container_width=True)
        except Exception as e:
            logger.error(f"Chart error: {e}")
            st.warning("Could not render chart")

    # State distribution pie/bar
    if "current_state" in db_data.columns:
        st.subheader("📊 State Distribution")
        state_counts = db_data.groupby(
            ["equipment_id", "current_state"]
        ).size().reset_index(name="count")

        if not state_counts.empty:
            st.bar_chart(
                state_counts.pivot(
                    index="equipment_id",
                    columns="current_state",
                    values="count",
                ).fillna(0),
                use_container_width=True,
            )


def render_activity_breakdown(db_data: pd.DataFrame):
    """Render activity distribution chart."""
    if db_data is None or db_data.empty:
        return

    if "current_activity" not in db_data.columns:
        return

    st.subheader("🔧 Activity Distribution")

    activity_counts = db_data.groupby(
        ["equipment_id", "current_activity"]
    ).size().reset_index(name="count")

    if not activity_counts.empty:
        st.bar_chart(
            activity_counts.pivot(
                index="equipment_id",
                columns="current_activity",
                values="count",
            ).fillna(0),
            use_container_width=True,
        )


def _format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
