"""
Dashboard - Status Panel Component

Displays live status of each detected machine.
"""
import streamlit as st
import pandas as pd


def render_status_panel(equipment_states: dict):
    """
    Render the equipment status panel showing current state of each machine.

    Args:
        equipment_states: Dict of equipment_id -> state info
    """
    if not equipment_states:
        st.info("🔍 No equipment detected yet...")
        return

    # Build dataframe
    rows = []
    for eid, state in equipment_states.items():
        current_state = state.get("state", state.get("current_state", "INACTIVE"))
        activity = state.get("activity", state.get("current_activity", "WAITING"))
        utilization = state.get("utilization", state.get("utilization_percent", 0))

        rows.append({
            "Equipment ID": eid,
            "Class": state.get("equipment_class", "unknown"),
            "Status": f"🟢 {current_state}" if current_state == "ACTIVE" else f"🔴 {current_state}",
            "Activity": _activity_emoji(activity) + " " + activity,
            "Utilization": f"{utilization:.1f}%",
        })

    df = pd.DataFrame(rows)

    # Custom styling
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Equipment ID": st.column_config.TextColumn("ID", width="small"),
            "Class": st.column_config.TextColumn("Type", width="small"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
            "Activity": st.column_config.TextColumn("Activity", width="medium"),
            "Utilization": st.column_config.TextColumn("Util %", width="small"),
        },
    )


def render_status_cards(equipment_states: dict):
    """Render status as individual cards for a more visual layout."""
    if not equipment_states:
        st.info("🔍 No equipment detected yet...")
        return

    cols = st.columns(min(len(equipment_states), 3))

    for idx, (eid, state) in enumerate(equipment_states.items()):
        col = cols[idx % len(cols)]
        current_state = state.get("state", state.get("current_state", "INACTIVE"))
        activity = state.get("activity", state.get("current_activity", "WAITING"))
        utilization = state.get("utilization", state.get("utilization_percent", 0))
        eq_class = state.get("equipment_class", "unknown")

        is_active = current_state == "ACTIVE"
        border_color = "#00ff88" if is_active else "#ff4444"
        bg_color = "rgba(0, 255, 136, 0.1)" if is_active else "rgba(255, 68, 68, 0.1)"
        status_icon = "🟢" if is_active else "🔴"

        with col:
            st.markdown(
                f"""
                <div style="
                    background: {bg_color};
                    border: 2px solid {border_color};
                    border-radius: 12px;
                    padding: 16px;
                    margin-bottom: 12px;
                    text-align: center;
                ">
                    <h4 style="margin: 0; color: #fff;">{eid}</h4>
                    <p style="margin: 4px 0; color: #aaa; font-size: 0.85em;">
                        {_class_emoji(eq_class)} {eq_class.replace('_', ' ').title()}
                    </p>
                    <h2 style="margin: 8px 0; color: {border_color};">
                        {status_icon} {current_state}
                    </h2>
                    <p style="margin: 4px 0; color: #ddd; font-size: 1.1em;">
                        {_activity_emoji(activity)} {activity.replace('_', ' ')}
                    </p>
                    <div style="
                        background: rgba(255,255,255,0.1);
                        border-radius: 8px;
                        padding: 8px;
                        margin-top: 8px;
                    ">
                        <span style="font-size: 1.3em; font-weight: bold; color: #fff;">
                            {utilization:.1f}%
                        </span>
                        <br/>
                        <span style="color: #888; font-size: 0.8em;">Utilization</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _activity_emoji(activity: str) -> str:
    """Map activity to emoji."""
    emoji_map = {
        "DIGGING": "⛏️",
        "SWINGING_LOADING": "🔄",
        "DUMPING": "📦",
        "TRAVELING": "🚛",
        "WAITING": "⏸️",
        "IDLE": "💤",
    }
    return emoji_map.get(activity, "❓")


def _class_emoji(eq_class: str) -> str:
    """Map equipment class to emoji."""
    emoji_map = {
        "excavator": "🏗️",
        "dump_truck": "🚚",
        "loader": "🚜",
        "bulldozer": "🚜",
        "crane": "🏗️",
        "vehicle": "🚗",
    }
    return emoji_map.get(eq_class, "🔧")
