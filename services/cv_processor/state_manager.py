"""
State Manager

Manages per-equipment state tracking including:
- ACTIVE/INACTIVE state with debouncing
- Cumulative time tracking (active, idle, total)
- Utilization percentage calculation
- State transition history
"""
import logging
import time
from typing import Dict, Any, Optional
from collections import defaultdict

from config import Config

logger = logging.getLogger("StateManager")


class EquipmentState:
    """Tracks the state of a single piece of equipment."""

    def __init__(self, equipment_id: str, equipment_class: str):
        self.equipment_id = equipment_id
        self.equipment_class = equipment_class

        # State
        self.current_state = "INACTIVE"
        self.current_activity = "WAITING"
        self.motion_source = "none"

        # Time tracking
        self.total_tracked_seconds = 0.0
        self.total_active_seconds = 0.0
        self.total_idle_seconds = 0.0
        self.utilization_percent = 0.0

        # Internal tracking
        self.last_update_time = None
        self.first_seen_time = None
        self.frame_count = 0

        # Debouncing
        self.pending_state = None
        self.pending_state_count = 0
        self.debounce_threshold = Config.STATE_DEBOUNCE_FRAMES

        # State history
        self.transitions = []

    def update(
        self,
        is_active: bool,
        activity: str,
        motion_source: str,
        timestamp_sec: float,
        frame_id: int,
    ) -> Dict[str, Any]:
        """
        Update equipment state with new frame data.

        Args:
            is_active: Whether equipment is active based on motion analysis
            activity: Classified activity string
            motion_source: Where motion was detected
            timestamp_sec: Video timestamp in seconds
            frame_id: Frame number

        Returns:
            Current state snapshot
        """
        self.frame_count += 1

        # Initialize timing
        if self.first_seen_time is None:
            self.first_seen_time = timestamp_sec
            self.last_update_time = timestamp_sec

        # Calculate time delta
        dt = timestamp_sec - self.last_update_time
        if dt < 0:
            dt = 0  # Handle edge cases
        self.last_update_time = timestamp_sec

        # Update cumulative time
        self.total_tracked_seconds = timestamp_sec - self.first_seen_time
        if self.total_tracked_seconds < 0:
            self.total_tracked_seconds = 0

        # Accumulate active/idle time based on CURRENT state (before transition)
        if self.current_state == "ACTIVE":
            self.total_active_seconds += dt
        else:
            self.total_idle_seconds += dt

        # Apply debounced state transition
        new_state = "ACTIVE" if is_active else "INACTIVE"
        self._debounce_state(new_state, timestamp_sec)

        # Update activity and motion source
        if self.current_state == "ACTIVE":
            self.current_activity = activity
            self.motion_source = motion_source
        else:
            self.current_activity = "WAITING"
            self.motion_source = "none"

        # Calculate utilization
        if self.total_tracked_seconds > 0:
            self.utilization_percent = round(
                (self.total_active_seconds / self.total_tracked_seconds) * 100, 1
            )
        else:
            self.utilization_percent = 0.0

        return self.get_snapshot()

    def _debounce_state(self, new_state: str, timestamp_sec: float):
        """
        Apply debouncing to state transitions.
        State must persist for N frames before transitioning.
        """
        if new_state == self.current_state:
            self.pending_state = None
            self.pending_state_count = 0
            return

        if new_state == self.pending_state:
            self.pending_state_count += 1
        else:
            self.pending_state = new_state
            self.pending_state_count = 1

        # Transition when threshold met
        if self.pending_state_count >= self.debounce_threshold:
            old_state = self.current_state
            self.current_state = new_state
            self.pending_state = None
            self.pending_state_count = 0

            self.transitions.append({
                "from": old_state,
                "to": new_state,
                "timestamp_sec": timestamp_sec,
            })

            logger.debug(
                f"  {self.equipment_id}: {old_state} → {new_state} "
                f"at t={timestamp_sec:.1f}s"
            )

    def get_snapshot(self) -> Dict[str, Any]:
        """Return current state as a dictionary."""
        return {
            "equipment_id": self.equipment_id,
            "equipment_class": self.equipment_class,
            "current_state": self.current_state,
            "current_activity": self.current_activity,
            "motion_source": self.motion_source,
            "total_tracked_seconds": round(self.total_tracked_seconds, 1),
            "total_active_seconds": round(self.total_active_seconds, 1),
            "total_idle_seconds": round(self.total_idle_seconds, 1),
            "utilization_percent": round(self.utilization_percent, 1),
            "frame_count": self.frame_count,
        }


class StateManager:
    """Manages state for all tracked equipment."""

    def __init__(self):
        self.equipment_states: Dict[str, EquipmentState] = {}
        logger.info("StateManager initialized")

    def update_equipment(
        self,
        equipment_id: str,
        equipment_class: str,
        is_active: bool,
        activity: str,
        motion_source: str,
        timestamp_sec: float,
        frame_id: int,
    ) -> Dict[str, Any]:
        """
        Update the state of a specific piece of equipment.

        Creates a new EquipmentState if this is the first time
        seeing this equipment_id.
        """
        if equipment_id not in self.equipment_states:
            self.equipment_states[equipment_id] = EquipmentState(
                equipment_id, equipment_class
            )
            logger.info(
                f"New equipment tracked: {equipment_id} ({equipment_class})"
            )

        return self.equipment_states[equipment_id].update(
            is_active=is_active,
            activity=activity,
            motion_source=motion_source,
            timestamp_sec=timestamp_sec,
            frame_id=frame_id,
        )

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Return snapshots of all tracked equipment."""
        return {
            eid: state.get_snapshot()
            for eid, state in self.equipment_states.items()
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return overall utilization summary."""
        if not self.equipment_states:
            return {"total_equipment": 0}

        total_active = sum(
            s.total_active_seconds for s in self.equipment_states.values()
        )
        total_idle = sum(
            s.total_idle_seconds for s in self.equipment_states.values()
        )
        total_tracked = sum(
            s.total_tracked_seconds for s in self.equipment_states.values()
        )

        return {
            "total_equipment": len(self.equipment_states),
            "total_active_seconds": round(total_active, 1),
            "total_idle_seconds": round(total_idle, 1),
            "total_tracked_seconds": round(total_tracked, 1),
            "overall_utilization_percent": round(
                (total_active / total_tracked) * 100, 1
            ) if total_tracked > 0 else 0,
            "equipment": {
                eid: state.get_snapshot()
                for eid, state in self.equipment_states.items()
            },
        }

    def reset(self):
        """Reset all state for a new video."""
        self.equipment_states.clear()
