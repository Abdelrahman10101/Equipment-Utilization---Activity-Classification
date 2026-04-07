"""
Activity Classifier — Temporal Pattern Enhanced

Classifies specific work activities for construction equipment by combining
rule-based heuristics with temporal flow pattern analysis from the
segmentation-enhanced motion analyzer.

The classifier uses TWO layers of analysis:
1. Instantaneous: region-based motion features (upper/middle/lower activity)
2. Temporal: flow direction patterns over a sliding window (digging cycles, etc.)

Activities:
- DIGGING: Arm moves downward then upward (scoop cycle), tracks stationary
- SWINGING_LOADING: Lateral/rotational motion with arm activity
- DUMPING: Arm moves upward / bed tilts, then release
- WAITING: No significant motion for sustained period
- TRAVELING: Full body motion (driving/repositioning)
"""
import logging
from typing import Dict, Any, Optional, Tuple
from collections import deque, Counter
from enum import Enum

from config import Config

logger = logging.getLogger("ActivityClassifier")


class Activity(str, Enum):
    DIGGING = "DIGGING"
    SWINGING_LOADING = "SWINGING_LOADING"
    DUMPING = "DUMPING"
    WAITING = "WAITING"
    TRAVELING = "TRAVELING"
    IDLE = "IDLE"


class ActivityClassifier:
    """
    Hybrid activity classifier combining:
      1. Rule-based heuristics on per-frame motion features
      2. Temporal pattern matching from flow direction history
      3. Sliding-window smoothing to reduce classification flicker
    """

    def __init__(self):
        self.window_size = Config.SLIDING_WINDOW_SIZE
        # Per-equipment sliding window of motion features
        self.feature_windows: Dict[str, deque] = {}
        # Per-equipment activity history for smoothing
        self.activity_history: Dict[str, deque] = {}

        logger.info("ActivityClassifier (temporal-enhanced) initialized")

    def classify(
        self,
        equipment_id: str,
        equipment_class: str,
        motion: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Classify the current activity of a piece of equipment.

        Args:
            equipment_id: Unique equipment identifier
            equipment_class: Type of equipment (excavator, dump_truck, etc.)
            motion: Motion analysis result from MotionAnalyzer (includes
                    temporal_patterns if available)

        Returns:
            Dict with 'activity', 'confidence', and 'raw_activity' keys
        """
        # Initialize windows if needed
        if equipment_id not in self.feature_windows:
            self.feature_windows[equipment_id] = deque(maxlen=self.window_size)
            self.activity_history[equipment_id] = deque(maxlen=self.window_size)

        # Store current motion features
        self.feature_windows[equipment_id].append(motion)

        # Get temporal patterns (from MotionAnalyzer's flow direction history)
        temporal = motion.get("temporal_patterns", {})

        # Classify based on equipment type
        if equipment_class in ("excavator", "backhoe"):
            activity, confidence = self._classify_excavator(
                equipment_id, motion, temporal
            )
        elif equipment_class == "dump_truck":
            activity, confidence = self._classify_dump_truck(
                equipment_id, motion, temporal
            )
        elif equipment_class in ("loader", "bulldozer"):
            activity, confidence = self._classify_loader(
                equipment_id, motion, temporal
            )
        else:
            activity, confidence = self._classify_generic(
                equipment_id, motion, temporal
            )

        # Apply temporal smoothing
        self.activity_history[equipment_id].append(activity)
        smoothed_activity = self._smooth_activity(equipment_id, activity)

        return {
            "activity": smoothed_activity.value,
            "confidence": round(confidence, 2),
            "raw_activity": activity.value,
        }

    def _classify_excavator(
        self,
        equipment_id: str,
        motion: Dict[str, Any],
        temporal: Dict[str, Any],
    ) -> Tuple[Activity, float]:
        """
        Classify excavator activities using both instantaneous motion
        features and temporal flow patterns.

        Decision hierarchy:
        1. WAITING — if not active (temporal stillness confirms)
        2. SWINGING_LOADING — if temporal says sustained_lateral OR
           high lateral asymmetry with upper/middle motion
        3. DIGGING — if temporal says down_then_up cycle OR
           arm moves downward with tracks stationary
        4. DUMPING — if temporal says up_then_release OR
           arm moves upward
        5. TRAVELING — full body motion
        """
        if not motion["is_active"]:
            # Confirm with temporal pattern
            if temporal.get("sustained_stillness", False):
                return Activity.WAITING, 0.95
            return Activity.WAITING, 0.85

        source = motion["motion_source"]
        upper = motion.get("upper_magnitude", 0)
        middle = motion.get("middle_magnitude", 0)
        lower = motion.get("lower_magnitude", 0)
        lateral = motion.get("lateral_asymmetry", 0)
        arm_dir = motion.get("arm_direction", {"dx": 0, "dy": 0})
        temporal_pattern = temporal.get("dominant_pattern", "")
        temporal_confidence = temporal.get("confidence", 0)

        # Check temporal feature window
        window = self.feature_windows[equipment_id]
        recent_dy_avg = 0
        if len(window) >= 3:
            recent_dy_avg = sum(
                w.get("arm_direction", {}).get("dy", 0) for w in window
            ) / len(window)

        # ─── TEMPORAL PATTERN PRIORITY ───
        # If temporal analysis has high confidence, trust it more

        if temporal_confidence >= 0.65:
            if temporal_pattern == "swinging":
                confidence = min(0.95, temporal_confidence + 0.1)
                return Activity.SWINGING_LOADING, confidence

            if temporal_pattern == "digging_cycle":
                confidence = min(0.95, temporal_confidence + 0.1)
                return Activity.DIGGING, confidence

            if temporal_pattern == "dumping_cycle":
                confidence = min(0.90, temporal_confidence + 0.1)
                return Activity.DUMPING, confidence

            if temporal_pattern == "loading_vibration":
                return Activity.SWINGING_LOADING, 0.65

        # ─── INSTANTANEOUS FEATURE RULES (fallback) ───

        # --- SWINGING/LOADING ---
        # High lateral motion + upper/middle activity = swing
        if (lateral > Config.MOTION_THRESHOLD * 1.5 or source == "swing") and (
            upper > Config.ARM_MOTION_THRESHOLD or middle > Config.MOTION_THRESHOLD
        ):
            confidence = min(0.90, 0.6 + lateral * 0.1)
            return Activity.SWINGING_LOADING, confidence

        # --- DIGGING ---
        # Arm active, downward direction, tracks stationary
        if source in ("arm_only", "partial") and arm_dir["dy"] > 0.3:
            confidence = min(0.85, 0.55 + upper * 0.05 + arm_dir["dy"] * 0.2)
            return Activity.DIGGING, confidence

        # Also: upper very active and lower not, generally downward trend
        if upper > Config.ARM_MOTION_THRESHOLD * 2 and lower < Config.TRACK_MOTION_THRESHOLD:
            if recent_dy_avg > 0:  # Generally downward over time
                return Activity.DIGGING, 0.65

        # --- DUMPING ---
        # Arm moving upward (negative dy in image coordinates)
        if source in ("arm_only", "partial") and arm_dir["dy"] < -0.3:
            confidence = min(0.85, 0.5 + abs(arm_dir["dy"]) * 0.2)
            return Activity.DUMPING, confidence

        # --- TRAVELING ---
        # Tracks active, all regions moving
        if source in ("full_body", "tracks_only") and lower > Config.TRACK_MOTION_THRESHOLD:
            return Activity.TRAVELING, 0.8

        # Default: if arm is moving, assume digging (most common excavator activity)
        if source == "arm_only":
            return Activity.DIGGING, 0.45

        return Activity.WAITING, 0.4

    def _classify_dump_truck(
        self,
        equipment_id: str,
        motion: Dict[str, Any],
        temporal: Dict[str, Any],
    ) -> Tuple[Activity, float]:
        """
        Classify dump truck activities.

        Key behaviors:
        - DUMPING: Upper region active (bed tilting), lower inactive
        - TRAVELING: Full body motion (driving)
        - SWINGING_LOADING: Being loaded — oscillating/vibration pattern
        - WAITING: No motion
        """
        if not motion["is_active"]:
            if temporal.get("sustained_stillness", False):
                return Activity.WAITING, 0.95
            return Activity.WAITING, 0.85

        source = motion["motion_source"]
        upper = motion.get("upper_magnitude", 0)
        lower = motion.get("lower_magnitude", 0)
        temporal_pattern = temporal.get("dominant_pattern", "")
        temporal_confidence = temporal.get("confidence", 0)

        # Temporal pattern: loading vibration = being loaded
        if temporal_pattern == "loading_vibration" and temporal_confidence >= 0.55:
            return Activity.SWINGING_LOADING, 0.7

        # Temporal pattern: up then release = dumping
        if temporal_pattern == "dumping_cycle" and temporal_confidence >= 0.60:
            return Activity.DUMPING, min(0.90, temporal_confidence + 0.1)

        # DUMPING: bed tilting (upper region moves, lower doesn't)
        if source == "arm_only" and upper > Config.ARM_MOTION_THRESHOLD:
            return Activity.DUMPING, 0.8

        # TRAVELING: full body motion
        if source in ("full_body", "tracks_only") and lower > Config.TRACK_MOTION_THRESHOLD:
            return Activity.TRAVELING, 0.85

        # SWINGING_LOADING: being loaded (vibration patterns from instantaneous features)
        if motion.get("overall_magnitude", 0) > Config.MOTION_THRESHOLD * 0.5:
            window = self.feature_windows[equipment_id]
            if len(window) >= 3:
                mags = [w.get("overall_magnitude", 0) for w in window]
                mean_mag = sum(mags) / len(mags)
                variance = sum((m - mean_mag) ** 2 for m in mags) / len(mags)
                if variance > 0.5:  # High variance = vibration
                    return Activity.SWINGING_LOADING, 0.6

        return Activity.WAITING, 0.5

    def _classify_loader(
        self,
        equipment_id: str,
        motion: Dict[str, Any],
        temporal: Dict[str, Any],
    ) -> Tuple[Activity, float]:
        """Classify loader/bulldozer activities."""
        if not motion["is_active"]:
            if temporal.get("sustained_stillness", False):
                return Activity.WAITING, 0.95
            return Activity.WAITING, 0.85

        source = motion["motion_source"]
        temporal_pattern = temporal.get("dominant_pattern", "")
        temporal_confidence = temporal.get("confidence", 0)

        # Temporal pattern matching
        if temporal_confidence >= 0.65:
            if temporal_pattern == "digging_cycle":
                return Activity.DIGGING, min(0.85, temporal_confidence + 0.1)
            if temporal_pattern == "dumping_cycle":
                return Activity.DUMPING, min(0.85, temporal_confidence + 0.1)

        if source == "arm_only":
            arm_dir = motion.get("arm_direction", {"dy": 0})
            if arm_dir["dy"] > 0:
                return Activity.DIGGING, 0.7
            else:
                return Activity.DUMPING, 0.7

        if source in ("full_body", "tracks_only"):
            return Activity.TRAVELING, 0.8

        return Activity.WAITING, 0.5

    def _classify_generic(
        self,
        equipment_id: str,
        motion: Dict[str, Any],
        temporal: Dict[str, Any],
    ) -> Tuple[Activity, float]:
        """Generic classifier for unknown equipment types."""
        if not motion["is_active"]:
            return Activity.WAITING, 0.9

        if motion.get("overall_magnitude", 0) > Config.MOTION_THRESHOLD * 2:
            return Activity.TRAVELING, 0.6

        return Activity.WAITING, 0.5

    def _smooth_activity(
        self, equipment_id: str, current: Activity
    ) -> Activity:
        """
        Apply temporal smoothing to activity classification.
        Uses majority voting over recent history to reduce flickering.
        """
        history = self.activity_history.get(equipment_id, [])
        if len(history) < 3:
            return current

        # Count recent activities
        recent = list(history)[-5:]  # Last 5 frames
        counts = Counter(recent)
        most_common = counts.most_common(1)[0][0]

        # If current matches most common, keep it
        # Otherwise, keep most common (reduces flicker)
        if counts[current] >= len(recent) * 0.4:
            return current
        return most_common

    def reset(self):
        """Reset state for a new video."""
        self.feature_windows.clear()
        self.activity_history.clear()
