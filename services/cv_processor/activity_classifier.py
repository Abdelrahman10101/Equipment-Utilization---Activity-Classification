"""
Activity Classifier

Classifies specific work activities for construction equipment based on
motion analysis results. Uses a hybrid approach combining rule-based
heuristics with temporal pattern matching.

Activities:
- DIGGING: Arm moves downward, tracks stationary (excavator)
- SWINGING_LOADING: Lateral/rotational motion with arm activity
- DUMPING: Arm moves upward / bed tilts (excavator/dump truck)
- WAITING: No significant motion for sustained period
"""
import logging
from typing import Dict, Any, Optional
from collections import deque
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
    Rule-based activity classifier using motion analysis features.
    Maintains a sliding window of motion states per equipment for
    temporal pattern matching.
    """

    def __init__(self):
        self.window_size = Config.SLIDING_WINDOW_SIZE
        # Per-equipment sliding window of motion features
        self.feature_windows: Dict[str, deque] = {}
        # Per-equipment activity history for smoothing
        self.activity_history: Dict[str, deque] = {}

        logger.info("ActivityClassifier initialized")

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
            motion: Motion analysis result from MotionAnalyzer

        Returns:
            Dict with 'activity' and 'confidence' keys
        """
        # Initialize windows if needed
        if equipment_id not in self.feature_windows:
            self.feature_windows[equipment_id] = deque(maxlen=self.window_size)
            self.activity_history[equipment_id] = deque(maxlen=self.window_size)

        # Store current motion features
        self.feature_windows[equipment_id].append(motion)

        # Classify based on equipment type
        if equipment_class in ("excavator", "backhoe"):
            activity, confidence = self._classify_excavator(equipment_id, motion)
        elif equipment_class == "dump_truck":
            activity, confidence = self._classify_dump_truck(equipment_id, motion)
        elif equipment_class in ("loader", "bulldozer"):
            activity, confidence = self._classify_loader(equipment_id, motion)
        else:
            activity, confidence = self._classify_generic(equipment_id, motion)

        # Apply temporal smoothing
        self.activity_history[equipment_id].append(activity)
        smoothed_activity = self._smooth_activity(equipment_id, activity)

        return {
            "activity": smoothed_activity.value,
            "confidence": round(confidence, 2),
            "raw_activity": activity.value,
        }

    def _classify_excavator(
        self, equipment_id: str, motion: Dict[str, Any]
    ) -> tuple:
        """
        Classify excavator activities using motion region analysis.

        Key heuristics:
        - DIGGING: Upper region active (arm down), lower region inactive
        - SWINGING_LOADING: High lateral asymmetry + middle/upper active
        - DUMPING: Upper region active (arm up direction)
        - WAITING: All regions below threshold
        """
        if not motion["is_active"]:
            return Activity.WAITING, 0.9

        source = motion["motion_source"]
        upper = motion.get("upper_magnitude", 0)
        middle = motion.get("middle_magnitude", 0)
        lower = motion.get("lower_magnitude", 0)
        lateral = motion.get("lateral_asymmetry", 0)
        arm_dir = motion.get("arm_direction", {"dx": 0, "dy": 0})

        # Check temporal features
        window = self.feature_windows[equipment_id]
        recent_upper_avg = 0
        recent_dy_avg = 0
        if len(window) >= 3:
            recent_upper_avg = sum(
                w.get("upper_magnitude", 0) for w in window
            ) / len(window)
            recent_dy_avg = sum(
                w.get("arm_direction", {}).get("dy", 0) for w in window
            ) / len(window)

        # --- SWINGING/LOADING ---
        # High lateral motion + upper/middle activity = swing
        if (lateral > Config.MOTION_THRESHOLD * 1.5 or source == "swing") and (
            upper > Config.ARM_MOTION_THRESHOLD or middle > Config.MOTION_THRESHOLD
        ):
            confidence = min(0.95, 0.6 + lateral * 0.1)
            return Activity.SWINGING_LOADING, confidence

        # --- DIGGING ---
        # Arm (upper) active, downward direction, tracks (lower) stationary
        if source in ("arm_only", "partial") and arm_dir["dy"] > 0.3:
            confidence = min(0.95, 0.6 + upper * 0.05 + arm_dir["dy"] * 0.2)
            return Activity.DIGGING, confidence

        # Also classify as digging if upper is very active and lower is not
        if upper > Config.ARM_MOTION_THRESHOLD * 2 and lower < Config.TRACK_MOTION_THRESHOLD:
            if recent_dy_avg > 0:  # Generally downward over time
                return Activity.DIGGING, 0.7

        # --- DUMPING ---
        # Arm moving upward (negative dy in image coordinates)
        if source in ("arm_only", "partial") and arm_dir["dy"] < -0.3:
            confidence = min(0.9, 0.5 + abs(arm_dir["dy"]) * 0.2)
            return Activity.DUMPING, confidence

        # --- TRAVELING ---
        # Lower region (tracks) active, all regions moving similarly
        if source in ("full_body", "tracks_only") and lower > Config.TRACK_MOTION_THRESHOLD:
            return Activity.TRAVELING, 0.8

        # Default: if active but no specific pattern, likely transitioning
        if source == "arm_only":
            return Activity.DIGGING, 0.5  # Default arm motion = assumed digging
        
        return Activity.WAITING, 0.4

    def _classify_dump_truck(
        self, equipment_id: str, motion: Dict[str, Any]
    ) -> tuple:
        """
        Classify dump truck activities.

        Key heuristics:
        - DUMPING: Upper region active (bed tilting), lower inactive
        - TRAVELING: Full body motion (driving)
        - WAITING: No motion
        """
        if not motion["is_active"]:
            return Activity.WAITING, 0.9

        source = motion["motion_source"]
        upper = motion.get("upper_magnitude", 0)
        lower = motion.get("lower_magnitude", 0)

        # DUMPING: bed tilting (upper region moves, lower doesn't)
        if source == "arm_only" and upper > Config.ARM_MOTION_THRESHOLD:
            return Activity.DUMPING, 0.8

        # TRAVELING: full body motion
        if source in ("full_body", "tracks_only") and lower > Config.TRACK_MOTION_THRESHOLD:
            return Activity.TRAVELING, 0.85

        # SWINGING_LOADING: being loaded (vibration patterns)
        if motion.get("overall_magnitude", 0) > Config.MOTION_THRESHOLD * 0.5:
            # Check for vibration pattern (loading causes shaking)
            window = self.feature_windows[equipment_id]
            if len(window) >= 3:
                mags = [w.get("overall_magnitude", 0) for w in window]
                variance = sum((m - sum(mags)/len(mags))**2 for m in mags) / len(mags)
                if variance > 0.5:  # High variance = vibration
                    return Activity.SWINGING_LOADING, 0.6

        return Activity.WAITING, 0.5

    def _classify_loader(
        self, equipment_id: str, motion: Dict[str, Any]
    ) -> tuple:
        """Classify loader/bulldozer activities."""
        if not motion["is_active"]:
            return Activity.WAITING, 0.9

        source = motion["motion_source"]
        upper = motion.get("upper_magnitude", 0)
        lower = motion.get("lower_magnitude", 0)

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
        self, equipment_id: str, motion: Dict[str, Any]
    ) -> tuple:
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
        from collections import Counter
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
