"""
CV Processor - Main Pipeline

Consumes frames from Kafka, runs the full CV pipeline:
  Decode → Detect → Track → Analyze Motion → Classify Activity → Update State

Publishes:
  - equipment-events: JSON analysis results per equipment per frame
  - annotated-frames: Frames with bounding boxes and labels drawn
"""
import sys
import time
import json
import base64
import logging
import signal
from typing import Optional

import cv2
import numpy as np
from confluent_kafka import Consumer, Producer, KafkaError

from config import Config
from detector import EquipmentDetector
from motion_analyzer import MotionAnalyzer
from activity_classifier import ActivityClassifier
from state_manager import StateManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("CVProcessor")

# Graceful shutdown
shutdown = False


def signal_handler(sig, frame):
    global shutdown
    logger.info("Shutdown signal received...")
    shutdown = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ── Colors for visualization ──
COLORS = {
    "ACTIVE": (0, 255, 0),        # Green
    "INACTIVE": (0, 0, 255),      # Red
    "DIGGING": (0, 200, 255),     # Orange
    "SWINGING_LOADING": (255, 200, 0),  # Cyan
    "DUMPING": (200, 0, 255),     # Purple
    "TRAVELING": (255, 255, 0),   # Yellow
    "WAITING": (128, 128, 128),   # Gray
    "IDLE": (128, 128, 128),
}


def draw_annotations(
    frame: np.ndarray,
    detections: list,
    states: dict,
) -> np.ndarray:
    """Draw bounding boxes, labels, and status on frame."""
    annotated = frame.copy()

    for det in detections:
        bbox = det["bbox"]
        equipment_id = det.get("equipment_id", "UNK")
        equipment_class = det.get("equipment_class", "unknown")

        # Get state info
        state_info = states.get(equipment_id, {})
        current_state = state_info.get("current_state", "INACTIVE")
        activity = state_info.get("current_activity", "WAITING")
        utilization = state_info.get("utilization_percent", 0)
        motion_source = state_info.get("motion_source", "none")

        # Choose color based on state
        color = COLORS.get(current_state, (255, 255, 255))

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Draw bounding box
        thickness = 3 if current_state == "ACTIVE" else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Draw label background
        label_lines = [
            f"{equipment_id} | {equipment_class}",
            f"State: {current_state} | {activity}",
            f"Util: {utilization:.1f}% | Motion: {motion_source}",
        ]

        y_offset = y1 - 10
        for i, line in enumerate(reversed(label_lines)):
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            y_pos = y_offset - i * 22

            # Background rectangle
            cv2.rectangle(
                annotated,
                (x1, y_pos - 15),
                (x1 + text_size[0] + 8, y_pos + 5),
                color,
                -1,
            )
            # Text
            cv2.putText(
                annotated,
                line,
                (x1 + 4, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # Draw motion indicator (small bar showing motion magnitude)
        motion = det.get("motion", {})
        mag = motion.get("overall_magnitude", 0)
        bar_width = min(int(mag * 20), x2 - x1)
        cv2.rectangle(
            annotated,
            (x1, y2 + 2),
            (x1 + bar_width, y2 + 8),
            COLORS.get(activity, (255, 255, 255)),
            -1,
        )

    # Draw global info overlay
    _draw_global_overlay(annotated, states)

    return annotated


def _draw_global_overlay(frame: np.ndarray, states: dict):
    """Draw global statistics overlay in top-right corner."""
    if not states:
        return

    h, w = frame.shape[:2]
    active_count = sum(1 for s in states.values() if s.get("current_state") == "ACTIVE")
    total_count = len(states)

    overlay_lines = [
        f"Equipment: {total_count} tracked",
        f"Active: {active_count} | Idle: {total_count - active_count}",
    ]

    x_start = w - 300
    y_start = 20

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start - 10, 5), (w - 5, y_start + len(overlay_lines) * 25 + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(overlay_lines):
        cv2.putText(
            frame,
            line,
            (x_start, y_start + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def create_consumer() -> Consumer:
    """Create Kafka consumer with retry logic."""
    conf = {
        "bootstrap.servers": Config.KAFKA_BROKER,
        "group.id": Config.CONSUMER_GROUP,
        "auto.offset.reset": "earliest",
        "max.partition.fetch.bytes": Config.MAX_MESSAGE_SIZE,
        "fetch.message.max.bytes": Config.MAX_MESSAGE_SIZE,
    }

    max_retries = 30
    for attempt in range(max_retries):
        try:
            consumer = Consumer(conf)
            consumer.subscribe([Config.RAW_FRAMES_TOPIC])
            logger.info(
                f"Consumer subscribed to {Config.RAW_FRAMES_TOPIC} "
                f"(group: {Config.CONSUMER_GROUP})"
            )
            return consumer
        except Exception as e:
            logger.warning(f"Consumer connection attempt {attempt + 1}/{max_retries}: {e}")
            time.sleep(2)

    logger.error("Failed to create Kafka consumer")
    sys.exit(1)


def create_producer() -> Producer:
    """Create Kafka producer."""
    conf = {
        "bootstrap.servers": Config.KAFKA_BROKER,
        "message.max.bytes": Config.MAX_MESSAGE_SIZE,
        "compression.type": "lz4",
    }

    max_retries = 30
    for attempt in range(max_retries):
        try:
            producer = Producer(conf)
            logger.info("Kafka producer created")
            return producer
        except Exception as e:
            logger.warning(f"Producer connection attempt {attempt + 1}/{max_retries}: {e}")
            time.sleep(2)

    logger.error("Failed to create Kafka producer")
    sys.exit(1)


def decode_frame(message: dict) -> Optional[np.ndarray]:
    """Decode base64 frame from Kafka message."""
    try:
        frame_b64 = message.get("frame_data")
        if not frame_b64:
            return None
        frame_bytes = base64.b64decode(frame_b64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Failed to decode frame: {e}")
        return None


def main():
    """Main processing loop."""
    logger.info("=" * 60)
    logger.info("Eagle Vision — CV Processor Starting")
    logger.info("=" * 60)

    # Initialize components
    detector = EquipmentDetector()
    motion_analyzer = MotionAnalyzer()
    activity_classifier = ActivityClassifier()
    state_manager = StateManager()

    # Kafka connections
    consumer = create_consumer()
    producer = create_producer()

    frame_count = 0
    start_time = time.time()

    logger.info("Waiting for frames...")

    while not shutdown:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            logger.error(f"Consumer error: {msg.error()}")
            continue

        # Decode message
        try:
            message = json.loads(msg.value().decode("utf-8"))
        except Exception as e:
            logger.error(f"Failed to parse message: {e}")
            continue

        # Handle sentinel messages
        if message.get("is_end_of_video"):
            logger.info(
                f"End of video: {message.get('video_name')} — "
                f"Published {message.get('total_frames_published', 0)} frames"
            )
            # Forward sentinel to downstream consumers
            try:
                producer.produce(
                    Config.ANNOTATED_FRAMES_TOPIC,
                    key=message.get('video_name', 'unknown').encode("utf-8"),
                    value=json.dumps(message).encode("utf-8"),
                )
                producer.flush(timeout=5)
            except Exception as e:
                logger.error(f"Failed to forward sentinel: {e}")
            continue

        if message.get("is_end_of_stream"):
            logger.info("End of stream received. All videos processed.")
            try:
                producer.produce(
                    Config.ANNOTATED_FRAMES_TOPIC,
                    key=b"__EOS__",
                    value=json.dumps(message).encode("utf-8"),
                )
                producer.flush(timeout=5)
            except Exception as e:
                logger.error(f"Failed to forward EOS sentinel: {e}")
            # Don't break — keep listening for more
            continue

        # Decode frame
        frame = decode_frame(message)
        if frame is None:
            continue

        frame_id = message.get("frame_id", 0)
        timestamp = message.get("timestamp", "00:00:00.000")
        timestamp_sec = message.get("timestamp_sec", 0)
        video_name = message.get("video_name", "unknown")

        # ── Step 1: Detect equipment ──
        detections = detector.detect_with_tracking(frame)

        # ── Step 2: Compute optical flow ──
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = motion_analyzer.compute_optical_flow(frame_gray)

        if flow is not None and detections:
            # ── Step 3: Analyze motion per equipment ──
            detections = motion_analyzer.analyze_equipment_motion(
                flow, detections, frame.shape
            )

            # ── Step 4: Classify activity & update state ──
            all_states = {}
            for det in detections:
                equipment_id = det.get("equipment_id", "UNK")
                equipment_class = det.get("equipment_class", "unknown")
                motion = det.get("motion", {})

                # Classify activity
                activity_result = activity_classifier.classify(
                    equipment_id, equipment_class, motion
                )

                # Update state
                state_snapshot = state_manager.update_equipment(
                    equipment_id=equipment_id,
                    equipment_class=equipment_class,
                    is_active=motion.get("is_active", False),
                    activity=activity_result["activity"],
                    motion_source=motion.get("motion_source", "none"),
                    timestamp_sec=timestamp_sec,
                    frame_id=frame_id,
                )

                all_states[equipment_id] = state_snapshot

                # ── Publish equipment event to Kafka ──
                event_payload = {
                    "frame_id": frame_id,
                    "equipment_id": equipment_id,
                    "equipment_class": equipment_class,
                    "timestamp": timestamp,
                    "video_name": video_name,
                    "utilization": {
                        "current_state": state_snapshot["current_state"],
                        "current_activity": state_snapshot["current_activity"],
                        "motion_source": state_snapshot["motion_source"],
                    },
                    "time_analytics": {
                        "total_tracked_seconds": state_snapshot["total_tracked_seconds"],
                        "total_active_seconds": state_snapshot["total_active_seconds"],
                        "total_idle_seconds": state_snapshot["total_idle_seconds"],
                        "utilization_percent": state_snapshot["utilization_percent"],
                    },
                    "bbox": {
                        "x": det["bbox"][0],
                        "y": det["bbox"][1],
                        "w": det["bbox"][2] - det["bbox"][0],
                        "h": det["bbox"][3] - det["bbox"][1],
                    },
                    "confidence": det.get("confidence", 0),
                }

                try:
                    producer.produce(
                        Config.EQUIPMENT_EVENTS_TOPIC,
                        key=equipment_id.encode("utf-8"),
                        value=json.dumps(event_payload).encode("utf-8"),
                    )
                except BufferError:
                    producer.flush(timeout=5)

            # ── Step 5: Draw annotations & publish annotated frame ──
            annotated_frame = draw_annotations(frame, detections, all_states)

            # Encode annotated frame
            _, buffer = cv2.imencode(
                ".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            annotated_b64 = base64.b64encode(buffer).decode("utf-8")

            annotated_msg = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "video_name": video_name,
                "frame_data": annotated_b64,
                "equipment_count": len(detections),
                "states": {
                    eid: {
                        "state": s["current_state"],
                        "activity": s["current_activity"],
                        "utilization": s["utilization_percent"],
                    }
                    for eid, s in all_states.items()
                },
            }

            try:
                producer.produce(
                    Config.ANNOTATED_FRAMES_TOPIC,
                    key=video_name.encode("utf-8"),
                    value=json.dumps(annotated_msg).encode("utf-8"),
                )
            except BufferError:
                producer.flush(timeout=5)

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            logger.info(
                f"Processed {frame_count} frames ({fps:.1f} FPS) — "
                f"Tracking {len(state_manager.equipment_states)} equipment"
            )
            producer.flush(timeout=1)

        producer.poll(0)

    # Cleanup
    consumer.close()
    producer.flush(timeout=10)

    # Log final summary
    summary = state_manager.get_summary()
    logger.info("=" * 60)
    logger.info("Final Utilization Summary:")
    logger.info(f"  Total equipment tracked: {summary['total_equipment']}")
    if summary['total_equipment'] > 0:
        logger.info(
            f"  Overall utilization: {summary['overall_utilization_percent']}%"
        )
        for eid, state in summary.get("equipment", {}).items():
            logger.info(
                f"  {eid} ({state['equipment_class']}): "
                f"{state['utilization_percent']}% utilization "
                f"({state['total_active_seconds']}s active / "
                f"{state['total_tracked_seconds']}s total)"
            )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
