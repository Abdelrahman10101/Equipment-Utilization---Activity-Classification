"""
CV Processor - Configuration
"""
import os


class Config:
    # Kafka
    KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
    RAW_FRAMES_TOPIC = os.getenv("KAFKA_RAW_FRAMES_TOPIC", "raw-frames")
    EQUIPMENT_EVENTS_TOPIC = os.getenv("KAFKA_EQUIPMENT_EVENTS_TOPIC", "equipment-events")
    ANNOTATED_FRAMES_TOPIC = os.getenv("KAFKA_ANNOTATED_FRAMES_TOPIC", "annotated-frames")
    CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "cv-processor-group")

    # YOLO11 (medium) — +6.6 mAP over v8s, fewer params
    YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11m.pt")
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))
    IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.5"))

    # Equipment class mapping from COCO to our labels
    # COCO classes: car=2, bus=5, truck=7
    COCO_EQUIPMENT_CLASSES = {
        7: "dump_truck",     # truck
        5: "dump_truck",     # bus (can resemble large trucks)
        2: "vehicle",        # car
    }

    # Motion analysis
    MOTION_THRESHOLD = float(os.getenv("MOTION_THRESHOLD", "2.0"))
    ARM_MOTION_THRESHOLD = float(os.getenv("ARM_MOTION_THRESHOLD", "1.5"))
    TRACK_MOTION_THRESHOLD = float(os.getenv("TRACK_MOTION_THRESHOLD", "2.5"))

    # State management
    STATE_DEBOUNCE_FRAMES = int(os.getenv("STATE_DEBOUNCE_FRAMES", "5"))
    SLIDING_WINDOW_SIZE = int(os.getenv("SLIDING_WINDOW_SIZE", "10"))

    # Kafka message size
    MAX_MESSAGE_SIZE = int(os.getenv("MAX_MESSAGE_SIZE", str(2 * 1024 * 1024)))
