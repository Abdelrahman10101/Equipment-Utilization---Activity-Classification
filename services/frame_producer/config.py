"""
Frame Producer - Configuration
"""
import os


class Config:
    # Kafka
    KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
    RAW_FRAMES_TOPIC = os.getenv("KAFKA_RAW_FRAMES_TOPIC", "raw-frames")

    # Video
    VIDEO_DIR = os.getenv("VIDEO_DIR", "/app/sample_videos")
    FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))
    TARGET_FPS = int(os.getenv("TARGET_FPS", "10"))
    FRAME_RESIZE_WIDTH = int(os.getenv("FRAME_RESIZE_WIDTH", "1280"))
    FRAME_RESIZE_HEIGHT = int(os.getenv("FRAME_RESIZE_HEIGHT", "720"))
    JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))

    # Max message size for Kafka (1MB default, we use 2MB)
    MAX_MESSAGE_SIZE = int(os.getenv("MAX_MESSAGE_SIZE", str(2 * 1024 * 1024)))
