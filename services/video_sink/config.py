import os

class Config:
    KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
    ANNOTATED_FRAMES_TOPIC = os.getenv("KAFKA_ANNOTATED_FRAMES_TOPIC", "annotated-frames")
    CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "video-sink-group")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/output_videos")
    TARGET_FPS = int(os.getenv("TARGET_FPS", "8"))
