import os
import sys
import time
import json
import base64
import logging
import signal
import cv2
import numpy as np
from confluent_kafka import Consumer, KafkaError
from config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("VideoSink")

shutdown = False

def signal_handler(sig, frame):
    global shutdown
    logger.info("Shutdown signal received...")
    shutdown = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def create_consumer() -> Consumer:
    conf = {
        "bootstrap.servers": Config.KAFKA_BROKER,
        "group.id": Config.CONSUMER_GROUP,
        "auto.offset.reset": "earliest",
        "fetch.message.max.bytes": 2 * 1024 * 1024,
    }
    max_retries = 30
    for attempt in range(max_retries):
        try:
            consumer = Consumer(conf)
            consumer.subscribe([Config.ANNOTATED_FRAMES_TOPIC])
            return consumer
        except Exception as e:
            time.sleep(2)
    sys.exit(1)

def build_output_path(video_name: str) -> str:
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_name))[0]
    return os.path.join(Config.OUTPUT_DIR, f"{base}_annotated.mp4")

def main():
    logger.info("Video Sink Starting")
    consumer = create_consumer()
    writers = {}
    
    while not shutdown:
        msg = consumer.poll(timeout=0.5)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            continue
        
        try:
            data = json.loads(msg.value().decode("utf-8"))
        except Exception as e:
            logger.error(f"Failed to parse event: {e}")
            continue

        video_name = data.get("video_name", "unknown")
        
        if data.get("is_end_of_video"):
            logger.info(f"Closing video writer for {video_name}")
            if video_name in writers:
                writers[video_name].release()
                del writers[video_name]
            continue
            
        if data.get("is_end_of_stream"):
            continue

        frame_b64 = data.get("frame_data")
        if not frame_b64:
            continue
            
        frame_bytes = base64.b64decode(frame_b64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            continue
            
        if video_name not in writers:
            h, w = frame.shape[:2]
            output_path = build_output_path(video_name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writers[video_name] = cv2.VideoWriter(output_path, fourcc, Config.TARGET_FPS, (w, h))
            logger.info(f"Started rendering: {output_path}")
            
        writers[video_name].write(frame)

    # Cleanup
    for v_name, writer in writers.items():
        writer.release()
    consumer.close()

if __name__ == "__main__":
    main()
