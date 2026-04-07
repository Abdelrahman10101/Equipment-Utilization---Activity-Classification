"""
Frame Producer Microservice

Reads video files from a directory, extracts frames at a configurable rate,
encodes them as base64 JPEG, and publishes them to a Kafka topic.
"""
import os
import sys
import time
import json
import glob
import base64
import logging
import signal

import cv2
from confluent_kafka import Producer, KafkaError

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("FrameProducer")

# Graceful shutdown
shutdown = False


def signal_handler(sig, frame):
    global shutdown
    logger.info("Shutdown signal received, finishing current video...")
    shutdown = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def create_producer() -> Producer:
    """Create and return a Kafka producer with retry logic."""
    conf = {
        "bootstrap.servers": Config.KAFKA_BROKER,
        "message.max.bytes": Config.MAX_MESSAGE_SIZE,
        "queue.buffering.max.messages": 1000,
        "queue.buffering.max.kbytes": 1048576,  # 1GB buffer
        "batch.num.messages": 10,
        "linger.ms": 50,
        "compression.type": "lz4",
        "acks": "all",
    }

    max_retries = 30
    for attempt in range(max_retries):
        try:
            producer = Producer(conf)
            # Test connection by listing topics
            logger.info(f"Connected to Kafka broker at {Config.KAFKA_BROKER}")
            return producer
        except Exception as e:
            logger.warning(
                f"Kafka connection attempt {attempt + 1}/{max_retries} failed: {e}"
            )
            time.sleep(2)

    logger.error("Failed to connect to Kafka after all retries")
    sys.exit(1)


def delivery_callback(err, msg):
    """Callback for Kafka message delivery."""
    if err is not None:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.debug(
            f"Message delivered to {msg.topic()} [{msg.partition()}] @ {msg.offset()}"
        )


def get_video_files(video_dir: str) -> list:
    """Find all video files in the specified directory."""
    extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
    video_files.sort()
    return video_files


def process_video(producer: Producer, video_path: str, video_index: int):
    """Process a single video file and publish frames to Kafka."""
    global shutdown

    video_name = os.path.basename(video_path)
    logger.info(f"Processing video: {video_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / original_fps if original_fps > 0 else 0

    logger.info(
        f"  Video info: {width}x{height}, {original_fps:.1f} FPS, "
        f"{total_frames} frames, {duration:.1f}s duration"
    )

    frame_count = 0
    published_count = 0

    # Calculate frame skip to achieve target FPS
    frame_skip = max(1, int(original_fps / Config.TARGET_FPS)) if original_fps > 0 else Config.FRAME_SKIP

    while not shutdown:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames to match target FPS
        if frame_count % frame_skip != 0:
            continue

        # Resize frame
        frame_resized = cv2.resize(
            frame, (Config.FRAME_RESIZE_WIDTH, Config.FRAME_RESIZE_HEIGHT)
        )

        # Encode as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY]
        _, buffer = cv2.imencode(".jpg", frame_resized, encode_params)
        frame_b64 = base64.b64encode(buffer).decode("utf-8")

        # Calculate video timestamp
        timestamp_sec = frame_count / original_fps if original_fps > 0 else 0
        hours = int(timestamp_sec // 3600)
        minutes = int((timestamp_sec % 3600) // 60)
        seconds = timestamp_sec % 60
        timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

        # Create message payload
        message = {
            "frame_id": frame_count,
            "video_name": video_name,
            "video_index": video_index,
            "timestamp": timestamp_str,
            "timestamp_sec": round(timestamp_sec, 3),
            "original_fps": round(original_fps, 2),
            "frame_width": Config.FRAME_RESIZE_WIDTH,
            "frame_height": Config.FRAME_RESIZE_HEIGHT,
            "frame_data": frame_b64,
        }

        # Publish to Kafka
        try:
            producer.produce(
                Config.RAW_FRAMES_TOPIC,
                key=video_name.encode("utf-8"),
                value=json.dumps(message).encode("utf-8"),
                callback=delivery_callback,
            )
            producer.poll(0)
            published_count += 1

            if published_count % 50 == 0:
                logger.info(
                    f"  Published {published_count} frames "
                    f"(frame {frame_count}/{total_frames}, "
                    f"time {timestamp_str})"
                )

        except BufferError:
            logger.warning("Kafka buffer full, waiting...")
            producer.flush(timeout=5)
            producer.produce(
                Config.RAW_FRAMES_TOPIC,
                key=video_name.encode("utf-8"),
                value=json.dumps(message).encode("utf-8"),
                callback=delivery_callback,
            )

        # Throttle to approximate real-time playback
        time.sleep(1.0 / Config.TARGET_FPS)

    cap.release()

    # Send end-of-video sentinel
    sentinel = {
        "frame_id": -1,
        "video_name": video_name,
        "video_index": video_index,
        "timestamp": timestamp_str if 'timestamp_str' in dir() else "00:00:00.000",
        "is_end_of_video": True,
        "total_frames_published": published_count,
    }
    producer.produce(
        Config.RAW_FRAMES_TOPIC,
        key=video_name.encode("utf-8"),
        value=json.dumps(sentinel).encode("utf-8"),
        callback=delivery_callback,
    )
    producer.flush(timeout=10)

    logger.info(
        f"  Finished video: {video_name} — "
        f"{published_count} frames published from {frame_count} total"
    )


def main():
    """Main entry point for the Frame Producer."""
    logger.info("=" * 60)
    logger.info("Eagle Vision — Frame Producer Starting")
    logger.info("=" * 60)

    # Find video files
    video_files = get_video_files(Config.VIDEO_DIR)
    if not video_files:
        logger.error(f"No video files found in {Config.VIDEO_DIR}")
        logger.info("Waiting for videos to appear...")
        while not shutdown:
            video_files = get_video_files(Config.VIDEO_DIR)
            if video_files:
                break
            time.sleep(5)

    if shutdown:
        return

    logger.info(f"Found {len(video_files)} video(s):")
    for vf in video_files:
        logger.info(f"  - {os.path.basename(vf)}")

    # Create Kafka producer
    producer = create_producer()

    # Process each video
    for idx, video_path in enumerate(video_files):
        if shutdown:
            break
        process_video(producer, video_path, idx)

    # Send global end-of-stream
    eos = {
        "frame_id": -999,
        "is_end_of_stream": True,
        "total_videos": len(video_files),
    }
    producer.produce(
        Config.RAW_FRAMES_TOPIC,
        key=b"__EOS__",
        value=json.dumps(eos).encode("utf-8"),
        callback=delivery_callback,
    )
    producer.flush(timeout=30)

    logger.info("All videos processed. Frame Producer shutting down.")


if __name__ == "__main__":
    main()
