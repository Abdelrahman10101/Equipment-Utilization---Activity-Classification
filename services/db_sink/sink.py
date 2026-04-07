"""
DB Sink Microservice

Consumes equipment events from Kafka and batch-inserts them
into TimescaleDB for persistent storage and dashboard queries.
"""
import sys
import time
import json
import logging
import signal

from confluent_kafka import Consumer, KafkaError

from config import Config
from models import create_connection, batch_insert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DBSink")

# Graceful shutdown
shutdown = False


def signal_handler(sig, frame):
    global shutdown
    logger.info("Shutdown signal received...")
    shutdown = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def create_consumer() -> Consumer:
    """Create Kafka consumer."""
    conf = {
        "bootstrap.servers": Config.KAFKA_BROKER,
        "group.id": Config.CONSUMER_GROUP,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
        "auto.commit.interval.ms": 5000,
    }

    max_retries = 30
    for attempt in range(max_retries):
        try:
            consumer = Consumer(conf)
            consumer.subscribe([Config.EQUIPMENT_EVENTS_TOPIC])
            logger.info(
                f"Subscribed to {Config.EQUIPMENT_EVENTS_TOPIC} "
                f"(group: {Config.CONSUMER_GROUP})"
            )
            return consumer
        except Exception as e:
            logger.warning(f"Connection attempt {attempt + 1}/{max_retries}: {e}")
            time.sleep(2)

    logger.error("Failed to create consumer")
    sys.exit(1)


def main():
    """Main sink loop."""
    logger.info("=" * 60)
    logger.info("Eagle Vision — DB Sink Starting")
    logger.info("=" * 60)

    # Connect to TimescaleDB
    conn = create_connection()

    # Create Kafka consumer
    consumer = create_consumer()

    # Batch buffer
    batch = []
    last_flush_time = time.time()
    total_inserted = 0

    logger.info("Waiting for equipment events...")

    while not shutdown:
        msg = consumer.poll(timeout=0.5)

        if msg is not None:
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error(f"Consumer error: {msg.error()}")
                continue

            try:
                event = json.loads(msg.value().decode("utf-8"))
                batch.append(event)
            except Exception as e:
                logger.error(f"Failed to parse event: {e}")
                continue

        # Flush batch if size or time threshold met
        now = time.time()
        should_flush = (
            len(batch) >= Config.BATCH_SIZE
            or (batch and now - last_flush_time >= Config.FLUSH_INTERVAL_SECONDS)
        )

        if should_flush:
            try:
                count = batch_insert(conn, batch)
                total_inserted += count
                logger.info(
                    f"Inserted {count} events (total: {total_inserted})"
                )
            except Exception as e:
                logger.error(f"Insert failed, reconnecting: {e}")
                try:
                    conn.close()
                except Exception:
                    pass
                conn = create_connection()
            finally:
                batch = []
                last_flush_time = now

    # Final flush
    if batch:
        try:
            count = batch_insert(conn, batch)
            total_inserted += count
            logger.info(f"Final flush: {count} events")
        except Exception as e:
            logger.error(f"Final flush failed: {e}")

    consumer.close()
    conn.close()
    logger.info(f"DB Sink shutdown. Total events inserted: {total_inserted}")


if __name__ == "__main__":
    main()
