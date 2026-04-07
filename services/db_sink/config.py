"""
DB Sink - Configuration
"""
import os


class Config:
    # Kafka
    KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
    EQUIPMENT_EVENTS_TOPIC = os.getenv("KAFKA_EQUIPMENT_EVENTS_TOPIC", "equipment-events")
    CONSUMER_GROUP = os.getenv("SINK_CONSUMER_GROUP", "db-sink-group")

    # PostgreSQL / TimescaleDB
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "timescaledb")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "eagle_vision")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "eagle")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "eagle_vision_2024")

    # Batching
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
    FLUSH_INTERVAL_SECONDS = float(os.getenv("FLUSH_INTERVAL_SECONDS", "2.0"))

    @classmethod
    def get_dsn(cls) -> str:
        return (
            f"host={cls.POSTGRES_HOST} port={cls.POSTGRES_PORT} "
            f"dbname={cls.POSTGRES_DB} user={cls.POSTGRES_USER} "
            f"password={cls.POSTGRES_PASSWORD}"
        )
