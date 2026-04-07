"""
Dashboard - Configuration
"""
import os


class Config:
    # Kafka
    KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
    EQUIPMENT_EVENTS_TOPIC = os.getenv("KAFKA_EQUIPMENT_EVENTS_TOPIC", "equipment-events")
    ANNOTATED_FRAMES_TOPIC = os.getenv("KAFKA_ANNOTATED_FRAMES_TOPIC", "annotated-frames")
    CONSUMER_GROUP = os.getenv("DASHBOARD_CONSUMER_GROUP", "dashboard-group")

    # TimescaleDB
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "timescaledb")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "eagle_vision")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "eagle")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "eagle_vision_2024")

    # Dashboard
    REFRESH_INTERVAL = int(os.getenv("DASHBOARD_REFRESH_INTERVAL", "1"))

    @classmethod
    def get_dsn(cls) -> str:
        return (
            f"host={cls.POSTGRES_HOST} port={cls.POSTGRES_PORT} "
            f"dbname={cls.POSTGRES_DB} user={cls.POSTGRES_USER} "
            f"password={cls.POSTGRES_PASSWORD}"
        )

    @classmethod
    def get_connection_string(cls) -> str:
        return (
            f"postgresql://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}"
            f"@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
        )
