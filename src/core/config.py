from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):

    DATA_PATH: Path = Path("data/customer_purchase_data.csv")

    MODEL_SERVER_URL: str = "http://model-api:8000/model"

    LOG_FILE: Path = Path("logs/server.log")
    LOG_LEVEL: str = "INFO"


settings = Settings()
