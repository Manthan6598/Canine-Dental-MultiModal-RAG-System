import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    PINECONE_API_KEY: str
    PINECONE_INDEX: str = "canine-dental-rag"
    MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.2"  # HF CPU-safe model

    class Config:
        env_file = ".env"

settings = Settings()
