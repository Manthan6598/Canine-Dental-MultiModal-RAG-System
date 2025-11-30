import uuid
from pinecone import Pinecone, ServerlessSpec

from ingestion.embedding import Embedder
from config.settings import settings

pc = Pinecone(api_key=settings.PINECONE_API_KEY)
embedder = Embedder()

def create_index_if_missing():
    if settings.PINECONE_INDEX not in pc.list_indexes().names():
        pc.create_index(
            name=settings.PINECONE_INDEX,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

def push_to_pinecone(records):
    index = pc.Index(settings.PINECONE_INDEX)
    index.upsert(records)

def build_record(raw, summary, type_, extra=None):
    return {
        "id": str(uuid.uuid4()),
        "values": embedder.embed(summary),
        "metadata": {
            "type": type_,
            "raw": raw,
            "summary": summary,
            **(extra or {})
        }
    }
