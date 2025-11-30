from pinecone import Pinecone
from config.settings import settings
from ingestion.embedding import Embedder

embedder = Embedder()
pc = Pinecone(api_key=settings.PINECONE_API_KEY)

class Retriever:
    def __init__(self):
        self.index = pc.Index(settings.PINECONE_INDEX)

    def search(self, query, top_k=5):
        q_emb = embedder.embed(query)
        return self.index.query(
            vector=q_emb,
            top_k=top_k,
            include_metadata=True
        )
