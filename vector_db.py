from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from config import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_DIMENSION

class QdrantStorage:
    def __init__(self, url=QDRANT_URL, collection=QDRANT_COLLECTION, dim=EMBEDDING_DIMENSION):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def list_sources(self) -> list[str]:
        sources: set[str] = set()
        offset = None
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection,
                with_payload=True,
                with_vectors=False,
                limit=100,
                offset=offset,
            )
            for r in results:
                src = (getattr(r, 'payload', None) or {}).get('source', '')
                if src:
                    sources.add(src)
            if offset is None:
                break
        return sorted(sources)

    def search(self, query_vector, top_k=5, source_filter: str | None = None):
        query_filter = None
        if source_filter:
            query_filter = Filter(must=[
                FieldCondition(key="source", match=MatchValue(value=source_filter))
            ])
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
            query_filter=query_filter,
        ).points
        contexts = []
        sources = []

        for r in results:
            payload = getattr(r, 'payload', None) or {}
            text = payload.get('text', '')
            source = payload.get('source', '')
            if text:
                contexts.append(text)
                sources.append(source)

        return {"contexts": contexts, "sources": list(set(sources))}
