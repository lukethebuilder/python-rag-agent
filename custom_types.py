import pydantic

class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list        # chunk → chunks (main.py uses chunks)
    source_id: str = None

class RAGUpsertResult(pydantic.BaseModel):
    ingested: int       # = int → : int (was assignment not type annotation)

class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]

class RAGQuery(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int