import openai
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery
from services.config import Settings

RAG_COLLECTION = "FeedbackLoopDocuments"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 3
RAG_PROPERTIES = [
    Property(name="chunk_id", data_type=DataType.TEXT),
    Property(name="document_id", data_type=DataType.TEXT),
    Property(name="source_type", data_type=DataType.TEXT),
    Property(name="url", data_type=DataType.TEXT),
    Property(name="title", data_type=DataType.TEXT),
    Property(name="chunk_index", data_type=DataType.INT),
    Property(name="content", data_type=DataType.TEXT),
    Property(name="created_at", data_type=DataType.DATE),]

class RagServiceError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code

def weaviate_client(settings: Settings):
    return weaviate.connect_to_local(
        host=settings.weaviate_host,
        port=settings.weaviate_port,
        grpc_port=settings.weaviate_grpc_port,
    )

def weaviate_status(settings: Settings) -> dict:
    client = weaviate_client(settings)
    try:
        return {"ready": client.is_ready(), "live": client.is_live()}
    finally:
        client.close()



def embedding(text: str, settings: Settings) -> list[float]:
    client = openai.OpenAI(base_url=settings.embedding_base_url, api_key="EMPTY", timeout=settings.embedding_timeout)
    try:
        response = client.embeddings.create(model=settings.embedding_model, input=text)
        vector = response.data[0].embedding
        if not isinstance(vector, list) or not vector:
            raise RagServiceError("Embedding 服務沒有回傳向量。", 503)
        return vector
    except Exception as error:
        raise RagServiceError("Embedding 服務目前無法使用。", 503) from error


def rag_collection(client):
    if not client.collections.exists(RAG_COLLECTION):
        client.collections.create(
            RAG_COLLECTION,
            vector_config=Configure.Vectors.self_provided(),
            properties=RAG_PROPERTIES,
        )
    collection = client.collections.get(RAG_COLLECTION)
    existing = {property.name for property in collection.config.get().properties}
    for property in RAG_PROPERTIES:
        if property.name not in existing:
            collection.config.add_property(property)
    return collection

def delete_document(document_id: str, settings: Settings) -> None:
    client = weaviate_client(settings)
    if client.collections.exists(RAG_COLLECTION):
        collection = client.collections.get(RAG_COLLECTION)
        collection.data.delete_many(Filter.by_property("document_id").equal(document_id))
    client.close()