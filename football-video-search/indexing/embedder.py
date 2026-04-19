from typing import List

from openai import AzureOpenAI

from config.settings import settings
from indexing.event_schema import FootballEvent

_client = AzureOpenAI(
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_api_key,
    api_version=settings.azure_openai_api_version,
)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch-embed a list of strings. Returns list of float vectors."""
    batch_size = settings.embedding_batch_size
    all_vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = _client.embeddings.create(
            model=settings.azure_embedding_deployment,
            input=batch,
        )
        all_vectors.extend([item.embedding for item in response.data])

    return all_vectors


def embed_events(events: List[FootballEvent]) -> None:
    """Embed event_description for each event; stores vector in event.embedding_vector."""
    texts = [e.event_description for e in events]
    if not texts:
        return
    vectors = embed_texts(texts)
    for event, vec in zip(events, vectors):
        event.embedding_vector = vec


def embed_query(query: str) -> List[float]:
    response = _client.embeddings.create(
        model=settings.azure_embedding_deployment,
        input=[query],
    )
    return response.data[0].embedding
