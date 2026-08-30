import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)


def create_client() -> OpenAI:
    """
    Create and return an OpenAI client.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured."
        )

    return OpenAI(api_key=api_key)


def create_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a single text.
    """
    if not text.strip():
        raise ValueError("Text cannot be empty.")

    client = create_client()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response.data[0].embedding


def create_embeddings(
    chunks: list[dict],
) -> list[list[float]]:
    """
    Generate embeddings for a list of FAQ chunks.

    The embedding is generated from each chunk's 'content',
    which contains the FAQ question and answer.
    """
    if not chunks:
        raise ValueError("Chunks list cannot be empty.")

    texts = []

    for chunk in chunks:
        content = chunk.get("content", "").strip()

        if not content:
            raise ValueError(
                f"Chunk {chunk.get('id')} has no content."
            )

        texts.append(content)

    client = create_client()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    embeddings = [
        item.embedding
        for item in sorted(
            response.data,
            key=lambda item: item.index,
        )
    ]

    return embeddings
