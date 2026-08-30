import json
from pathlib import Path

import faiss
import numpy as np

from src.domains import get_domain
from src.embeddings import create_embedding


def load_faiss_index(domain_key: str) -> faiss.Index:
    """
    Load the persisted FAISS index for one domain.
    """
    index_dir: Path = get_domain(domain_key)["index_dir"]
    path = index_dir / "faiss.index"

    if not path.exists():
        raise FileNotFoundError(
            f"FAISS index not found for domain '{domain_key}': {path}. "
            f"Run 'python -m src.vector_store' first."
        )

    return faiss.read_index(str(path))


def load_chunks(domain_key: str) -> list[dict]:
    """
    Load persisted FAQ chunks for one domain.
    """
    index_dir: Path = get_domain(domain_key)["index_dir"]
    path = index_dir / "chunks.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found for domain '{domain_key}': {path}. "
            f"Run 'python -m src.vector_store' first."
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def search_chunks(question: str, domain_key: str, k: int = 3) -> list[dict]:
    """
    Search a domain's FAISS index and return the most relevant chunks.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    index = load_faiss_index(domain_key)
    chunks = load_chunks(domain_key)

    k = min(k, index.ntotal)

    query_embedding = create_embedding(question)

    query_vector = np.array([query_embedding], dtype="float32")

    # The stored vectors were normalized when the index was built.
    # The query must also be normalized for cosine similarity.
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, k)

    results = []

    for score, index_position in zip(scores[0], indices[0]):
        if index_position == -1:
            continue

        chunk = chunks[index_position]

        results.append(
            {
                "chunk_id": chunk["id"],
                "section": chunk["section"],
                "question": chunk["question"],
                "text": chunk["text"],
                "similarity": float(score),
            }
        )

    return results
