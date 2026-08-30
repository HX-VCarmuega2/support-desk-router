import json
import sys
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAIError

from src.chunking import chunk_by_faq, load_document, print_stats
from src.domains import DOMAINS
from src.embeddings import create_embeddings


def build_faiss_index(embeddings: list[list[float]]) -> faiss.Index:
    """
    Create a FAISS index using cosine similarity.

    Embeddings are normalized first and then stored in an
    IndexFlatIP index. For normalized vectors, inner product
    is equivalent to cosine similarity.
    """
    if not embeddings:
        raise ValueError("Embeddings list cannot be empty.")

    vectors = np.array(embeddings, dtype="float32")

    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    return index


def save_chunks(chunks: list[dict], index_dir: Path) -> None:
    """
    Save chunk metadata and content as JSON.
    """
    index_dir.mkdir(parents=True, exist_ok=True)

    path = index_dir / "chunks.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)


def save_faiss_index(index: faiss.Index, index_dir: Path) -> None:
    """
    Save the FAISS index to disk.
    """
    index_dir.mkdir(parents=True, exist_ok=True)

    path = index_dir / "faiss.index"

    faiss.write_index(index, str(path))


def build_domain_index(domain_key: str) -> dict:
    """
    Build and persist the FAISS index and chunk metadata for one domain.

    Returns a small summary dict for reporting purposes.
    """
    domain = DOMAINS[domain_key]

    document = load_document(domain["document"])
    chunks = chunk_by_faq(document)
    print_stats(domain["name"], chunks)

    embeddings = create_embeddings(chunks)

    index = build_faiss_index(embeddings)

    save_faiss_index(index, domain["index_dir"])
    save_chunks(chunks, domain["index_dir"])

    return {
        "domain": domain_key,
        "chunks": len(chunks),
        "vectors": index.ntotal,
        "dimensions": index.d,
    }


def main():
    print("Building vector stores for all domains...\n")

    summaries = []

    try:
        for domain_key in DOMAINS:
            summaries.append(build_domain_index(domain_key))

    except FileNotFoundError as error:
        print(f"Missing source document: {error}")
        sys.exit(1)

    except ValueError as error:
        print(f"Invalid data: {error}")
        sys.exit(1)

    except OpenAIError as error:
        print(f"OpenAI API error: {error}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)

    for summary in summaries:
        print(
            f"{summary['domain']:<8} "
            f"chunks={summary['chunks']:<4} "
            f"vectors={summary['vectors']:<4} "
            f"dims={summary['dimensions']}"
        )


if __name__ == "__main__":
    main()
