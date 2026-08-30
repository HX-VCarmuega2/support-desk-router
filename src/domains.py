"""Central registry of the support domains: where each domain's source
document lives and where its built vector index is stored.

Every other module (chunking, embeddings, vector store, retriever, agents)
looks up paths through this registry instead of hardcoding them, so adding
a fourth domain later means adding one entry here.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DOMAINS = {
    "hr": {
        "name": "HR",
        "document": DATA_DIR / "hr_docs" / "hr_faq.md",
        "index_dir": DATA_DIR / "hr_docs" / "index",
    },
    "tech": {
        "name": "Tech/IT",
        "document": DATA_DIR / "tech_docs" / "tech_faq.md",
        "index_dir": DATA_DIR / "tech_docs" / "index",
    },
    "finance": {
        "name": "Finance",
        "document": DATA_DIR / "finance_docs" / "finance_faq.md",
        "index_dir": DATA_DIR / "finance_docs" / "index",
    },
}


def get_domain(domain_key: str) -> dict:
    if domain_key not in DOMAINS:
        raise ValueError(
            f"Unknown domain '{domain_key}'. "
            f"Valid domains: {', '.join(DOMAINS)}"
        )

    return DOMAINS[domain_key]
