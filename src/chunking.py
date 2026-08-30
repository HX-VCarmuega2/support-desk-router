import re
from pathlib import Path


def load_document(path: Path) -> str:
    """
    Load a domain's FAQ source document.
    """
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    return path.read_text(encoding="utf-8")


def chunk_by_faq(text: str) -> list[dict]:
    """
    Split an FAQ document into semantic chunks.

    Each FAQ question and answer is kept together and enriched
    with metadata such as FAQ id and section.
    """
    chunks = []

    current_section = None
    current_faq = None

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            if current_faq and current_faq["answer_lines"]:
                current_faq["answer_lines"].append("")
            continue

        # Detect sections:
        # SECTION 1 — ONBOARDING
        section_match = re.match(
            r"SECTION\s+\d+\s+—\s+(.+)",
            line,
        )

        if section_match:
            if current_faq:
                chunks.append(_build_chunk(current_faq))
                current_faq = None

            current_section = section_match.group(1).strip()
            continue

        # Detect FAQs:
        # FAQ 01 — What does a new employee need to complete before their first day?
        faq_match = re.match(
            r"FAQ\s+(\d+)\s+—\s+(.+)",
            line,
        )

        if faq_match:
            if current_faq:
                chunks.append(_build_chunk(current_faq))

            current_faq = {
                "id": int(faq_match.group(1)),
                "section": current_section,
                "question": faq_match.group(2).strip(),
                "answer_lines": [],
            }

            continue

        # Add content only when we are inside a FAQ.
        if current_faq:
            current_faq["answer_lines"].append(line)

    # Save final FAQ.
    if current_faq:
        chunks.append(_build_chunk(current_faq))

    return chunks


def _build_chunk(faq: dict) -> dict:
    """
    Convert the temporary FAQ structure into the final chunk format.
    """
    answer = "\n".join(faq["answer_lines"]).strip()

    content = f"{faq['question']}\n\n{answer}"

    return {
        "id": faq["id"],
        "section": faq["section"],
        "question": faq["question"],
        "text": answer,
        "content": content,
    }


def print_stats(domain_name: str, chunks: list[dict]) -> None:
    """
    Print basic statistics about generated chunks.
    """
    if not chunks:
        print(f"[{domain_name}] No chunks generated.")
        return

    word_counts = [len(chunk["content"].split()) for chunk in chunks]

    print(f"[{domain_name}] {len(chunks)} chunks | "
          f"avg {sum(word_counts) / len(word_counts):.1f} words | "
          f"min {min(word_counts)} | max {max(word_counts)}")
