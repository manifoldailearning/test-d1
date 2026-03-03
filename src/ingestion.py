from typing import Dict, List


def load_documents() -> List[Dict]:
    """
    Load raw documents from some source.

    For this reference implementation we keep everything in-memory with three
    simple pseudo-documents across different "types".
    """
    docs = [
        {
            "id": "doc_pdf_q4_report",
            "source_type": "pdf",
            "content": (
                "Q4 2024 revenue report. Our Q4 2024 revenue was $4.2M, up 23% from Q3. "
                "This report describes quarterly financial performance."
            ),
            "metadata": {"title": "Q4 2024 Revenue Report"},
        },
        {
            "id": "doc_docx_pto_policy",
            "source_type": "docx",
            "content": (
                "PTO policy document. Employees receive 20 days of paid time off (PTO) per year. "
                "Managers may approve additional leave in special cases."
            ),
            "metadata": {"title": "PTO Policy"},
        },
        {
            "id": "doc_html_support_hours",
            "source_type": "html",
            "content": (
                "Customer support hours. Standard support is available Monday to Friday 9am–5pm. "
                "Premium customers receive 24/7 support including weekends and holidays."
            ),
            "metadata": {"title": "Support Hours"},
        },
    ]
    return docs


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """
    Convert full documents into semantic chunks with overlap and metadata.

    This reference version uses a very simple sentence-based splitting and
    adds minimal metadata required by the assignment and grader.
    """
    chunks: List[Dict] = []
    for doc in documents:
        doc_id = doc["id"]
        source_type = doc["source_type"]
        text = doc["content"]

        # Very simple "sentence" splitting.
        raw_sentences = [s.strip() for s in text.split(".") if s.strip()]

        # Build overlapping chunks of up to 2 sentences.
        position = 0
        for i in range(0, len(raw_sentences), 2):
            sentence_window = raw_sentences[i : i + 2]
            if not sentence_window:
                continue
            chunk_text = ". ".join(sentence_window)
            if not chunk_text.endswith("."):
                chunk_text += "."

            chunk = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{position}",
                "text": chunk_text,
                "metadata": {
                    "source_type": source_type,
                    "position": position,
                    # default min_role is employee; see access_control.filter_chunks_by_access
                    "min_role": "employee",
                },
            }
            chunks.append(chunk)
            position += 1

    return chunks

