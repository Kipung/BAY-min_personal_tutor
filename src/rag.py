"""
rag.py — Firestore RAG loader for BAY-min tutoring sessions.

Fetches lesson modules from Firestore and builds a formatted context
block to inject into the Gemini Live system instruction at session start.

Also provides a simple keyword-based retriever for dynamic context lookup
(useful for future dynamic injection during a session).

Firestore schema assumed:
  modules/{module_id}
    instructional_content:
      concepts[]:          {id, term, definition, example}
      example_walkthrough[]: {id, title, steps[], answer, has_visual, citation_page}
    quiz_questions:
      guided[]:            {id, type, prompt, answer, correct_answer, options[]}
      independent[]:       same
      word_problems[]:     same + difficulty
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MODULES_COLLECTION = "modules"


@dataclass
class Chunk:
    """A single retrievable piece of lesson content."""
    chunk_id: str
    module_id: str
    chunk_type: str   # "concept" | "example" | "question"
    text: str         # full text used for retrieval and context injection


class FirestoreRAG:
    """
    Loads lesson modules from Firestore and provides:

      load()                  — fetch modules and build the chunk index
      build_system_context()  — returns a formatted string for the system instruction
      retrieve(query, top_k)  — keyword-based retrieval of the most relevant chunks
    """

    def __init__(self, db, module_pattern: str | None = None) -> None:
        """
        Args:
            db:              firebase_admin.firestore.client() instance
            module_pattern:  regex matched against document IDs to select modules;
                             None loads every document in the modules collection.
                             Example: r"^math_grade4_ch1_les\d+_"
        """
        self._db = db
        self._module_pattern = re.compile(module_pattern) if module_pattern else None
        self._chunks: list[Chunk] = []
        self._modules: list[dict] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Fetch modules from Firestore and build the in-memory chunk index."""
        col = self._db.collection(MODULES_COLLECTION)
        raw_docs = list(col.stream())

        for doc in raw_docs:
            if self._module_pattern and not self._module_pattern.search(doc.id):
                continue
            if not doc.exists:
                print(f"[RAG] Warning: document '{doc.id}' not found — skipping.")
                continue
            data = doc.to_dict()
            self._modules.append(data)
            self._index_module(data)

        print(
            f"[RAG] Loaded {len(self._modules)} module(s), "
            f"{len(self._chunks)} chunks total."
        )

    def build_system_context(self) -> str:
        """
        Returns a formatted block of all lesson content suitable for use
        as part of the Gemini Live system instruction.
        """
        if not self._modules:
            return ""

        sections: list[str] = []
        for data in self._modules:
            sections.append(self._format_module(data))

        header = (
            "=== TUTORING LESSON CONTENT ===\n"
            "Use the following lesson material to teach and quiz the student. "
            "Follow the session_mode when provided (e.g. 'teach_then_quiz': "
            "explain concepts first, then ask practice questions).\n"
        )
        return header + "\n\n".join(sections)

    def retrieve(self, query: str, top_k: int = 3) -> list[Chunk]:
        """
        Simple keyword-overlap retrieval.  Returns up to top_k chunks
        most relevant to the query, ranked by token overlap.
        """
        tokens = set(re.findall(r"\w+", query.lower()))
        if not tokens:
            return []

        scored: list[tuple[int, Chunk]] = []
        for chunk in self._chunks:
            chunk_tokens = set(re.findall(r"\w+", chunk.text.lower()))
            score = len(tokens & chunk_tokens)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _index_module(self, data: dict) -> None:
        """Parse a module document and add its chunks to the index."""
        mid = data.get("module_id", data.get("lesson_id", "unknown"))
        ic = data.get("instructional_content", {})

        # --- Concepts ---
        for c in ic.get("concepts", []):
            text = (
                f"Concept — {c.get('term', '')}:\n"
                f"  Definition: {c.get('definition', '').strip()}\n"
                f"  Example: {c.get('example', '')}"
            )
            self._chunks.append(
                Chunk(f"{mid}_{c.get('id', 'cx')}", mid, "concept", text)
            )

        # --- Worked examples ---
        for we in ic.get("example_walkthrough", []):
            steps = "\n".join(f"  {s}" for s in we.get("steps", []))
            page = we.get("citation_page", "")
            text = (
                f"Worked Example — {we.get('title', '')} (page {page}):\n"
                f"{steps}\n"
                f"  Answer: {we.get('answer', '')}"
            )
            self._chunks.append(
                Chunk(f"{mid}_{we.get('id', 'we')}", mid, "example", text)
            )

        # --- Quiz questions ---
        qq = data.get("quiz_questions", {})
        for category in ("guided", "independent", "word_problems"):
            for q in qq.get(category, []):
                opts = ", ".join(q.get("options", []))
                difficulty = (
                    f"  Difficulty: {q['difficulty']}\n" if "difficulty" in q else ""
                )
                page = q.get("citation_page", "")
                text = (
                    f"Practice Question ({category}, page {page}) — {q.get('prompt', '')}\n"
                    f"  Options: {opts}\n"
                    f"{difficulty}"
                    f"  Correct Answer: {q.get('correct_answer', '')}\n"
                    f"  Full Explanation: {q.get('answer', '')}"
                )
                self._chunks.append(
                    Chunk(f"{mid}_{q.get('id', 'q')}", mid, "question", text)
                )

    def _format_module(self, data: dict) -> str:
        """Format one module as a human-readable context block."""
        mid = data.get("module_id", "unknown")
        title = data.get("title", mid)
        grade = data.get("grade_level", "")
        desc = data.get("description", "")
        session_mode = data.get("session_mode", "")

        chapter = data.get("chapter", "")
        lesson = data.get("lesson", "")
        citation = data.get("citation", {})
        textbook = citation.get("textbook", "")
        pages = citation.get("pages", [])
        pages_str = ", ".join(str(p) for p in pages) if pages else ""

        lines: list[str] = [
            f"--- Lesson: {title} (Grade {grade}, Chapter {chapter}, Lesson {lesson}) ---",
        ]
        if textbook:
            lines.append(f"Textbook: {textbook}")
        if pages_str:
            lines.append(f"Pages: {pages_str}")
        if desc:
            lines.append(f"Overview: {desc}")
        lines.append("")

        # Concepts
        concept_chunks = [
            c for c in self._chunks
            if c.module_id == mid and c.chunk_type == "concept"
        ]
        if concept_chunks:
            lines.append("** Key Concepts **")
            for c in concept_chunks:
                lines.append(c.text)
            lines.append("")

        # Worked examples
        example_chunks = [
            c for c in self._chunks
            if c.module_id == mid and c.chunk_type == "example"
        ]
        if example_chunks:
            lines.append("** Worked Examples **")
            for c in example_chunks:
                lines.append(c.text)
            lines.append("")

        # Questions
        question_chunks = [
            c for c in self._chunks
            if c.module_id == mid and c.chunk_type == "question"
        ]
        if question_chunks:
            lines.append("** Practice Questions (use these to quiz the student) **")
            for c in question_chunks:
                lines.append(c.text)
            lines.append("")

        return "\n".join(lines)


if __name__ == "__main__":
    import firebase_admin
    from firebase_admin import credentials, firestore

    fb_creds = credentials.Certificate("credentials.json")
    firebase_admin.initialize_app(fb_creds)
    db = firestore.client()

    rag = FirestoreRAG(db, module_pattern=r"^math_grade4_ch1_les\d+_")
    rag.load()

    print("\n" + "=" * 60)
    print("SYSTEM CONTEXT PREVIEW")
    print("=" * 60)
    print(rag.build_system_context())

    print("\n" + "=" * 60)
    print("RETRIEVAL TEST")
    print("=" * 60)
    test_queries = ["addition algorithm", "estimate sum", "Commutative Property"]
    for q in test_queries:
        print(f'\nQuery: "{q}"')
        for chunk in rag.retrieve(q, top_k=2):
            print(f"  [{chunk.chunk_type}] {chunk.chunk_id}")
            print(f"    {chunk.text[:120].strip()}...")
