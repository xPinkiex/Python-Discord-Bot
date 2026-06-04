#!/usr/bin/env python3
"""One-time migration script for ChromaDB memory schema.

Adds category, importance, user_name, last_accessed, access_count to existing memories.
Cleans text by stripping <@ID> mentions, (userID:...) tags, and collapsing double spaces.
If text is changed by cleaning, the old document is deleted and re-added (re-embedded).

Run once:
    python scripts/migrate_memory.py

If already migrated (category field exists in metadata), the script will skip.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DB_DIR = PROJECT_ROOT / "bong_data" / "chroma_db"

sys.path.insert(0, str(SRC_DIR))

import re

_MENTION_RE = re.compile(r"<@!?\d+>")
_USERID_TAG_RE = re.compile(r"\s*\(userID:?\s*\d+\)", re.IGNORECASE)


def _clean_for_embedding(text: str) -> str:
    text = _MENTION_RE.sub("", text)
    text = _USERID_TAG_RE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def main():
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    print(f"ChromaDB path: {DB_DIR}")
    print("Connecting to ChromaDB...")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(
        collection_name="bong_memories",
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )
    collection = db._collection

    result = collection.get(include=["documents", "metadatas"])
    if not result["ids"]:
        print("No memories found. Nothing to migrate.")
        return

    # Check if already migrated
    if any("category" in meta for meta in result["metadatas"]):
        print("Already migrated (category field found in metadata). Skipping.")
        return

    print(f"Found {len(result['ids'])} memories to migrate.")
    print()

    ids_to_delete = []
    texts_to_add = []
    metas_to_add = []

    updated_ids = []
    updated_metas = []

    for i, (doc_id, text, meta) in enumerate(zip(result["ids"], result["documents"], result["metadatas"])):
        cleaned = _clean_for_embedding(text)

        new_meta = dict(meta)
        new_meta.setdefault("category", "fact")
        new_meta.setdefault("importance", 3)
        new_meta["user_name"] = str(meta.get("username", ""))
        new_meta.setdefault("last_accessed", meta.get("saved_at", 0))
        new_meta.setdefault("access_count", 0)

        text_changed = cleaned != text
        if text_changed:
            print(f"  [{i+1}] Text cleaned: \"{text[:60]}...\" -> \"{cleaned[:60]}...\"")
            ids_to_delete.append(doc_id)
            texts_to_add.append(cleaned)
            metas_to_add.append(new_meta)
        else:
            print(f"  [{i+1}] Metadata only: \"{text[:60]}...\"")
            updated_ids.append(doc_id)
            updated_metas.append(new_meta)

    print()
    print(f"Updating {len(updated_ids)} memories in-place...")
    if updated_ids:
        collection.update(ids=updated_ids, metadatas=updated_metas)

    print(f"Re-embedding {len(ids_to_delete)} memories with cleaned text...")
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        db.add_texts(texts_to_add, metadatas=metas_to_add)

    print()
    print(f"Migration complete: {len(updated_ids)} metadata updates, {len(ids_to_delete)} re-embeddings.")

    # Verify
    result2 = collection.get(include=["metadatas"])
    migrated = sum(1 for m in result2["metadatas"] if "category" in m)
    print(f"Verification: {migrated}/{len(result2['ids'])} memories have 'category' field.")


if __name__ == "__main__":
    main()