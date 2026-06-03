#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BONG_DATA = PROJECT_ROOT / "bong_data"

CSI = "\033["
RESET = f"{CSI}0m"
COLOR_EVEN = f"{CSI}36m"
COLOR_ODD = f"{CSI}33m"

DB_DIR = BONG_DATA / "chroma_db"

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

db = Chroma(
    collection_name="bong_memories",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    persist_directory=str(DB_DIR),
)

collection = db._collection


def format_user(uid):
    if uid is not None:
        return str(uid)
    return "unknown"


def _get_indexed_memories(user_id: int | None = None) -> list:
    all_data = collection.get(include=["documents", "metadatas"])
    if not all_data["ids"]:
        return []

    indexed = []
    for doc_id, text, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
        if user_id is not None and meta.get("user_id") != user_id:
            continue
        class _Doc:
            pass
        doc = _Doc()
        doc.id = doc_id
        doc.page_content = text
        doc.metadata = meta
        indexed.append(doc)
    return indexed


def search_memories(query: str, k: int = 10) -> str:
    """Search memories by query. Returns a formatted string of results."""
    total = collection.count()
    if total == 0:
        return "No memories stored yet."

    results = db.similarity_search_with_relevance_scores(query, k=k)
    if not results:
        return "No results found."

    lines = []
    for i, (doc, score) in enumerate(results, 1):
        uid = doc.metadata.get('user_id')
        lines.append(f"{i}. [{score:.2f}] {doc.page_content}")
        if uid is not None:
            lines.append(f"   user_id: {format_user(uid)}")
    return "\n".join(lines)


def list_memories() -> str:
    """List all memories. Returns a formatted string."""
    total = collection.count()
    if total == 0:
        return "No memories stored yet."

    all_data = collection.get(include=["documents", "metadatas"])
    lines = [f"Total memories: {total}\n"]
    for i, (text, meta) in enumerate(zip(all_data["documents"], all_data["metadatas"]), 1):
        uid = meta.get('user_id')
        lines.append(f"  {i}. {text}")
        lines.append(f"     user_id: {format_user(uid)}")
    return "\n".join(lines)


def add_memory(text: str, user_id: int | None = None) -> str:
    """Add a memory. Returns a confirmation string."""
    meta = {}
    if user_id is not None:
        meta["user_id"] = user_id
    db.add_texts(
        texts=[text],
        metadatas=[meta] if meta else None,
    )
    return f"Added: {text}"


def delete_memory_by_query(query: str, user_id: int | None = None) -> str:
    """Delete memories by index or matching a query. If user_id is given, only delete that user's memories."""
    if query.isdigit():
        idx = int(query) - 1
        indexed = _get_indexed_memories(user_id)
        if not indexed:
            return "No memories stored yet."
        if idx < 0 or idx >= len(indexed):
            return f"Index {query} out of range (1-{len(indexed)})."
        doc = indexed[idx]
        doc_id = doc.id
        uid = doc.metadata.get('user_id')
        collection.delete(ids=[doc_id])
        return f"Deleted memory at index {query}:\n  {doc.page_content}\n  user_id: {format_user(uid)}"

    total = collection.count()
    if total == 0:
        return "No memories stored yet."

    if user_id is not None:
        results = db.similarity_search_with_relevance_scores(query, k=10, filter={"user_id": user_id})
    else:
        results = db.similarity_search_with_relevance_scores(query, k=10)

    if not results:
        return "No results found."

    to_delete = []
    lines = []
    for i, (doc, score) in enumerate(results, 1):
        uid = doc.metadata.get('user_id')
        lines.append(f"  {i}. [{score:.2f}] {doc.page_content}")
        lines.append(f"     user_id: {format_user(uid)}")
        to_delete.append(doc.id if hasattr(doc, 'id') else doc.metadata.get("id"))

    if not to_delete:
        return "No matching memories to delete."

    for doc_id in to_delete:
        if doc_id:
            collection.delete(ids=[doc_id])

    return f"Deleted {len(to_delete)} memory(ies):\n" + "\n".join(lines)


def forget_user(user_id: int) -> str:
    """Delete all memories belonging to a user."""
    all_data = collection.get(include=["documents", "metadatas"])
    if not all_data["ids"]:
        return "No memories stored."

    matching_ids = []
    for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
        if meta.get("user_id") == user_id:
            matching_ids.append(doc_id)

    if not matching_ids:
        return f"No memories found for user {user_id}."

    collection.delete(ids=matching_ids)
    return f"Deleted {len(matching_ids)} memory(ies) for user {user_id}."


def edit_memory(query: str, new_text: str = "", new_user_id: int | None = None) -> str:
    """Search for a memory by query and edit it. If new_text is empty, only updates metadata."""
    total = collection.count()
    if total == 0:
        return "No memories stored yet."

    results = db.similarity_search_with_relevance_scores(query, k=10)
    if not results:
        return "No results found."

    doc, score = results[0]
    doc_id = doc.id if hasattr(doc, 'id') else doc.metadata.get("id")

    orig_meta = doc.metadata.copy() if doc.metadata else {}
    if new_user_id is not None:
        orig_meta["user_id"] = new_user_id

    if new_text:
        collection.delete(ids=[doc_id])
        db.add_texts(
            texts=[new_text],
            metadatas=[orig_meta],
        )
        return f"Updated memory: {new_text} (user_id: {format_user(orig_meta.get('user_id'))})"
    else:
        collection.update(ids=[doc_id], metadatas=[orig_meta])
        return f"Updated metadata: user_id={format_user(orig_meta.get('user_id'))}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage Bong's long-term memory")
    parser.add_argument("-w", "--what", help="Search memories by query")
    parser.add_argument("-k", type=int, default=10, help="Number of search results (default: 10)")
    parser.add_argument("-a", "--add", nargs="?", const="", help="Add a memory. Provide text or leave blank to be prompted")
    parser.add_argument("-l", "--list", action="store_true", help="List all memories")
    parser.add_argument("-d", "--delete", nargs="?", const="", help="Delete memories. Provide an index number or search query; leave blank to list and pick interactively")
    parser.add_argument("-e", "--edit", nargs="?", const="", help="Edit a memory. Optionally provide a search query to filter, or leave blank to list all")
    parser.add_argument("-u", "--user", type=int, help="Filter by user ID")
    parser.add_argument("-m", "--migrate", nargs="*", help="Batch metadata migration. 0 args: interactive. 1 arg KEY=VALUE: add/set metadata on all memories. 2 args OLD_KEY=VAL NEW_KEY=VAL: replace metadata on matching memories.")
    args = parser.parse_args()

    if args.add is not None:
        text = args.add
        if not text:
            text = input("Enter memory to add: ").strip()
            if not text:
                print("Cancelled.")
                sys.exit()
        print(add_memory(text, user_id=args.user))

    elif args.delete is not None:
        if args.delete:
            if args.delete.isdigit():
                indexed = _get_indexed_memories(user_id=args.user)
                idx = int(args.delete) - 1
                if not indexed:
                    print("No memories stored yet.")
                    sys.exit()
                if idx < 0 or idx >= len(indexed):
                    print(f"Index {args.delete} out of range (1-{len(indexed)}).")
                    sys.exit()
                doc = indexed[idx]
                uid = doc.metadata.get('user_id')
                print(f"  Will delete: {doc.page_content}")
                print(f"  user_id: {format_user(uid)}")
                confirm = input("Confirm deletion? [y/N] ").strip().lower()
                if confirm != "y":
                    print("Cancelled.")
                    sys.exit()
                print(delete_memory_by_query(args.delete, user_id=args.user))
            else:
                print(delete_memory_by_query(args.delete, user_id=args.user))
        else:
            indexed = _get_indexed_memories(user_id=args.user)
            if not indexed:
                print("No memories stored yet.")
                sys.exit()
            for i, doc in enumerate(indexed, 1):
                uid = doc.metadata.get('user_id')
                c = COLOR_EVEN if i % 2 == 0 else COLOR_ODD
                print(f"  {c}{i}. {doc.page_content}{RESET}")
                print(f"  {c}   user_id: {format_user(uid)}{RESET}")
            print()
            choice = input("Enter index or search query to delete (or Enter to cancel): ").strip()
            if not choice:
                print("Cancelled.")
                sys.exit()
            if choice.isdigit():
                idx = int(choice) - 1
                if idx < 0 or idx >= len(indexed):
                    print(f"Index {choice} out of range (1-{len(indexed)}).")
                    sys.exit()
                doc = indexed[idx]
                uid = doc.metadata.get('user_id')
                print(f"  Will delete: {doc.page_content}")
                print(f"  user_id: {format_user(uid)}")
                confirm = input("Confirm deletion? [y/N] ").strip().lower()
                if confirm != "y":
                    print("Cancelled.")
                    sys.exit()
                collection.delete(ids=[doc.id])
                print(f"Deleted memory at index {choice}.")
            else:
                print(delete_memory_by_query(choice, user_id=args.user))

    elif args.what:
        print(search_memories(args.what, k=args.k))

    elif args.edit is not None:
        if args.edit and args.edit.isdigit():
            indexed = _get_indexed_memories(user_id=args.user)
            if not indexed:
                print("No memories saved yet.")
                sys.exit()
            idx = int(args.edit) - 1
            if idx < 0 or idx >= len(indexed):
                print(f"Index {args.edit} out of range (1-{len(indexed)}).")
                sys.exit()
            for i, doc in enumerate(indexed, 1):
                uid = doc.metadata.get('user_id')
                c = COLOR_EVEN if i % 2 == 0 else COLOR_ODD
                marker = ">>>" if i - 1 == idx else "   "
                print(f"  {c}{marker} {i}. {doc.page_content}{RESET}")
                print(f"  {c}       user_id: {format_user(uid)}{RESET}")
            doc = indexed[idx]
            doc_id = doc.id
        elif args.edit:
            results = db.similarity_search_with_relevance_scores(args.edit, k=10)
            indexed = []
            for i, (doc, score) in enumerate(results, 1):
                uid = doc.metadata.get('user_id')
                c = COLOR_EVEN if i % 2 == 0 else COLOR_ODD
                print(f"  {c}{i}. [{score:.2f}] {doc.page_content}{RESET}")
                print(f"  {c}   user_id: {format_user(uid)}{RESET}")
                indexed.append((doc, score))

            if not indexed:
                print("No results found.")
                sys.exit()

            print()
            choice = input("Enter the index number of the memory to edit, or press Enter to cancel: ").strip()
            if not choice or not choice.isdigit():
                print("Cancelled.")
                sys.exit()

            idx = int(choice) - 1
            if idx < 0 or idx >= len(indexed):
                print(f"Index {choice} out of range.")
                sys.exit()

            doc, score = indexed[idx]
            doc_id = doc.id if hasattr(doc, 'id') else doc.metadata.get("id")
        else:
            indexed = _get_indexed_memories(user_id=args.user)
            if not indexed:
                print("No memories saved yet.")
                sys.exit()
            for i, doc in enumerate(indexed, 1):
                uid = doc.metadata.get('user_id')
                c = COLOR_EVEN if i % 2 == 0 else COLOR_ODD
                print(f"  {c}{i}. {doc.page_content}{RESET}")
                print(f"  {c}   user_id: {format_user(uid)}{RESET}")

            if not indexed:
                print("No results found.")
                sys.exit()

            print()
            choice = input("Enter the index number of the memory to edit, or press Enter to cancel: ").strip()
            if not choice or not choice.isdigit():
                print("Cancelled.")
                sys.exit()

            idx = int(choice) - 1
            if idx < 0 or idx >= len(indexed):
                print(f"Index {choice} out of range.")
                sys.exit()

            doc = indexed[idx]
            doc_id = doc.id

        print(f"\nEditing: {doc.page_content}\n")
        uid = doc.metadata.get('user_id')
        print(f"Current user_id: {format_user(uid)}")
        new_uid = input("New user_id (or Enter to keep current): ").strip()
        new_text = input("New text (or Enter to keep current): ").strip()
        if not new_uid and not new_text:
            print("Cancelled.")
            sys.exit()

        orig_meta = doc.metadata.copy() if doc.metadata else {}
        if new_uid:
            orig_meta["user_id"] = int(new_uid)
        if "saved_at" in orig_meta:
            del orig_meta["saved_at"]

        if new_text:
            collection.delete(ids=[doc_id])
            db.add_texts(
                texts=[new_text],
                metadatas=[orig_meta],
            )
            print(f"Updated memory: {new_text} (user_id: {orig_meta.get('user_id', 'unknown')})")
        else:
            collection.update(ids=[doc_id], metadatas=[orig_meta])
            print(f"Updated metadata: user_id={orig_meta.get('user_id', 'unknown')}")

    elif args.list:
        print(list_memories())

    elif args.migrate is not None:
        total = collection.count()
        if total == 0:
            print("No memories stored yet.")
            sys.exit()

        all_data = collection.get(include=["documents", "metadatas"])
        if not all_data["ids"]:
            print("No memories stored yet.")
            sys.exit()

        def parse_kv(s):
            if "=" not in s:
                print(f"  Invalid format '{s}', expected KEY=VALUE")
                return None, None
            k, v = s.split("=", 1)
            try:
                v = int(v)
            except ValueError:
                pass
            return k.strip(), v

        if len(args.migrate) == 2:
            old_k, old_v = parse_kv(args.migrate[0])
            new_k, new_v = parse_kv(args.migrate[1])
            if old_k is None or new_k is None:
                sys.exit()

            matching = [
                (doc_id, meta)
                for doc_id, meta in zip(all_data["ids"], all_data["metadatas"])
                if meta.get(old_k) == old_v
            ]

            if not matching:
                print(f"No memories with {old_k}={old_v} found.")
                sys.exit()

            print(f"Found {len(matching)} memories with {old_k}={old_v}.")
            confirm = input(f"Replace: remove {old_k}={old_v}, set {new_k}={new_v}? [y/N] ").strip().lower()
            if confirm != "y":
                print("Cancelled.")
                sys.exit()

            for doc_id, meta in matching:
                new_meta = {k: v for k, v in meta.items() if k != old_k}
                new_meta[new_k] = new_v
                collection.update(ids=[doc_id], metadatas=[new_meta])

            print(f"Updated {len(matching)} memory(s): {old_k}={old_v} -> {new_k}={new_v}")

        elif len(args.migrate) == 1:
            key, val = parse_kv(args.migrate[0])
            if key is None:
                sys.exit()

            confirm = input(f"Set {key}={val} on ALL {total} memories? [y/N] ").strip().lower()
            if confirm != "y":
                print("Cancelled.")
                sys.exit()

            for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
                new_meta = dict(meta)
                new_meta[key] = val
                collection.update(ids=[doc_id], metadatas=[new_meta])

            print(f"Updated {total} memory(s): set {key}={val}")

        else:
            needs_update = [
                (doc_id, text, meta)
                for doc_id, text, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"])
                if "saved_at" in meta or "user_id" not in meta
            ]

            if not needs_update:
                print("No memories need metadata updates. Use KEY=VALUE mode to add/replace metadata.")
                sys.exit()

            print(f"Found {len(needs_update)} memories needing updates:\n")

            updated = 0
            for i, (doc_id, text, meta) in enumerate(needs_update, 1):
                c = COLOR_EVEN if i % 2 == 0 else COLOR_ODD
                print(f"  {c}{i}. {text[:80]}{RESET}")
                meta_str = ", ".join(f"{k}={v}" for k, v in meta.items())
                print(f"  {c}   metadata: {meta_str}{RESET}")

                changes = input(f"  Set metadata (KEY=VALUE pairs, space-separated, or Enter to skip): ").strip()
                if not changes:
                    print("  Skipped.\n")
                    continue

                new_meta = {k: v for k, v in meta.items() if k != "saved_at"}
                for token in changes.split():
                    k, v = parse_kv(token)
                    if k is not None:
                        new_meta[k] = v

                collection.update(ids=[doc_id], metadatas=[new_meta])
                updated += 1
                meta_str = ", ".join(f"{k}={v}" for k, v in new_meta.items())
                print(f"  Updated: {meta_str}\n")

            print(f"Migrated {updated} memory(s).")

    else:
        parser.print_help()