#!/usr/bin/env python3
# TODO: Add manual merge
import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

CSI = "\033["
RESET = f"{CSI}0m"
COLOR_EVEN = f"{CSI}36m"
COLOR_ODD = f"{CSI}33m"

DB_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

parser = argparse.ArgumentParser(description="Manage Bong's long-term memory")
parser.add_argument("-w", "--what", help="Search memories by query")
parser.add_argument("-k", type=int, default=10, help="Number of search results (default: 10)")
parser.add_argument("-a", "--add", nargs="?", const="", help="Add a memory. Provide text or leave blank to be prompted")
parser.add_argument("-l", "--list", action="store_true", help="List all memories")
parser.add_argument("-d", "--delete", nargs="?", const="", help="Delete memories. Optionally provide a search query to filter, or leave blank to list all")
parser.add_argument("-e", "--edit", nargs="?", const="", help="Edit a memory. Optionally provide a search query to filter, or leave blank to list all")
parser.add_argument("-u", "--user", type=int, help="Filter by user ID")
parser.add_argument("-m", "--migrate", nargs="*", help="Batch metadata migration. 0 args: interactive. 1 arg KEY=VALUE: add/set metadata on all memories. 2 args OLD_KEY=VAL NEW_KEY=VAL: replace metadata on matching memories.")
args = parser.parse_args()

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

if args.add is not None:
    text = args.add
    if not text:
        text = input("Enter memory to add: ").strip()
        if not text:
            print("Cancelled.")
            exit()
    meta = {}
    if args.user:
        meta["user_id"] = args.user
    db.add_texts(
        texts=[text],
        metadatas=[meta] if meta else None,
    )
    print(f"Added: {text}")

elif args.delete is not None:
    total = collection.count()
    if total == 0:
        print("No memories saved yet.")
        exit()

    if args.delete:
        results = db.similarity_search_with_relevance_scores(args.delete, k=10)
        indexed = []
        for i, (doc, score) in enumerate(results, 1):
            uid = doc.metadata.get('user_id')
            c = COLOR_EVEN if i % 2 == 0 else COLOR_ODD
            print(f"  {c}{i}. [{score:.2f}] {doc.page_content}{RESET}")
            print(f"  {c}   user_id: {format_user(uid)}{RESET}")
            indexed.append((doc, score))
    else:
        all_data = collection.get(include=["documents", "metadatas"])
        if not all_data["ids"]:
            print("No memories saved yet.")
            exit()
        indexed = []
        for i, (doc_id, text, meta) in enumerate(zip(all_data["ids"], all_data["documents"], all_data["metadatas"]), 1):
            class _Doc:
                pass
            doc = _Doc()
            doc.id = doc_id
            doc.page_content = text
            doc.metadata = meta
            uid = meta.get('user_id')
            c = COLOR_EVEN if i % 2 == 0 else COLOR_ODD
            print(f"  {c}{i}. {doc.page_content}{RESET}")
            print(f"  {c}   user_id: {format_user(uid)}{RESET}")
            indexed.append((doc, None))

    if not indexed:
        print("No results found.")
        exit()

    print()
    choice = input("Enter index numbers to delete (comma-separated), or press Enter to cancel: ").strip()
    if not choice:
        print("Cancelled.")
        exit()

    to_delete = []
    for num in choice.split(","):
        num = num.strip()
        if not num.isdigit():
            print(f"  Skipping invalid index: {num}")
            continue
        idx = int(num) - 1
        if 0 <= idx < len(indexed):
            doc, score = indexed[idx]
            to_delete.append(doc.id if hasattr(doc, 'id') else doc.metadata.get("id"))
            print(f"  Marked for deletion: {doc.page_content[:60]}...")
        else:
            print(f"  Index {num} out of range, skipping.")

    if not to_delete:
        print("Nothing to delete.")
        exit()

    confirm = input(f"\nDelete {len(to_delete)} memory(s)? [y/N] ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        exit()

    collection.delete(ids=to_delete)
    print(f"Deleted {len(to_delete)} memory(s).")

elif args.what:
    total = collection.count()
    print(f"Total memories stored: {total}\n")

    if total == 0:
        print("No memories saved yet.")
        exit()

    results = db.similarity_search_with_relevance_scores(args.what, k=args.k)

    if not results:
        print("No results found.")
    else:
        for i, (doc, score) in enumerate(results, 1):
            c = COLOR_EVEN if i % 2 == 0 else COLOR_ODD
            print(f"{c}[{score:.2f}] {doc.page_content}{RESET}")
            if doc.metadata:
                uid = doc.metadata.get('user_id')
                print(f"{c}    user_id: {format_user(uid)}{RESET}")
            print()

elif args.edit is not None:
    total = collection.count()
    if total == 0:
        print("No memories saved yet.")
        exit()

    if args.edit:
        results = db.similarity_search_with_relevance_scores(args.edit, k=10)
        indexed = []
        for i, (doc, score) in enumerate(results, 1):
            uid = doc.metadata.get('user_id')
            c = COLOR_EVEN if i % 2 == 0 else COLOR_ODD
            print(f"  {c}{i}. [{score:.2f}] {doc.page_content}{RESET}")
            print(f"  {c}   user_id: {format_user(uid)}{RESET}")
            indexed.append((doc, score))
    else:
        all_data = collection.get(include=["documents", "metadatas"])
        if not all_data["ids"]:
            print("No memories saved yet.")
            exit()
        indexed = []
        for i, (doc_id, text, meta) in enumerate(zip(all_data["ids"], all_data["documents"], all_data["metadatas"]), 1):
            class _Doc:
                pass
            doc = _Doc()
            doc.id = doc_id
            doc.page_content = text
            doc.metadata = meta
            uid = meta.get('user_id')
            c = COLOR_EVEN if i % 2 == 0 else COLOR_ODD
            print(f"  {c}{i}. {doc.page_content}{RESET}")
            print(f"  {c}   user_id: {format_user(uid)}{RESET}")
            indexed.append((doc, None))

    if not indexed:
        print("No results found.")
        exit()

    print()
    choice = input("Enter the index number of the memory to edit, or press Enter to cancel: ").strip()
    if not choice or not choice.isdigit():
        print("Cancelled.")
        exit()

    idx = int(choice) - 1
    if idx < 0 or idx >= len(indexed):
        print(f"Index {choice} out of range.")
        exit()

    doc, score = indexed[idx]
    doc_id = doc.id if hasattr(doc, 'id') else doc.metadata.get("id")
    print(f"\nEditing: {doc.page_content}\n")
    uid = doc.metadata.get('user_id')
    print(f"Current user_id: {format_user(uid)}")
    new_uid = input("New user_id (or Enter to keep current): ").strip()
    new_text = input("New text (or Enter to keep current): ").strip()
    if not new_uid and not new_text:
        print("Cancelled.")
        exit()

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
    total = collection.count()
    print(f"Total memories stored: {total}\n")

    if total == 0:
        print("No memories saved yet.")
        exit()

    all_data = collection.get(include=["documents", "metadatas"])
    for i, (text, meta) in enumerate(zip(all_data["documents"], all_data["metadatas"]), 1):
        uid = meta.get('user_id')
        c = COLOR_EVEN if i % 2 == 0 else COLOR_ODD
        print(f"  {c}{i}. {text}{RESET}")
        print(f"  {c}   user_id: {format_user(uid)}{RESET}")

elif args.migrate is not None:
    total = collection.count()
    if total == 0:
        print("No memories stored yet.")
        exit()

    all_data = collection.get(include=["documents", "metadatas"])
    if not all_data["ids"]:
        print("No memories stored yet.")
        exit()

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
            exit()

        matching = [
            (doc_id, meta)
            for doc_id, meta in zip(all_data["ids"], all_data["metadatas"])
            if meta.get(old_k) == old_v
        ]

        if not matching:
            print(f"No memories with {old_k}={old_v} found.")
            exit()

        print(f"Found {len(matching)} memories with {old_k}={old_v}.")
        confirm = input(f"Replace: remove {old_k}={old_v}, set {new_k}={new_v}? [y/N] ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            exit()

        for doc_id, meta in matching:
            new_meta = {k: v for k, v in meta.items() if k != old_k}
            new_meta[new_k] = new_v
            collection.update(ids=[doc_id], metadatas=[new_meta])

        print(f"Updated {len(matching)} memory(s): {old_k}={old_v} -> {new_k}={new_v}")

    elif len(args.migrate) == 1:
        key, val = parse_kv(args.migrate[0])
        if key is None:
            exit()

        confirm = input(f"Set {key}={val} on ALL {total} memories? [y/N] ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            exit()

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
            exit()

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