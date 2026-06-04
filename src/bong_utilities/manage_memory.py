#!/usr/bin/env python3
"""CLI tool for managing Bong's long-term memory (ChromaDB).

Usage:
    python -m bong_utilities.manage_memory -l            List all memories
    python -m bong_utilities.manage_memory -l -u Eve     List Eve's memories
    python -m bong_utilities.manage_memory -s "dubstep"  Search memories
    python -m bong_utilities.manage_memory -a             Add a memory (interactive)
    python -m bong_utilities.manage_memory -e 3           Edit memory #3
    python -m bong_utilities.manage_memory -d 3           Delete memory #3
    python -m bong_utilities.manage_memory --forget-user Eve  Delete all of Eve's memories
    python -m bong_utilities.manage_memory --expire       Remove expired memories
    python -m bong_utilities.manage_memory -d 3 --dry-run  Preview without changes
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bong_memory_helpers
import user_data

CSI = "\033["
RESET = f"{CSI}0m"
BOLD = f"{CSI}1m"
DIM = f"{CSI}2m"
RED = f"{CSI}31m"
GREEN = f"{CSI}32m"
YELLOW = f"{CSI}33m"
CYAN = f"{CSI}36m"
BOLD_RED = f"{CSI}1;31m"
BOLD_GREEN = f"{CSI}1;32m"
BOLD_YELLOW = f"{CSI}1;33m"
BOLD_CYAN = f"{CSI}1;36m"

VALID_CATEGORIES = ("preference", "fact", "relationship", "inside_joke", "instruction")

CATEGORY_ICONS = {
    "preference": "\u2605",
    "instruction": "!",
    "relationship": "\u2665",
    "fact": "\u2022",
    "inside_joke": "\u263a",
}


def _format_timestamp(ts) -> str:
    if not ts:
        return DIM + "never" + RESET
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError, OSError):
        return DIM + "unknown" + RESET


def _format_importance(imp) -> str:
    try:
        imp = int(imp)
    except (ValueError, TypeError):
        imp = 3
    stars = "\u2605" * imp + "\u2606" * (5 - imp)
    return f"{YELLOW}{stars}{RESET} ({imp})"


def _format_category(cat) -> str:
    cat = str(cat) if cat else "fact"
    icon = CATEGORY_ICONS.get(cat, "\u2022")
    return f"{icon} {cat}"


def _get_collection():
    """Return the ChromaDB collection. Extracted for easy testing."""
    return bong_memory_helpers._vector_db._collection


def _get_all_memories(user_id: int | None = None) -> list[tuple]:
    """Fetch all memories, optionally filtered by user_id.
    Returns list of (doc_id, text, metadata) tuples with 1-based indexing ready.
    """
    collection = _get_collection()
    result = collection.get(include=["documents", "metadatas"])
    if not result["ids"]:
        return []

    items = []
    for doc_id, text, meta in zip(result["ids"], result["documents"], result["metadatas"]):
        if user_id is not None and meta.get("user_id") != user_id:
            continue
        items.append((doc_id, text, meta))
    return items


def _resolve_user(arg: str) -> tuple[int | None, str | None]:
    """Resolve a user argument (display name or numeric ID) to (user_id, display_name).
    Returns (None, None) if not found.
    """
    if arg.strip().isdigit():
        uid = int(arg.strip())
        user_data.load_users()
        if uid in user_data._user_data:
            name = user_data._user_data[uid].get("display_name", str(uid))
        else:
            name = str(uid)
        return uid, name

    resolved_id, warning = bong_memory_helpers.resolve_name_to_id(arg)
    if resolved_id is not None:
        user_data.load_users()
        if resolved_id in user_data._user_data:
            name = user_data._user_data[resolved_id].get("display_name", arg)
        else:
            name = arg
        if warning:
            print(f"{YELLOW}Warning: {warning}{RESET}")
            confirm = input(f"{BOLD}Use this match anyway? [y/N]{RESET} ").strip().lower()
            if confirm != "y":
                return None, None
        return resolved_id, name
    return None, None


def cmd_list(args):
    user_id = None
    display_name = None
    if args.user:
        user_id, display_name = _resolve_user(args.user)
        if user_id is None:
            print(f"{RED}User not found: {args.user}{RESET}")
            return

    items = _get_all_memories(user_id)
    if not items:
        if user_id:
            print(f"No memories found for {BOLD}{display_name}{RESET}.")
        else:
            print("No memories stored yet.")
        return

    header = f"{BOLD}Total: {len(items)} memories"
    if user_id:
        header += f" (filtered to {display_name})"
    header += RESET
    print(header)
    print()

    for i, (doc_id, text, meta) in enumerate(items, 1):
        c = CYAN if i % 2 == 0 else YELLOW
        user_name = meta.get("user_name", DIM + "unknown" + RESET)
        category = _format_category(meta.get("category"))
        importance = _format_importance(meta.get("importance"))
        saved = _format_timestamp(meta.get("saved_at"))
        accessed = _format_timestamp(meta.get("last_accessed"))
        count = meta.get("access_count", 0)
        try:
            count = int(count)
        except (ValueError, TypeError):
            count = 0

        print(f"  {c}{BOLD}{i:>3}.{RESET} {c}{text}{RESET}")
        print(f"  {c}     user: {user_name}  |  {category}  |  {importance}{RESET}")
        print(f"  {c}     saved: {saved}  |  accessed: {accessed}  |  used {count}x{RESET}")
        print()


def cmd_search(args):
    user_id = None
    display_name = None
    if args.user:
        user_id, display_name = _resolve_user(args.user)
        if user_id is None:
            print(f"{RED}User not found: {args.user}{RESET}")
            return

    k = args.k or 10
    query = bong_memory_helpers._clean_for_embedding(args.search)

    if user_id is not None:
        results = bong_memory_helpers._vector_db.similarity_search_with_relevance_scores(
            query, k=k, filter={"user_id": user_id}
        )
    else:
        results = bong_memory_helpers._vector_db.similarity_search_with_relevance_scores(
            query, k=k
        )

    if not results:
        print("No results found.")
        return

    header = f"{BOLD}Results for \"{args.search}\""
    if user_id:
        header += f" (user: {display_name})"
    header += RESET
    print(header)
    print()

    for i, (doc, score) in enumerate(results, 1):
        c = CYAN if i % 2 == 0 else YELLOW
        meta = doc.metadata
        user_name = meta.get("user_name", DIM + "unknown" + RESET)
        category = _format_category(meta.get("category"))
        importance = _format_importance(meta.get("importance"))
        saved = _format_timestamp(meta.get("saved_at"))

        print(f"  {c}{BOLD}{i:>3}.{RESET} {c}[{score:.2f}] {doc.page_content}{RESET}")
        print(f"  {c}     user: {user_name}  |  {category}  |  {importance}  |  saved: {saved}{RESET}")
        print()


def cmd_add(args):
    user_id = None
    display_name = None
    if args.user:
        user_id, display_name = _resolve_user(args.user)
        if user_id is None:
            print(f"{RED}User not found: {args.user}{RESET}")
            return

    text = args.add
    if not text:
        text = input(f"{BOLD}Enter memory text:{RESET} ").strip()
        if not text:
            print("Cancelled.")
            return

    print(f"\n  Category options: {', '.join(VALID_CATEGORIES)}")
    category = input(f"{BOLD}Category [{DIM}fact{RESET}{BOLD}]:{RESET} ").strip().lower()
    if not category or category not in VALID_CATEGORIES:
        category = "fact"

    print(f"  Importance: 1=trivial, 2=low, 3=normal, 4=high, 5=critical")
    imp_str = input(f"{BOLD}Importance [{DIM}3{RESET}{BOLD}]:{RESET} ").strip()
    try:
        importance = max(1, min(5, int(imp_str)))
    except (ValueError, TypeError):
        importance = 3

    if not user_id:
        user_str = input(f"{BOLD}User (display name or ID, Enter for none):{RESET} ").strip()
        if user_str:
            user_id, display_name = _resolve_user(user_str)

    clean_text = bong_memory_helpers._clean_for_embedding(text)
    if not clean_text:
        print(f"{RED}Text was empty after cleaning.{RESET}")
        return

    meta = {
        "category": category,
        "importance": importance,
        "saved_at": datetime.now().timestamp(),
        "last_accessed": datetime.now().timestamp(),
        "access_count": 0,
    }
    if user_id is not None:
        meta["user_id"] = user_id
        meta["user_name"] = display_name or str(user_id)

    print()
    print(f"  {BOLD}Preview:{RESET}")
    print(f"  {CYAN}{clean_text}{RESET}")
    print(f"  category: {category}  |  importance: {importance}  |  user: {display_name or 'none'}")

    confirm = input(f"\n{BOLD}Add this memory? [y/N]:{RESET} ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    bong_memory_helpers._vector_db.add_texts([clean_text], metadatas=[meta])
    print(f"{GREEN}Added: {clean_text}{RESET}")


def cmd_edit(args):
    user_id = None
    display_name = None
    if args.user:
        user_id, display_name = _resolve_user(args.user)
        if user_id is None:
            print(f"{RED}User not found: {args.user}{RESET}")
            return

    items = _get_all_memories(user_id)

    target = args.edit
    doc_id = None
    meta = None
    text = None

    if target.isdigit():
        idx = int(target) - 1
        if idx < 0 or idx >= len(items):
            if not items:
                print("No memories stored yet.")
            else:
                print(f"{RED}Index {target} out of range (1-{len(items)}).{RESET}")
            return
        doc_id, text, meta = items[idx]
    else:
        query = bong_memory_helpers._clean_for_embedding(target)
        if user_id is not None:
            results = bong_memory_helpers._vector_db.similarity_search_with_relevance_scores(
                query, k=5, filter={"user_id": user_id}
            )
        else:
            results = bong_memory_helpers._vector_db.similarity_search_with_relevance_scores(
                query, k=5
            )
        if not results:
            print("No results found.")
            return

        print(f"{BOLD}Matching memories:{RESET}")
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. [{score:.2f}] {doc.page_content}")
            m = doc.metadata
            print(f"     user: {m.get('user_name', '?')}  |  {_format_category(m.get('category'))}  |  {_format_importance(m.get('importance'))}")

        choice = input(f"\n{BOLD}Enter index to edit (or Enter to cancel):{RESET} ").strip()
        if not choice or not choice.isdigit():
            print("Cancelled.")
            return
        idx = int(choice) - 1
        if idx < 0 or idx >= len(results):
            print(f"{RED}Invalid index.{RESET}")
            return
        doc, _ = results[idx]
        doc_id = doc.id if hasattr(doc, 'id') else doc.metadata.get("id")
        text = doc.page_content
        meta = dict(doc.metadata)

    print(f"\n  {BOLD}Editing memory:{RESET}")
    print(f"  {CYAN}{text}{RESET}")
    print(f"  user: {meta.get('user_name', 'none')}  |  {_format_category(meta.get('category'))}  |  {_format_importance(meta.get('importance'))}")
    print(f"  saved: {_format_timestamp(meta.get('saved_at'))}  |  accessed: {_format_timestamp(meta.get('last_accessed'))}  |  used {meta.get('access_count', 0)}x")
    print()

    changes = {}
    while True:
        print(f"  {BOLD}Edit which field?{RESET}")
        print(f"    1. Text")
        print(f"    2. Category (current: {meta.get('category', 'fact')})")
        print(f"    3. Importance (current: {meta.get('importance', 3)})")
        print(f"    4. User (current: {meta.get('user_name', 'none')})")
        print(f"    5. {GREEN}Apply changes{RESET}")
        print(f"    6. Cancel")

        field = input(f"\n  {BOLD}Choose [1-6]:{RESET} ").strip()

        if field == "1":
            new_text = input(f"  {BOLD}New text [{DIM}keep current{RESET}{BOLD}]:{RESET} ").strip()
            if new_text:
                changes["text"] = new_text
        elif field == "2":
            print(f"  Options: {', '.join(VALID_CATEGORIES)}")
            new_cat = input(f"  {BOLD}New category [{DIM}{meta.get('category', 'fact')}{RESET}{BOLD}]:{RESET} ").strip().lower()
            if new_cat and new_cat in VALID_CATEGORIES:
                changes["category"] = new_cat
            elif new_cat:
                print(f"  {RED}Invalid category.{RESET}")
        elif field == "3":
            print(f"  1=trivial, 2=low, 3=normal, 4=high, 5=critical")
            imp_str = input(f"  {BOLD}New importance [{DIM}{meta.get('importance', 3)}{RESET}{BOLD}]:{RESET} ").strip()
            try:
                new_imp = int(imp_str)
                if 1 <= new_imp <= 5:
                    changes["importance"] = new_imp
                else:
                    print(f"  {RED}Must be 1-5.{RESET}")
            except ValueError:
                if imp_str:
                    print(f"  {RED}Invalid number.{RESET}")
        elif field == "4":
            new_user_str = input(f"  {BOLD}New user (display name or ID, or 'none' to clear) [{DIM}{meta.get('user_name', 'none')}{RESET}{BOLD}]:{RESET} ").strip()
            if new_user_str.lower() == "none":
                changes["user_id"] = None
                changes["user_name"] = ""
            elif new_user_str:
                new_uid, new_uname = _resolve_user(new_user_str)
                if new_uid is not None:
                    changes["user_id"] = new_uid
                    changes["user_name"] = new_uname
                else:
                    print(f"  {RED}User not found.{RESET}")
        elif field == "5":
            break
        elif field == "6":
            print("Cancelled.")
            return
        else:
            print(f"  {RED}Invalid choice.{RESET}")

    if not changes:
        print("No changes made. Cancelled.")
        return

    print(f"\n  {BOLD}Summary of changes:{RESET}")
    new_text = changes.get("text", text)
    new_meta = dict(meta)
    for key, val in changes.items():
        if key == "text":
            continue
        elif key == "user_id":
            if val is None:
                new_meta.pop("user_id", None)
                new_meta["user_name"] = ""
            else:
                new_meta["user_id"] = val
        elif key == "user_name":
            new_meta["user_name"] = val
        else:
            new_meta[key] = val

    if "text" in changes:
        old_clean = bong_memory_helpers._clean_for_embedding(text)
        new_clean = bong_memory_helpers._clean_for_embedding(changes["text"])
        print(f"  {RED}- {old_clean}{RESET}")
        print(f"  {GREEN}+ {new_clean}{RESET}")
    if "category" in changes:
        print(f"  category: {meta.get('category', 'fact')} -> {changes['category']}")
    if "importance" in changes:
        print(f"  importance: {meta.get('importance', 3)} -> {changes['importance']}")
    if "user_id" in changes:
        old_user = meta.get("user_name", "none")
        new_user = changes.get("user_name", str(changes.get("user_id", "")))
        print(f"  user: {old_user} -> {new_user}")

    confirm = input(f"\n  {BOLD}Apply changes? [y/N]:{RESET} ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    collection = _get_collection()

    if "text" in changes:
        clean_new = bong_memory_helpers._clean_for_embedding(changes["text"])
        collection.delete(ids=[doc_id])
        bong_memory_helpers._vector_db.add_texts([clean_new], metadatas=[new_meta])
        print(f"{GREEN}Updated memory: {clean_new}{RESET}")
    else:
        collection.update(ids=[doc_id], metadatas=[new_meta])
        print(f"{GREEN}Updated metadata.{RESET}")


def cmd_delete(args):
    user_id = None
    display_name = None
    if args.user:
        user_id, display_name = _resolve_user(args.user)
        if user_id is None:
            print(f"{RED}User not found: {args.user}{RESET}")
            return

    items = _get_all_memories(user_id)
    target = args.delete
    to_delete = []

    if target.isdigit():
        idx = int(target) - 1
        if idx < 0 or idx >= len(items):
            if not items:
                print("No memories stored yet.")
            else:
                print(f"{RED}Index {target} out of range (1-{len(items)}).{RESET}")
            return
        doc_id, text, meta = items[idx]
        to_delete.append((doc_id, text, meta))
    else:
        query = bong_memory_helpers._clean_for_embedding(target)
        if user_id is not None:
            results = bong_memory_helpers._vector_db.similarity_search_with_relevance_scores(
                query, k=10, filter={"user_id": user_id}
            )
        else:
            results = bong_memory_helpers._vector_db.similarity_search_with_relevance_scores(
                query, k=10
            )

        if not results:
            print("No results found.")
            return

        print(f"{BOLD}Matching memories:{RESET}")
        print()
        shown = 0
        for doc, score in results:
            if score < bong_memory_helpers.MIN_RELEVANCE:
                continue
            shown += 1
            m = doc.metadata
            uid = doc.id if hasattr(doc, 'id') else m.get("id")
            c = CYAN if shown % 2 == 0 else YELLOW
            print(f"  {c}{shown:>3}. [{score:.2f}] {doc.page_content}{RESET}")
            print(f"  {c}     user: {m.get('user_name', '?')}  |  {_format_category(m.get('category'))}  |  {_format_importance(m.get('importance'))}{RESET}")
            to_delete.append((uid, doc.page_content, m))

        if not to_delete:
            print("No results above relevance threshold.")
            return

        if len(to_delete) > 1:
            print(f"\n  {BOLD_YELLOW}{len(to_delete)} memories matched.{RESET}")
            choice = input(f"  {BOLD}Enter index to delete, 'all' to delete all, or Enter to cancel:{RESET} ").strip()
            if choice.lower() == "all":
                pass
            elif choice.isdigit():
                idx = int(choice) - 1
                if idx < 0 or idx >= len(to_delete):
                    print(f"{RED}Invalid index.{RESET}")
                    return
                to_delete = [to_delete[idx]]
            else:
                print("Cancelled.")
                return

    if not to_delete:
        print("Nothing to delete.")
        return

    print(f"\n  {BOLD_RED}The following {len(to_delete)} memory(ies) will be DELETED:{RESET}")
    for i, (_, text, meta) in enumerate(to_delete, 1):
        c = RED
        print(f"  {c}{i}. {text}{RESET}")
        print(f"  {c}   user: {meta.get('user_name', '?')}  |  {_format_category(meta.get('category'))}  |  {_format_importance(meta.get('importance'))}{RESET}")

    if args.dry_run:
        print(f"\n  {YELLOW}Dry run — no changes made.{RESET}")
        return

    confirm = input(f"\n  {BOLD_RED}Type 'yes' to confirm deletion:{RESET} ").strip()
    if confirm != "yes":
        print("Cancelled.")
        return

    collection = _get_collection()
    doc_ids = [d[0] for d in to_delete if d[0]]
    if doc_ids:
        collection.delete(ids=doc_ids)
    print(f"{GREEN}Deleted {len(doc_ids)} memory(ies).{RESET}")


def cmd_forget_user(args):
    name_or_id = args.forget_user
    user_id, display_name = _resolve_user(name_or_id)

    if user_id is None:
        print(f"{RED}User not found: {name_or_id}{RESET}")
        return

    items = _get_all_memories(user_id)

    if not items:
        print(f"No memories found for {BOLD}{display_name}{RESET}.")
        return

    print(f"\n  {BOLD}Found {len(items)} memories for {display_name} (user_id: {user_id}):{RESET}")
    print()
    for i, (_, text, meta) in enumerate(items[:10], 1):
        c = CYAN if i % 2 == 0 else YELLOW
        print(f"  {c}{i}. {text}{RESET}")
        print(f"  {c}   {_format_category(meta.get('category'))}  |  {_format_importance(meta.get('importance'))}{RESET}")
    if len(items) > 10:
        print(f"  {DIM}... and {len(items) - 10} more{RESET}")
    print()

    if args.dry_run:
        print(f"  {YELLOW}Dry run — would delete {len(items)} memories for {display_name}.{RESET}")
        return

    print(f"  {BOLD_RED}ALL {len(items)} memories for {display_name} will be permanently deleted.{RESET}")
    confirm_name = input(f"  {BOLD_RED}Type '{display_name}' to confirm:{RESET} ").strip()
    if confirm_name != display_name:
        print("Cancelled.")
        return

    collection = _get_collection()
    doc_ids = [d[0] for d in items if d[0]]
    collection.delete(ids=doc_ids)
    print(f"{GREEN}Deleted {len(doc_ids)} memories for {display_name}.{RESET}")


def cmd_expire(args):
    collection = _get_collection()
    result = collection.get(include=["metadatas"])
    if not result["ids"]:
        print("No memories stored.")
        return

    expired = []
    now = datetime.now().timestamp()
    for doc_id, meta in zip(result["ids"], result["metadatas"]):
        saved_at = meta.get("saved_at", 0)
        last_accessed = meta.get("last_accessed", 0)
        importance = meta.get("importance", 3)
        if not isinstance(importance, (int, float)):
            importance = 3
        cutoff = max(saved_at, last_accessed) + (bong_memory_helpers.MEMORY_EXPIRY_BASE_DAYS * int(importance) * 86400)
        if now > cutoff:
            expired.append((doc_id, meta))

    if not expired:
        print(f"{GREEN}No expired memories found.{RESET}")
        return

    print(f"\n  {BOLD_YELLOW}{len(expired)} expired memories:{RESET}")
    print()
    for i, (_, meta) in enumerate(expired[:20], 1):
        c = YELLOW if i % 2 == 0 else RED
        user_name = meta.get("user_name", "?")
        importance = meta.get("importance", 3)
        saved = _format_timestamp(meta.get("saved_at"))
        days_ago = int((now - max(float(meta.get("saved_at", 0)), float(meta.get("last_accessed", 0)))) / 86400)
        print(f"  {c}{i}. user: {user_name}  |  importance: {importance}  |  saved: {saved}  |  {days_ago} days old{RESET}")
    if len(expired) > 20:
        print(f"  {DIM}... and {len(expired) - 20} more{RESET}")

    if args.dry_run:
        print(f"\n  {YELLOW}Dry run — would delete {len(expired)} expired memories.{RESET}")
        return

    print(f"\n  {BOLD_RED}Permanently delete {len(expired)} expired memories?{RESET}")
    confirm = input(f"  {BOLD_RED}Type 'yes' to confirm:{RESET} ").strip()
    if confirm != "yes":
        print("Cancelled.")
        return

    collection.delete(ids=[d[0] for d in expired])
    print(f"{GREEN}Deleted {len(expired)} expired memories.{RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage Bong's long-term memory (ChromaDB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s -l                          List all memories
  %(prog)s -l -u Eve                   List Eve's memories
  %(prog)s -s "dubstep"                Search for memories about dubstep
  %(prog)s -s "likes cars" -u RadonFox  Search RadonFox's memories
  %(prog)s -a                           Add a memory (interactive)
  %(prog)s -e 3                         Edit memory #3
  %(prog)s -e "dubstep"                 Edit by search query
  %(prog)s -d 5                         Delete memory #5
  %(prog)s -d "old fact" --dry-run      Preview deletion
  %(prog)s --forget-user Eve            Delete all of Eve's memories
  %(prog)s --expire                     Remove expired memories""",
    )
    parser.add_argument("-l", "--list", action="store_true", help="List all memories")
    parser.add_argument("-s", "--search", type=str, help="Search memories by query")
    parser.add_argument("-k", type=int, default=10, help="Number of search results (default: 10)")
    parser.add_argument("-a", "--add", nargs="?", const="", help="Add a memory. Provide text or leave blank for interactive prompt")
    parser.add_argument("-e", "--edit", type=str, help="Edit a memory by index or search query")
    parser.add_argument("-d", "--delete", type=str, help="Delete memory(ies) by index or search query")
    parser.add_argument("--forget-user", type=str, metavar="NAME_OR_ID", help="Delete all memories for a user")
    parser.add_argument("--expire", action="store_true", help="Remove memories past their expiration date")
    parser.add_argument("-u", "--user", type=str, help="Filter by user (display name or Discord ID)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without making them (works with -d, --forget-user, --expire)")

    args = parser.parse_args()

    user_data.load_users()

    if args.list:
        cmd_list(args)
    elif args.search:
        cmd_search(args)
    elif args.add is not None:
        cmd_add(args)
    elif args.edit:
        cmd_edit(args)
    elif args.delete:
        cmd_delete(args)
    elif args.forget_user:
        cmd_forget_user(args)
    elif args.expire:
        cmd_expire(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()