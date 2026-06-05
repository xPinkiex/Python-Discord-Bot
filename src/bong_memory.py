import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.tools import tool
import debug
import bong_tools
import bong_memory_helpers
import user_data

VALID_CATEGORIES = {"preference", "fact", "relationship", "inside_joke", "instruction"}


def _check_llm():
    if not user_data.has_permission(bong_tools.current_user_id, "llm"):
        return "You don't have permission to use memory features. Ask an admin to grant you the llm tag."
    return None


@tool
def save_memory(fact: str, category: str, importance: int, about: str = "") -> str:
    """Save an important fact to long-term memory. Use this to remember things about people, preferences, inside jokes, or any information worth recalling later. Be selective — only save things that are genuinely useful to remember. If the fact contradicts or updates something already remembered, the old memory will be replaced automatically.

    Always use display names, never Discord IDs or mentions. Write facts normally — not in third person and never refer to yourself as "Bong" (e.g. write "Eve loves dubstep" not "Bong remembers that Eve loves dubstep", write "I prefer to be called a creecher" not "Bong's preference is to be called a creecher").

    Use the 'about' field to indicate WHO the fact is about. For facts about a specific person, pass their display name. For general facts not about anyone in particular, leave about empty.

    Args:
        fact: A concise fact written normally, using display names only, never <@ID> mentions (e.g. "Eve loves dubstep and skrillex", "the server has a meme channel").
        category: The type of fact: "preference" (likes/dislikes), "fact" (general knowledge), "relationship" (how people relate to each other), "inside_joke" (recurring jokes), "instruction" (things to always remember or do).
        importance: How important this fact is on a scale of 1-5. 1=trivial detail, 3=normal, 5=critical (allergies, core identity, essential instructions).
        about: The display name of WHO this fact is about. Pass a display name for facts about a specific person, or leave empty for general facts not about anyone. Always use display names, never Discord IDs (e.g. "Eve", "RadonFox").
    """
    denied = _check_llm()
    if denied:
        return denied
    try:
        category = category.lower().strip()
        if category not in VALID_CATEGORIES:
            category = "fact"
        importance = max(1, min(5, importance))

        clean_fact = bong_memory_helpers._clean_for_embedding(fact)
        if not clean_fact:
            return "Fact was empty after cleaning."

        # Resolve who the fact is about
        about_name = about.strip() if about else ""
        unresolved_warning = ""
        if about_name:
            resolved_id, warning = bong_memory_helpers.resolve_name_to_id(about_name)
            if resolved_id is not None:
                target_user_id = resolved_id
                target_user_name = about_name
            else:
                target_user_id = None
                target_user_name = ""
                unresolved_warning = f'Couldn\'t find anyone named "{about_name}". Saved as a general fact instead. Ask the user to clarify or provide their exact display name.'
        else:
            target_user_id = None
            target_user_name = ""

        dedup_id, is_near_dup = bong_memory_helpers.tiered_dedup(clean_fact, target_user_id)

        if is_near_dup is False and dedup_id is not None:
            # Near-duplicate — skip saving
            try:
                old = bong_memory_helpers._vector_db._collection.get(ids=[dedup_id], include=["documents"])
                old_text = old["documents"][0] if old["documents"] else "(unknown)"
                return f"Already remembered something very similar: {old_text}"
            except Exception:
                return "Already remembered something very similar."

        if dedup_id is not None and is_near_dup:
            # Contradiction — replace old memory with updated one
            collection = bong_memory_helpers._vector_db._collection
            try:
                old = collection.get(ids=[dedup_id], include=["documents", "metadatas"])
                old_text = old["documents"][0] if old["documents"] else "(unknown)"
                old_meta = dict(old["metadatas"][0]) if old["metadatas"] else {}
                collection.delete(ids=[dedup_id])
                if target_user_id is not None:
                    old_meta["user_id"] = target_user_id
                else:
                    old_meta.pop("user_id", None)
                old_meta["user_name"] = target_user_name
                old_meta["category"] = category
                old_meta["importance"] = importance
                old_meta["saved_at"] = datetime.now().timestamp()
                old_meta["last_accessed"] = datetime.now().timestamp()
                bong_memory_helpers._vector_db.add_texts([clean_fact], metadatas=[old_meta])
                result = f"Updated memory: {old_text} → {clean_fact}"
                if unresolved_warning:
                    result += f" ({unresolved_warning})"
                return result
            except Exception as e:
                fallback_meta = {
                    "user_name": target_user_name,
                    "category": category,
                    "importance": importance,
                    "saved_at": datetime.now().timestamp(),
                    "last_accessed": datetime.now().timestamp(),
                    "access_count": 0,
                }
                if target_user_id is not None:
                    fallback_meta["user_id"] = target_user_id
                bong_memory_helpers._vector_db.add_texts(
                    [clean_fact],
                    metadatas=[fallback_meta],
                )
                result = f"Remembered: {clean_fact} (failed to replace old: {e})"
                if unresolved_warning:
                    result += f" ({unresolved_warning})"
                return result

        # New fact — save it
        new_meta = {
            "user_name": target_user_name,
            "category": category,
            "importance": importance,
            "saved_at": datetime.now().timestamp(),
            "last_accessed": datetime.now().timestamp(),
            "access_count": 0,
        }
        if target_user_id is not None:
            new_meta["user_id"] = target_user_id
        bong_memory_helpers._vector_db.add_texts(
            [clean_fact],
            metadatas=[new_meta],
        )
        result = f"Remembered: {clean_fact}"
        if unresolved_warning:
            result += f" ({unresolved_warning})"
        return result
    except Exception as e:
        return f"Failed to save memory: {e}"


@tool
def recall_memories(query: str, about: str = "") -> str:
    """Search long-term memories. Use this to recall things you've previously saved. Leave 'about' empty for a general search, or pass a display name to search memories about a specific person. If the name is matched fuzzily, you'll see a warning — double-check with the user if unsure. Write queries normally, not in third person.
    Args:
        query: What to search for, written normally — not in third person (e.g. "music preferences", "my favorite color", "who likes cars").
        about: The display name of the person whose memories to search. Leave empty for a general search across all memories. Use display names only, never Discord IDs.
    """
    denied = _check_llm()
    if denied:
        return denied
    about_name = about.strip() if about else ""
    if about_name:
        results = bong_memory_helpers.retrieve_memories(query, user_id=None, about_name=about_name)
    else:
        results = bong_memory_helpers.retrieve_memories(query, user_id=bong_tools.current_user_id, about_name="")
    if not results or not results.strip():
        if about_name:
            return f"No relevant memories found about {about_name}."
        return "No relevant memories found."
    return results


@tool
def forget_memory(query: str) -> str:
    """Delete a long-term memory that is no longer accurate or wanted. Use this when someone tells you to forget something, or when you realize a saved memory is wrong. Searches all memories and deletes the best match. You can only forget memories about yourself or general facts — not other people's memories. Always use display names in the query, never Discord IDs.
    Args:
        query: A description of the memory to forget using display names only (e.g. "Eve likes dubstep", "my favorite color").
    """
    denied = _check_llm()
    if denied:
        return denied
    try:
        clean_query = bong_memory_helpers._clean_for_embedding(query)
        results = bong_memory_helpers._vector_db.similarity_search_with_relevance_scores(
            clean_query, k=5
        )
        best_doc = None
        best_score = 0.0
        for doc, score in results:
            if score > best_score:
                best_doc = doc
                best_score = score

        if best_doc is None or best_score < 0.5:
            return "No similar memory found to forget. Try describing it differently."

        doc_id = best_doc.id if hasattr(best_doc, 'id') else best_doc.metadata.get("id")
        if not doc_id:
            return "Found a matching memory but couldn't identify it for deletion."

        memory_user_id = best_doc.metadata.get("user_id")
        if memory_user_id is not None:
            try:
                memory_user_id = int(memory_user_id)
            except (ValueError, TypeError):
                memory_user_id = None

        current = bong_tools.current_user_id
        is_owner = current is not None and user_data.is_admin(current)

        if memory_user_id is not None and memory_user_id != current and not is_owner:
            return "No similar memory found to forget. Try describing it differently."

        collection = bong_memory_helpers._vector_db._collection
        collection.delete(ids=[doc_id])
        return f"Forgotten: {best_doc.page_content}"
    except Exception as e:
        return f"Failed to forget memory: {e}"


tools = [save_memory, recall_memories, forget_memory]