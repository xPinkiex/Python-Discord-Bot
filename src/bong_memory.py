import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.tools import tool
import debug
import bong_tools
import bong_memory_helpers


@tool
def save_memory(fact: str) -> str:
    """Save an important fact to long-term memory. Use this to remember things about users, preferences, inside jokes, or any information worth recalling later. Be selective — only save things that are genuinely useful to remember. If the fact contradicts or updates something already remembered, the old memory will be replaced automatically.
    Args:
        fact: A concise fact or piece of information to remember (e.g. "Eve loves dubstep and skrillex", "Radon is an orange fox who likes cars").
    """
    try:
        clean_fact = bong_memory_helpers._clean_for_embedding(fact)

        contradiction_id = bong_memory_helpers._find_contradiction(clean_fact, bong_tools.current_user_id)
        if contradiction_id:
            collection = bong_memory_helpers._vector_db._collection
            try:
                old = collection.get(ids=[contradiction_id], include=["documents", "metadatas"])
                old_text = old["documents"][0] if old["documents"] else "(unknown)"
                old_meta = dict(old["metadatas"][0]) if old["metadatas"] else {}
                collection.delete(ids=[contradiction_id])
                old_meta["saved_at"] = datetime.now().timestamp()
                if bong_tools.current_username:
                    old_meta["username"] = bong_tools.current_username
                bong_memory_helpers._vector_db.add_texts([clean_fact], metadatas=[old_meta])
                return f"Updated memory: {old_text} → {clean_fact}"
            except Exception as e:
                bong_memory_helpers._vector_db.add_texts(
                    [clean_fact],
                    metadatas=[{"user_id": bong_tools.current_user_id, "saved_at": datetime.now().timestamp(), "username": bong_tools.current_username or ""}],
                )
                return f"Remembered: {clean_fact} (failed to replace old: {e})"

        similar = bong_memory_helpers._vector_db.similarity_search_with_relevance_scores(clean_fact, k=3)
        for doc, score in similar:
            if score >= 0.7:
                existing_uid = doc.metadata.get("user_id")
                if existing_uid == bong_tools.current_user_id:
                    return f"Already remembered something similar: {doc.page_content}"

        bong_memory_helpers._vector_db.add_texts(
            [clean_fact],
            metadatas=[{"user_id": bong_tools.current_user_id, "saved_at": datetime.now().timestamp(), "username": bong_tools.current_username or ""}],
        )
        return f"Remembered: {clean_fact}"
    except Exception as e:
        return f"Failed to save memory: {e}"


@tool
def recall_memories_by_userid(query: str) -> str:
    """Search the current user's long-term memories. Use this when you need to recall something you've previously saved about the user you're talking to.
    Args:
        query: What to search for (e.g. "music preferences", "inside jokes about cars").
    """
    results = bong_memory_helpers.retrieve_memories(query, user_id=bong_tools.current_user_id)
    if not results:
        return "No relevant memories found for this user."
    return results


@tool
def recall_memories_general(query: str) -> str:
    """Search all long-term memories regardless of user. Use this when you need to recall something about someone other than the current user, or a general fact not tied to a specific person.
    Args:
        query: What to search for (e.g. "Radon's fursona", "inside jokes", "who likes dubstep").
    """
    results = bong_memory_helpers.retrieve_memories(query)
    if not results:
        return "No relevant memories found."
    return results


@tool
def forget_memory(query: str) -> str:
    """Delete a long-term memory that is no longer accurate or wanted. Use this when someone tells you to forget something, or when you realize a saved memory is wrong. Searches for the most similar memory and deletes it. Can only delete memories belonging to the current user.
    Args:
        query: A description of the memory to forget (e.g. "Eve likes dubstep", "Radon's favorite color").
    """
    try:
        clean_query = bong_memory_helpers._clean_for_embedding(query)
        results = bong_memory_helpers._vector_db.similarity_search_with_relevance_scores(clean_query, k=3, filter={"user_id": bong_tools.current_user_id})
        for doc, score in results:
            if score >= 0.5:
                doc_id = doc.id if hasattr(doc, 'id') else doc.metadata.get("id")
                if doc_id:
                    collection = bong_memory_helpers._vector_db._collection
                    collection.delete(ids=[doc_id])
                    return f"Forgetted: {doc.page_content}"
        return "No similar memory found to forget. Try describing it differently."
    except Exception as e:
        return f"Failed to forget memory: {e}"


tools = [save_memory, recall_memories_by_userid, recall_memories_general, forget_memory]