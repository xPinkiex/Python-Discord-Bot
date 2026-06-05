import difflib
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bong_tools
import debug

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

DB_DIR = bong_tools.BONG_DATA / "chroma_db"
_embeddings = OllamaEmbeddings(model="nomic-embed-text", keep_alive=-1)
_vector_db = Chroma(
    collection_name="bong_memories",
    embedding_function=_embeddings,
    persist_directory=str(DB_DIR),
)

_MENTION_RE = re.compile(r"<@!?\d+>")
_USERID_TAG_RE = re.compile(r"\s*\(userID:?\s*\d+\)", re.IGNORECASE)

CATEGORY_WEIGHTS = {
    "preference": 0.3,
    "instruction": 0.25,
    "relationship": 0.2,
    "fact": 0.1,
    "inside_joke": 0.05,
}

USER_MEMORY_SCORE_BOOST = 0.3
MEMORY_EXPIRY_BASE_DAYS = 30
RECENCY_BOOST = 0.15
RECENCY_HALFLIFE_DAYS = 60.0
CONTRADICTION_THRESHOLD = 0.75
NEAR_DUPLICATE_THRESHOLD = 0.92
MIN_RELEVANCE_USER = 0.3
MIN_RELEVANCE_GENERAL = 0.5
CONTRADICTION_MODEL = "gemma3:12b-cloud"
_contradiction_model = ChatOllama(model=CONTRADICTION_MODEL, temperature=0.0, num_predict=5, keep_alive=-1)


def _clean_for_embedding(text: str) -> str:
    text = _MENTION_RE.sub("", text)
    text = _USERID_TAG_RE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _apply_category_boost(score: float, category: str) -> float:
    return score + CATEGORY_WEIGHTS.get(category, 0.0)


def _apply_importance_boost(score: float, importance: int) -> float:
    return score + 0.05 * importance


def _apply_user_match_boost(score: float, from_user_search: bool) -> float:
    if from_user_search:
        return score + USER_MEMORY_SCORE_BOOST
    return score


def _apply_recency_boost(score: float, saved_at, last_accessed, halflife_days: float = RECENCY_HALFLIFE_DAYS) -> float:
    ts = max(saved_at, last_accessed) if saved_at and last_accessed else (saved_at or last_accessed or 0)
    if not ts:
        return score
    age_days = (datetime.now().timestamp() - ts) / 86400.0
    if age_days < 0:
        age_days = 0
    recency = RECENCY_BOOST * (0.5 ** (age_days / halflife_days))
    return score + recency


def _batch_increment_access_counts(doc_ids: list):
    valid_ids = [did for did in doc_ids if did is not None]
    if not valid_ids:
        return
    try:
        collection = _vector_db._collection
        result = collection.get(ids=valid_ids, include=["metadatas"])
        if not result["metadatas"]:
            return
        now = datetime.now().timestamp()
        updated_ids = []
        updated_metas = []
        for doc_id, meta in zip(result["ids"], result["metadatas"]):
            new_meta = dict(meta)
            raw_count = new_meta.get("access_count", 0)
            new_meta["access_count"] = (int(raw_count) + 1) if isinstance(raw_count, (int, float)) else 1
            new_meta["last_accessed"] = now
            updated_ids.append(doc_id)
            updated_metas.append(new_meta)
        collection.update(ids=updated_ids, metadatas=updated_metas)
    except Exception as e:
        debug.error("Memory", f"Failed to increment access counts: {e}")


def resolve_name_to_id(name: str) -> tuple[int | None, str | None]:
    """Resolve a display name to a Discord user ID.

    Returns (user_id, warning) where:
      - (user_id, None)      -> exact match
      - (user_id, "message") -> fuzzy match, warning to show LLM
      - (None, None)         -> no match found
    """
    import user_data

    name_lower = name.lower().strip()

    # 1. Exact match in users.json display_name
    for uid, entry in user_data._user_data.items():
        dn = entry.get("display_name", "")
        if dn and dn.lower() == name_lower:
            return int(uid), None

    # 2. Exact match in ChromaDB user_name metadata
    try:
        collection = _vector_db._collection
        result = collection.get(include=["metadatas"])
        for meta in result["metadatas"]:
            un = str(meta.get("user_name", ""))
            if un.lower() == name_lower:
                uid = meta.get("user_id")
                if uid is not None:
                    try:
                        return int(uid), None
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass

    # 3. Fuzzy match against all known names
    all_names = []
    name_to_uid: dict[str, int] = {}
    for uid, entry in user_data._user_data.items():
        dn = entry.get("display_name", "")
        if dn:
            all_names.append(dn.lower())
            name_to_uid[dn.lower()] = int(uid)

    try:
        collection = _vector_db._collection
        result = collection.get(include=["metadatas"])
        for meta in result["metadatas"]:
            un = str(meta.get("user_name", ""))
            uid = meta.get("user_id")
            if un and uid is not None:
                key = un.lower()
                if key not in name_to_uid:
                    all_names.append(key)
                    try:
                        name_to_uid[key] = int(uid)
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass

    if all_names:
        matches = difflib.get_close_matches(name_lower, all_names, n=1, cutoff=0.6)
        if matches:
            matched_uid = name_to_uid.get(matches[0])
            if matched_uid is not None:
                return int(matched_uid), f"The name \"{name}\" was matched fuzzily to \"{matches[0]}\" — this may not be the person you meant."

    return None, None


def retrieve_memories(query: str, user_id: int | None, about_name: str = "", k: int = 10) -> str:
    try:
        target_user_id = user_id
        warning = None
        skip_user_search = False

        if about_name:
            resolved_id, warning = resolve_name_to_id(about_name)
            if resolved_id is not None:
                target_user_id = resolved_id
            else:
                # Name not resolved — skip user-scoped search, do general only
                skip_user_search = True

        # If user_id is None and no about_name, this is a general search
        if target_user_id is None:
            skip_user_search = True

        cleaned_query = _clean_for_embedding(query)

        seen_ids = set()
        all_results = []

        # User-scoped search (for memories about a specific person)
        if not skip_user_search and target_user_id is not None:
            user_results = _vector_db.similarity_search_with_relevance_scores(
                cleaned_query, k=k, filter={"user_id": target_user_id}
            )
            for doc, score in user_results:
                if score < MIN_RELEVANCE_USER:
                    continue
                doc_id = doc.id if hasattr(doc, 'id') else doc.metadata.get("id")
                norm = doc.page_content.strip().lower()
                dedup_key = doc_id if doc_id is not None else norm
                if dedup_key in seen_ids:
                    continue
                seen_ids.add(dedup_key)
                adjusted = score
                category = str(doc.metadata.get("category", "fact"))
                importance = doc.metadata.get("importance", 3)
                if not isinstance(importance, (int, float)):
                    importance = 3
                importance = int(importance)
                saved_at = doc.metadata.get("saved_at")
                last_accessed = doc.metadata.get("last_accessed")
                adjusted = _apply_category_boost(adjusted, category)
                adjusted = _apply_importance_boost(adjusted, importance)
                adjusted = _apply_user_match_boost(adjusted, True)
                adjusted = _apply_recency_boost(adjusted, saved_at, last_accessed)
                all_results.append((doc, doc_id, adjusted))

        # General search (fills remaining slots, includes user_id=None general facts)
        general_results = _vector_db.similarity_search_with_relevance_scores(
            cleaned_query, k=k * 3
        )
        for doc, score in general_results:
            doc_user_id = doc.metadata.get("user_id")
            threshold = MIN_RELEVANCE_USER if doc_user_id is None else MIN_RELEVANCE_GENERAL
            if score < threshold:
                continue
            doc_id = doc.id if hasattr(doc, 'id') else doc.metadata.get("id")
            norm = doc.page_content.strip().lower()
            dedup_key = doc_id if doc_id is not None else norm
            if dedup_key in seen_ids:
                continue
            seen_ids.add(dedup_key)
            adjusted = score
            category = str(doc.metadata.get("category", "fact"))
            importance = doc.metadata.get("importance", 3)
            if not isinstance(importance, (int, float)):
                importance = 3
            importance = int(importance)
            saved_at = doc.metadata.get("saved_at")
            last_accessed = doc.metadata.get("last_accessed")
            is_about_target = target_user_id is not None and doc_user_id == target_user_id
            adjusted = _apply_category_boost(adjusted, category)
            adjusted = _apply_importance_boost(adjusted, importance)
            adjusted = _apply_user_match_boost(adjusted, is_about_target)
            adjusted = _apply_recency_boost(adjusted, saved_at, last_accessed)
            all_results.append((doc, doc_id, adjusted))

        if not all_results:
            debug.log("Memory", "No relevant memories found")
            result = ""
            if warning:
                result = f"⚠️ {warning}\n\n"
            return result

        _batch_increment_access_counts([doc_id for _, doc_id, _ in all_results])

        debug.log("Memory", f"Retrieved {len(all_results)} memories for query")
        ranked = sorted(all_results, key=lambda x: x[2], reverse=True)[:k]
        formatted = []
        for doc, _, _ in ranked:
            meta_parts = []
            saved_at = doc.metadata.get("saved_at")
            if saved_at:
                try:
                    meta_parts.append(f"saved {datetime.fromtimestamp(saved_at).strftime('%Y-%m-%d')}")
                except Exception:
                    pass
            uname = doc.metadata.get("user_name")
            if uname:
                meta_parts.append(f"about {uname}")
            category = doc.metadata.get("category")
            if category and category != "fact":
                meta_parts.append(category)
            importance = doc.metadata.get("importance")
            if importance and int(importance) >= 4:
                meta_parts.append(f"priority {importance}")
            meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
            formatted.append(f"- {doc.page_content}{meta_str}")

        result = "\n".join(formatted)
        if warning:
            result = f"⚠️ {warning}\n\n" + result
        return result
    except Exception as e:
        debug.log("Memory", f"Retrieval error: {e}")
        return ""


from llm_utils import _extract_response_text


def _is_contradiction(new_fact: str, existing_fact: str) -> bool:
    try:
        prompt = (
            f"Are these two facts contradictory (i.e. they cannot both be true)? "
            f"Answer ONLY 'YES' or 'NO'.\n\n"
            f"Fact A: {existing_fact}\n"
            f"Fact B: {new_fact}"
        )
        response = _contradiction_model.invoke([
            SystemMessage(content="You are a precise logic checker. Answer only YES or NO."),
            HumanMessage(content=prompt),
        ])
        answer = _extract_response_text(response).upper()
        return "YES" in answer
    except Exception as e:
        debug.log("Memory", f"Contradiction check failed: {e}")
        return False


def tiered_dedup(fact: str, user_id: int | None) -> tuple[str | None, bool]:
    """Check if a fact duplicates or contradicts an existing memory.

    Returns (doc_id_to_replace, is_near_duplicate_or_contradiction):
      - (None, False)            -> genuinely new fact, save it
      - (doc_id, False)          -> near-duplicate (similarity >= 0.92), skip saving
      - (doc_id, True)           -> contradiction (0.75-0.92), should replace
    """
    try:
        all_candidates = []
        seen_ids = set()

        if user_id:
            user_results = _vector_db.similarity_search_with_relevance_scores(
                fact, k=5, filter={"user_id": user_id}
            )
            for doc, score in user_results:
                if score < CONTRADICTION_THRESHOLD:
                    continue
                doc_id = doc.id if hasattr(doc, 'id') else doc.metadata.get("id")
                dedup_key = doc_id if doc_id is not None else doc.page_content.strip().lower()
                if dedup_key in seen_ids:
                    continue
                seen_ids.add(dedup_key)
                all_candidates.append((doc, score, doc_id, True))

        general_results = _vector_db.similarity_search_with_relevance_scores(fact, k=5)
        for doc, score in general_results:
            if score < CONTRADICTION_THRESHOLD:
                continue
            doc_id = doc.id if hasattr(doc, 'id') else doc.metadata.get("id")
            dedup_key = doc_id if doc_id is not None else doc.page_content.strip().lower()
            if dedup_key in seen_ids:
                continue
            seen_ids.add(dedup_key)
            is_user = user_id is not None and doc.metadata.get("user_id") == user_id
            all_candidates.append((doc, score, doc_id, is_user))

        # Check highest-similarity candidates first
        all_candidates.sort(key=lambda x: x[1], reverse=True)

        for doc, score, doc_id, is_user in all_candidates:
            if score >= NEAR_DUPLICATE_THRESHOLD:
                # Near-duplicate: if it belongs to this user, skip saving
                if is_user:
                    return doc_id, False
                # Not this user's memory — could be about someone else, save as new
                continue
            elif score >= CONTRADICTION_THRESHOLD:
                if is_user:
                    if _is_contradiction(fact, doc.page_content):
                        return doc_id, True
                # For non-user memories at this similarity, skip contradiction check
                # to avoid overwriting someone else's memory

        return None, False
    except Exception as e:
        debug.log("Memory", f"Dedup check failed: {e}")
        return None, False


def _expire_old_memories():
    try:
        collection = _vector_db._collection
        result = collection.get(include=["metadatas"])
        if not result["ids"]:
            return
        expired_ids = []
        now = datetime.now().timestamp()
        for doc_id, meta in zip(result["ids"], result["metadatas"]):
            saved_at = meta.get("saved_at", 0)
            last_accessed = meta.get("last_accessed", 0)
            importance = meta.get("importance", 3)
            if not isinstance(importance, (int, float)):
                importance = 3
            cutoff = max(saved_at, last_accessed) + (MEMORY_EXPIRY_BASE_DAYS * int(importance) * 86400)
            if now > cutoff:
                expired_ids.append(doc_id)
        if expired_ids:
            collection.delete(ids=expired_ids)
            debug.log("Memory", f"Expired {len(expired_ids)} old memories")
    except Exception as e:
        debug.log("Memory", f"Expiry cleanup failed: {e}")


def memory_count() -> int:
    return _vector_db._collection.count()