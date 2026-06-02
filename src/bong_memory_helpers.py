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

_BOILERPLATE = re.compile(r"\bbong\b['']?s?\b", re.IGNORECASE)
_USERID_TAG = re.compile(r"\s*\(userID:?\s*\d+\)", re.IGNORECASE)
USER_MEMORY_SCORE_BOOST = 0.25
MEMORY_EXPIRY_DAYS = 180
CONTRADICTION_THRESHOLD = 0.75
_contradiction_model = ChatOllama(model="gemma3:12b-cloud", temperature=0.0, num_predict=5, keep_alive=-1)


def _clean_for_embedding(text: str) -> str:
    text = _BOILERPLATE.sub("", text)
    text = _USERID_TAG.sub("", text)
    return text.strip()


def _apply_recency_boost(score: float, saved_at: float, halflife_days: float = 60.0) -> float:
    if not saved_at:
        return score
    age_days = (datetime.now().timestamp() - saved_at) / 86400.0
    if age_days < 0:
        age_days = 0
    recency_boost = 0.15 * (0.5 ** (age_days / halflife_days))
    return score + recency_boost


def _batch_increment_access_counts(doc_ids: list):
    valid_ids = [did for did in doc_ids if did is not None]
    if not valid_ids:
        return
    try:
        collection = bong_memory_helpers._vector_db._collection
        result = collection.get(ids=valid_ids, include=["metadatas"])
        if not result["metadatas"]:
            return
        updated_ids = []
        updated_metas = []
        for i, meta in enumerate(result["metadatas"]):
            new_meta = dict(meta)
            raw_count = new_meta.get("access_count", 0)
            new_meta["access_count"] = (int(raw_count) + 1) if isinstance(raw_count, (int, float)) else 1
            updated_ids.append(result["ids"][i])
            updated_metas.append(new_meta)
        collection.update(ids=updated_ids, metadatas=updated_metas)
    except Exception:
        pass


def retrieve_memories(query: str, username: str = "", user_id: int | None = None, k: int = 10) -> str:
    try:
        seen_ids = set()
        all_results = []
        cleaned_query = bong_memory_helpers._clean_for_embedding(query)
        cleaned_name = bong_memory_helpers._clean_for_embedding(username) if username else ""

        searches = []
        is_user_search = []
        if user_id:
            searches.append(bong_memory_helpers._vector_db.similarity_search_with_relevance_scores(
                cleaned_query, k=k, filter={"user_id": user_id}
            ))
            is_user_search.append(True)
        searches.append(bong_memory_helpers._vector_db.similarity_search_with_relevance_scores(cleaned_query, k=k))
        is_user_search.append(False)
        if cleaned_name:
            searches.append(bong_memory_helpers._vector_db.similarity_search_with_relevance_scores(cleaned_name, k=k))
            is_user_search.append(False)

        for search_docs, from_user_search in zip(searches, is_user_search):
            for doc, score in search_docs:
                if score < 0.5:
                    continue
                doc_id = doc.id if hasattr(doc, 'id') else doc.metadata.get("id")
                norm = doc.page_content.strip().lower()
                dedup_key = doc_id if doc_id is not None else norm
                if dedup_key in seen_ids:
                    continue
                seen_ids.add(dedup_key)
                adjusted_score = score * (1.0 + bong_memory_helpers.USER_MEMORY_SCORE_BOOST) if from_user_search else score
                saved_at = doc.metadata.get("saved_at")
                if saved_at:
                    adjusted_score = bong_memory_helpers._apply_recency_boost(adjusted_score, saved_at)
                access_count = doc.metadata.get("access_count", 0)
                if access_count:
                    adjusted_score += min(0.05 * access_count, 0.25)
                all_results.append((doc, doc_id, adjusted_score))

        if not all_results:
            debug.log("Memory", "No relevant memories found")
            return ""
        bong_memory_helpers._batch_increment_access_counts([doc_id for _, doc_id, _ in all_results])
        debug.log("Memory", f"Retrieved {len(all_results)} memories for query")
        formatted = []
        for doc, _, s in sorted(all_results, key=lambda x: x[2], reverse=True):
            meta_parts = []
            saved_at = doc.metadata.get("saved_at")
            if saved_at:
                try:
                    meta_parts.append(f"saved {datetime.fromtimestamp(saved_at).strftime('%Y-%m-%d')}")
                except Exception:
                    pass
            uname = doc.metadata.get("username")
            if uname:
                meta_parts.append(f"about {uname}")
            meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
            formatted.append(f"- {doc.page_content}{meta_str}")
        return "\n".join(formatted)
    except Exception as e:
        debug.log("Memory", f"Retrieval error: {e}")
        return ""


def _extract_response_text(response) -> str:
    content = response.content
    if isinstance(content, list):
        return "".join(chunk.text if hasattr(chunk, "text") else str(chunk) for chunk in content)
    return str(content or "")


def _is_contradiction(new_fact: str, existing_fact: str) -> bool:
    try:
        prompt = (
            f"Are these two facts contradictory (i.e. they cannot both be true)? "
            f"Answer ONLY 'YES' or 'NO'.\n\n"
            f"Fact A: {existing_fact}\n"
            f"Fact B: {new_fact}"
        )
        response = bong_memory_helpers._contradiction_model.invoke([
            SystemMessage(content="You are a precise logic checker. Answer only YES or NO."),
            HumanMessage(content=prompt),
        ])
        answer = bong_memory_helpers._extract_response_text(response).upper()
        return "YES" in answer
    except Exception as e:
        debug.log("Memory", f"Contradiction check failed: {e}")
        return False


def _find_contradiction(fact: str, user_id: int | None) -> str | None:
    try:
        candidates = []
        if user_id:
            similar = bong_memory_helpers._vector_db.similarity_search_with_relevance_scores(
                fact, k=5, filter={"user_id": user_id}
            )
            for doc, score in similar:
                if score >= bong_memory_helpers.CONTRADICTION_THRESHOLD:
                    candidates.append(doc)
        if not candidates:
            similar_general = bong_memory_helpers._vector_db.similarity_search_with_relevance_scores(fact, k=5)
            for doc, score in similar_general:
                if score >= bong_memory_helpers.CONTRADICTION_THRESHOLD:
                    if not user_id or doc.metadata.get("user_id") == user_id:
                        candidates.append(doc)
        for doc in candidates:
            if bong_memory_helpers._is_contradiction(fact, doc.page_content):
                return doc.id if hasattr(doc, 'id') else doc.metadata.get("id")
        return None
    except Exception:
        return None


def _expire_old_memories(days: int = MEMORY_EXPIRY_DAYS):
    try:
        cutoff = (datetime.now() - timedelta(days=days)).timestamp()
        collection = bong_memory_helpers._vector_db._collection
        result = collection.get(where={"saved_at": {"$lt": cutoff}})
        if result["ids"]:
            collection.delete(ids=result["ids"])
            debug.log("Memory", f"Expired {len(result['ids'])} old memories")
    except Exception as e:
        debug.log("Memory", f"Expiry cleanup failed: {e}")


import bong_memory_helpers