# bong_tools.py — LangChain tool definitions and shared state for Bong
#
# This file defines all the tools the LLM can call (react, web_search, play_audio,
# save_memory, etc.) and the shared state variables that the cog (bong.py) reads
# after the tool loop finishes. Since LangChain's @tool decorator creates sync
# functions and Discord's API is async, tools write to pending_* flags here and
# the cog dispatches the actual Discord calls afterwards.
#
# The file also manages the ChromaDB vector store for long-term memory, the music/
# image/text file libraries, and the bot's Discord user ID.

import random
import re
from datetime import datetime, timedelta
from pathlib import Path

# DuckDuckGo search client
from ddgs import DDGS
# LangChain's @tool decorator — wraps a function so the LLM can call it by name
from langchain_core.tools import tool
# YouTube downloader — used to fetch mp3 audio from URLs
from yt_dlp import YoutubeDL

# ChromaDB vector store for persistent long-term memory
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Self-reference import — tools use bong_tools.X to read/write module-level state
# reliably even after hot reloads (importlib.reload replaces the module object)
import bong_tools
import debug

# --- Directory paths for saved media ---
# Path(__file__).parent resolves to the bot's root directory regardless of cwd
DOWNLOAD_DIR = Path(__file__).parent / "saved_sounds"
DOWNLOAD_DIR.mkdir(exist_ok=True)  # Create the folder if it doesn't exist yet
IMAGE_DIR = Path(__file__).parent / "saved_images"
IMAGE_DIR.mkdir(exist_ok=True)
TEXT_DIR = Path(__file__).parent / "saved_texts"
TEXT_DIR.mkdir(exist_ok=True)

# --- Vector DB for long-term memory ---
# ChromaDB stores text embeddings locally in chroma_db/. When save_memory is called,
# the fact is embedded with nomic-embed-text and stored for later semantic search.
DB_DIR = Path(__file__).parent / "chroma_db"
_embeddings = OllamaEmbeddings(model="nomic-embed-text", keep_alive=-1)
_vector_db = Chroma(
    collection_name="bong_memories",
    embedding_function=_embeddings,
    persist_directory=str(DB_DIR),
)

# Regex patterns used to clean text before embedding — strips "Bong"/"Bong's"
# and "(userID: 123456)" tags so the embedding focuses on the actual content
_BOILERPLATE = re.compile(r"\bbong\b['']?s?\b", re.IGNORECASE)
_USERID_TAG = re.compile(r"\s*\(userID:?\s*\d+\)", re.IGNORECASE)
# Score boost applied to user-specific memory matches vs general matches (0.25 = 25%)
USER_MEMORY_SCORE_BOOST = 0.25
# How many days before memories expire automatically
MEMORY_EXPIRY_DAYS = 180
# Minimum similarity score for a memory to be considered a potential contradiction
CONTRADICTION_THRESHOLD = 0.75
# Lightweight LLM used to judge whether two memories truly contradict each other
_contradiction_model = ChatOllama(model="gemma3:12b-cloud", temperature=0.0, num_predict=5, keep_alive=-1)


def _clean_for_embedding(text: str) -> str:
    """Remove bot boilerplate from text before embedding/search to reduce noise."""
    text = _BOILERPLATE.sub("", text)
    text = _USERID_TAG.sub("", text)
    return text.strip()


def _apply_recency_boost(score: float, saved_at: float, halflife_days: float = 60.0) -> float:
    """Boost a memory's score based on how recently it was saved.

    Uses exponential decay: memories are worth half their boost after `halflife_days`.
    The boost ranges from +0.15 (just saved) down to near 0 for very old memories.
    """
    if not saved_at:
        return score
    age_days = (datetime.now().timestamp() - saved_at) / 86400.0
    if age_days < 0:
        age_days = 0
    recency_boost = 0.15 * (0.5 ** (age_days / halflife_days))
    return score + recency_boost


def _batch_increment_access_counts(doc_ids: list):
    """Increment the access_count metadata on multiple memory documents at once.

    Takes a list of doc IDs, fetches all their metadatas in one call, then
    updates them all in one call — avoiding O(2n) DB round-trips per recall.
    Skips any IDs that are None.
    """
    valid_ids = [did for did in doc_ids if did is not None]
    if not valid_ids:
        return
    try:
        collection = bong_tools._vector_db._collection
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
    """Retrieve the k most relevant long-term memories for a given query.

    Searches work in three layers, merged and deduplicated:
      1. If user_id is given — search only that user's saved memories (score boosted)
      2. General search across all memories
      3. If username is given — search by username as a secondary query

    Scores are adjusted by:
      - User-scoped boost (personal memories rank higher)
      - Recency boost (newer memories rank higher)
      - Frequency boost (frequently recalled memories rank higher)

    Returns a newline-separated list of memories with metadata, sorted by relevance, or "" if none found.
    """
    try:
        seen_ids = set()
        all_results = []

        cleaned_query = bong_tools._clean_for_embedding(query)
        cleaned_name = bong_tools._clean_for_embedding(username) if username else ""

        searches = []
        is_user_search = []

        if user_id:
            searches.append(bong_tools._vector_db.similarity_search_with_relevance_scores(
                cleaned_query, k=k, filter={"user_id": user_id}
            ))
            is_user_search.append(True)

        searches.append(bong_tools._vector_db.similarity_search_with_relevance_scores(cleaned_query, k=k))
        is_user_search.append(False)

        if cleaned_name:
            searches.append(bong_tools._vector_db.similarity_search_with_relevance_scores(cleaned_name, k=k))
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

                adjusted_score = score * (1.0 + bong_tools.USER_MEMORY_SCORE_BOOST) if from_user_search else score

                saved_at = doc.metadata.get("saved_at")
                if saved_at:
                    adjusted_score = bong_tools._apply_recency_boost(adjusted_score, saved_at)

                access_count = doc.metadata.get("access_count", 0)
                if access_count:
                    adjusted_score += min(0.05 * access_count, 0.25)

                all_results.append((doc, doc_id, adjusted_score))

        if not all_results:
            debug.log("Memory", "No relevant memories found")
            return ""

        bong_tools._batch_increment_access_counts([doc_id for _, doc_id, _ in all_results])

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
    """Extract plain text from an LLM response, handling both str and list content."""
    content = response.content
    if isinstance(content, list):
        return "".join(chunk.text if hasattr(chunk, "text") else str(chunk) for chunk in content)
    return str(content or "")


def _is_contradiction(new_fact: str, existing_fact: str) -> bool:
    """Use a lightweight LLM to judge whether a new fact contradicts an existing one.

    Returns True if the two facts are mutually exclusive — i.e. they cannot both
    be true at the same time about the same subject. Things like "Eve likes rock"
    and "Eve likes dubstep" are NOT contradictory and return False.
    """
    try:
        prompt = (
            f"Are these two facts contradictory (i.e. they cannot both be true)? "
            f"Answer ONLY 'YES' or 'NO'.\n\n"
            f"Fact A: {existing_fact}\n"
            f"Fact B: {new_fact}"
        )
        response = bong_tools._contradiction_model.invoke([
            SystemMessage(content="You are a precise logic checker. Answer only YES or NO."),
            HumanMessage(content=prompt),
        ])
        answer = bong_tools._extract_response_text(response).upper()
        return "YES" in answer
    except Exception as e:
        debug.log("Memory", f"Contradiction check failed: {e}")
        return False


def _find_contradiction(fact: str, user_id: int | None) -> str | None:
    """Find an existing memory that semantically contradicts a new fact.

    First finds highly similar memories via embedding search, then uses a
    lightweight LLM (gemma3:12b-cloud) to judge whether the similarity is
    actually a contradiction. Returns the doc ID of the contradictory memory,
    or None if no contradiction found.
    """
    try:
        candidates = []

        if user_id:
            similar = bong_tools._vector_db.similarity_search_with_relevance_scores(
                fact, k=5, filter={"user_id": user_id}
            )
            for doc, score in similar:
                if score >= bong_tools.CONTRADICTION_THRESHOLD:
                    candidates.append(doc)

        if not candidates:
            similar_general = bong_tools._vector_db.similarity_search_with_relevance_scores(
                fact, k=5
            )
            for doc, score in similar_general:
                if score >= bong_tools.CONTRADICTION_THRESHOLD:
                    if not user_id or doc.metadata.get("user_id") == user_id:
                        candidates.append(doc)

        for doc in candidates:
            if bong_tools._is_contradiction(fact, doc.page_content):
                return doc.id if hasattr(doc, 'id') else doc.metadata.get("id")

        return None
    except Exception:
        return None


def _expire_old_memories(days: int = MEMORY_EXPIRY_DAYS):
    """Delete memories older than the given number of days."""
    try:
        cutoff = (datetime.now() - timedelta(days=days)).timestamp()
        collection = bong_tools._vector_db._collection
        result = collection.get(where={"saved_at": {"$lt": cutoff}})
        if result["ids"]:
            collection.delete(ids=result["ids"])
            debug.log("Memory", f"Expired {len(result['ids'])} old memories")
    except Exception as e:
        debug.log("Memory", f"Expiry cleanup failed: {e}")



# --- Bong's Discord user ID ---
# Used by the classifier in bong.py to detect mentions/pings directed at the bot.
# Change this if the bot runs under a different Discord account.
BOT_USER_ID = "698627881760456724"

# --- Shared pending state ---
# These flags are written to by the sync tool functions during the LLM's tool loop,
# then read and acted on by the async cog in bong.py after the loop finishes.
# This bridge pattern exists because LangChain tools are sync but Discord calls are async.
pending_reactions = []       # List of emoji strings queued by the react tool
pending_join_voice = None    # Discord user ID to join voice with (set by join_voice)
pending_leave_voice = None   # Flag set by leave_voice
pending_shutdown = False      # Flag set by shutdown
pending_play_audio = None    # File path of the mp3 to play (set by play_audio)
pending_pause = False        # Flag set by pause_audio
pending_resume = False       # Flag set by resume_audio
pending_stop = False         # Flag set by stop_audio
pending_skip = False         # Flag set by skip_audio
pending_skip_target = None   # File path for the next track after skip (independent of pending_play_audio)
pending_skip_info = ""       # Human-readable name of the skip target track

pending_send_image = None    # File path to send as an image attachment
pending_send_text = None     # File path to send as a text file attachment

# --- Voice/music state (set by the cog before each tool loop) ---
voice_connected = False   # True if bot is currently in a voice channel
caller_in_voice = False   # True if the user who sent the message is in a voice channel
current_user_id = None    # Discord user ID of the user who sent the current message

# --- Authorization and playback state ---
authorized = False        # Whether the current user is in ALLOWED_USERS (set by cog)
current_username = ""     # Display name of the user who sent the current message (set by cog)
shuffle_enabled = False   # Whether shuffle mode is on
loop_enabled = False      # Whether loop mode is on
loop_track = None         # File path of the track being looped (None = loop current track)
current_track = None       # File path of the currently playing track


def reset_pending():
    """Clear all pending state flags. Called from the cog on exception to prevent stale state leaking into the next message."""
    bong_tools.pending_reactions.clear()
    bong_tools.pending_join_voice = None
    bong_tools.pending_leave_voice = None
    bong_tools.pending_shutdown = False
    bong_tools.pending_play_audio = None
    bong_tools.pending_pause = False
    bong_tools.pending_resume = False
    bong_tools.pending_stop = False
    bong_tools.pending_skip = False
    bong_tools.pending_skip_target = None
    bong_tools.pending_skip_info = ""
    bong_tools.pending_send_image = None
    bong_tools.pending_send_text = None


# --- File library caches ---
# These lists are refreshed on demand by scanning the respective directories.
# They're populated once at module load and then updated by refresh functions.
image_library = []

def refresh_image_library():
    """Rescan the saved_images directory and update image_library."""
    bong_tools.image_library = sorted(
        p for p in bong_tools.IMAGE_DIR.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
    )

text_library = []

def refresh_text_library():
    """Rescan the saved_texts directory and update text_library."""
    bong_tools.text_library = sorted(
        p for p in bong_tools.TEXT_DIR.iterdir()
        if p.suffix.lower() in (".txt", ".md", ".py", ".json", ".csv", ".xml", ".yaml", ".yml", ".cfg", ".ini", ".log", ".toml", ".rs", ".js", ".ts", ".html", ".css", ".sh", ".bat")
    )

music_library = []

def refresh_music_library():
    """Rescan the saved_sounds directory and update music_library."""
    bong_tools.music_library = sorted(bong_tools.DOWNLOAD_DIR.glob("*.mp3"))

# Initial population of all libraries at module load time
refresh_music_library()
refresh_image_library()
refresh_text_library()


# ========== Tool definitions ==========
# Each function decorated with @tool becomes callable by the LLM.
# The docstring serves as the tool's description — the LLM uses it to decide
# when and how to call each tool. Args become parameters the LLM must provide.

@tool
def react(emojis: str) -> str:
    """React to the user's message with one or more emojis. Use this to express emotion or acknowledge the message.
    Args:
        emojis: One or more emoji characters to react with, separated by spaces (e.g. ❤️, 👍 😂, 🤔 💡 🎉)
    """
    for emoji in emojis.split():
        bong_tools.pending_reactions.append(emoji)
    return f"Reacted with {emojis}"

@tool
def current_time() -> str:
    """Get the current system time. Use this when the user asks what time it is or needs to know the current time.
    """
    return datetime.now().strftime("%H:%M")

# --- Search tools ---

@tool
def web_search(query: str) -> str:
    """Search the web for information. Use this when you need to look up facts, news, or any information you don't know.
    Args:
        query: The search query string.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        return f"{results[0]['title']}: {results[0]['body']}"
    except Exception as e:
        return f"Search error: {e}"
        
@tool
def youtube_search(query: str) -> str:
    """Search YouTube for videos. Use this when the user wants to find a YouTube video or when you need to find a YouTube URL to download audio from.
    Args:
        query: The search query string.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.videos(query, max_results=3))
        if not results:
            return "No YouTube results found."
        lines = []
        for r in results:
            title = r.get("title", "Untitled")
            url = r.get("content", r.get("url", ""))
            # Only include results that are actually YouTube links
            if "youtube.com" in url or "youtu.be" in url:
                lines.append(f"- {title}: {url}")
        if not lines:
            return "No YouTube results found."
        return "YouTube results:\n" + "\n".join(lines[:3])
    except Exception as e:
        return f"YouTube search error: {e}"

# --- Audio download tools ---

@tool
def download_music(query: str) -> str:
    """Download an mp3 audio file. Accepts either a YouTube URL or a song name. If given a song name, automatically searches YouTube for it. The current music library is listed above — check it before downloading, and if the song is already there use play_audio instead.
    Args:
        query: A YouTube URL or a song name to search for on YouTube (e.g. "Jersey by Mayday Parade" or "https://youtube.com/watch?v=...").
    """
    bong_tools.refresh_music_library()

    # Determine if the query is a direct URL or a search term
    is_url = "youtube.com" in query or "youtu.be" in query
    url = query if is_url else ""

    # Check if a similar song already exists in the library (fuzzy match by name)
    query_lower = query.lower()
    fuzzy_matches = [(i, f) for i, f in enumerate(bong_tools.music_library) if query_lower in f.stem.lower() or f.stem.lower() in query_lower]
    if fuzzy_matches:
        matched = ", ".join(f"{f.stem} (index {i})" for i, f in fuzzy_matches)
        return f"A similar song is already in the library: {matched}. Use play_audio to play it instead of downloading again."

    # If not a URL, search YouTube for a matching video
    if not is_url:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.videos(f"{query} official audio", max_results=5))
            for r in results:
                candidate_url = r.get("content", r.get("url", ""))
                if "youtube.com" in candidate_url or "youtu.be" in candidate_url:
                    url = candidate_url
                    break
            if not url:
                return f"Could not find a YouTube result for '{query}'. Try providing a YouTube URL directly."
        except Exception as e:
            return f"YouTube search failed: {e}"

    # Download the audio using yt-dlp
    try:
        # First pass: extract info without downloading to get the clean title
        with YoutubeDL({"quiet": True, "no_warnings": True}) as probe:
            info = probe.extract_info(url, download=False)
            title = str(info.get("title", "unknown") or "unknown")
            # Strip characters that are unsafe for filenames
            clean_title = re.sub(r'[^\w\s\[\]\(\)\{\}]', '', title).strip()
            clean_title = re.sub(r' {2,}', ' ', clean_title)
            if not clean_title:
                clean_title = f"untitled_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            candidate = bong_tools.DOWNLOAD_DIR / f"{clean_title}.mp3"
            if candidate.exists():
                return f"'{clean_title}' is already in the library. Use play_audio with '{clean_title}' to play it."
        # Second pass: actually download and convert to mp3
        out_template = str(bong_tools.DOWNLOAD_DIR / f"{clean_title}.%(ext)s")
        ydl_opts: dict = {
            "format": "bestaudio/best",
            "noplaylist": True,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
        }
        with YoutubeDL(ydl_opts) as ydl:  # type: ignore[arg-type]
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            mp3_path = Path(filename).with_suffix(".mp3")
        if not mp3_path.exists():
            return f"Download completed but file not found: {title}"
        bong_tools.refresh_music_library()
        return f"Downloaded '{title}'. Saved to saved_sounds folder."
    except Exception as e:
        return f"Download failed: {e}"

@tool
def list_music() -> str:
    """List all downloaded mp3 files in the saved_sounds folder. Use this when the user wants to browse or play music. The full library is provided in context — use the index numbers to select tracks with play_audio.
    """
    bong_tools.refresh_music_library()
    files = bong_tools.music_library
    if not files:
        return "No music files found in the saved_sounds folder."
    return f"{len(files)} songs available. The full library is in your context."

@tool
def search_music(query: str) -> str:
    """Search the music library by name. Use this when the user wants to play a song and you need to find its exact index. Always use this before play_audio if the user mentions a song by name instead of an index number.
    Args:
        query: Part of the song name to search for (e.g. "paradise", "mayday", "jersey").
    """
    bong_tools.refresh_music_library()
    files = bong_tools.music_library
    if not files:
        return "No music files available. Download some first."
    q = query.lower()
    matches = [(i, f) for i, f in enumerate(files) if q in f.stem.lower()]
    if not matches:
        return f"No songs matching '{query}' found. Use list_music to see all available tracks."
    lines = [f"  {i}: {f.stem}" for i, f in matches]
    return f"Found {len(matches)} match(es) for '{query}':\n" + "\n".join(lines)

# --- Audio playback tools ---
# These set pending flags that the cog reads and acts on asynchronously

@tool
def play_audio(index: int = -1, name: str = "") -> str:
    """Play a downloaded mp3 file in the voice channel the user is currently in. Only works if the user is in a voice channel. You can provide either an index number from list_music, or a song name to fuzzy-match against the library. Always use search_music first if the user gives a song name.
    Args:
        index: The index number of the track from list_music (e.g. 0, 1, 2). Use -1 if providing a name instead.
        name: A song name to fuzzy-match against the library. Only used if index is -1 or not provided.
    """
    if not bong_tools.caller_in_voice:
        return "The user needs to be in a voice channel to use music commands. This might be someone trolling from outside the voice channel."
    files = bong_tools.music_library
    if not files:
        return "No music files available. Download some first."
    if not bong_tools.voice_connected and not bong_tools.pending_join_voice:
        return "Not in a voice channel. Join a voice channel first using join_voice before playing music."
    # If a name was provided, try to match it against the library
    if name:
        bong_tools.refresh_music_library()
        files = bong_tools.music_library
        name_lower = name.lower()
        # Try exact match first
        exact = [(i, f) for i, f in enumerate(files) if f.stem.lower() == name_lower]
        if exact:
            i, f = exact[0]
            bong_tools.pending_play_audio = str(f)
            return f"Queued '{f.stem}' for playback."
        # Fall back to partial match (substring in either direction)
        partial = [(i, f) for i, f in enumerate(files) if name_lower in f.stem.lower() or f.stem.lower() in name_lower]
        if partial:
            i, f = partial[0]
            bong_tools.pending_play_audio = str(f)
            return f"Queued '{f.stem}' for playback."
        return f"No song matching '{name}' found. Use search_music to find the right track."
    # Otherwise, use the index
    if index < 0 or index >= len(files):
        return f"Index {index} out of range. Use list_music or search_music to find the right track (0-{len(files)-1})."
    bong_tools.pending_play_audio = str(files[index])
    return f"Queued '{files[index].stem}' for playback."

@tool
def pause_audio() -> str:
    """Pause the currently playing audio in voice chat. Only use this if the user is in a voice channel — if they are not, tell them to join one and do not call this tool.
    """
    if not bong_tools.caller_in_voice:
        return "The user needs to be in a voice channel to use music commands. This might be someone trolling from outside the voice channel."
    bong_tools.pending_pause = True
    return "Pausing audio."

@tool
def resume_audio() -> str:
    """Resume paused audio in voice chat. Only use this if the user is in a voice channel — if they are not, tell them to join one and do not call this tool.
    """
    if not bong_tools.caller_in_voice:
        return "The user needs to be in a voice channel to use music commands. This might be someone trolling from outside the voice channel."
    bong_tools.pending_resume = True
    return "Resuming audio."

@tool
def stop_audio() -> str:
    """Stop audio playback in voice chat entirely. Only use this if the user is in a voice channel — if they are not, tell them to join one and do not call this tool.
    """
    if not bong_tools.caller_in_voice:
        return "The user needs to be in a voice channel to use music commands. This might be someone trolling from outside the voice channel."
    bong_tools.pending_stop = True
    return "Stopping audio."

@tool
def skip_audio() -> str:
    """Skip the currently playing song and play the next one. Only use this if the user is in a voice channel — if they are not, tell them to join one and do not call this tool. If shuffle is enabled the next song will be random, otherwise it does nothing since there is no queue.
    """
    if not bong_tools.caller_in_voice:
        return "The user needs to be in a voice channel to use music commands. This might be someone trolling from outside the voice channel."
    bong_tools.pending_skip = True
    bong_tools.refresh_music_library()
    files = bong_tools.music_library
    if bong_tools.shuffle_enabled and files:
        next_track = random.choice(files)
        bong_tools.pending_skip_target = str(next_track)
        bong_tools.pending_skip_info = next_track.stem
        return f"Skipping to '{next_track.stem}'."
    elif not bong_tools.shuffle_enabled:
        return "Shuffle is not enabled. Enable shuffle first or specify a song to skip to."
    return "No music files available to skip to."

@tool
def loop_audio(index: int = -1) -> str:
    """Loop the current song or a specific song by index. Only use this if the user is in a voice channel — if they are not, tell them to join one and do not call this tool. When enabled, the song will replay from the beginning when it finishes. Loop takes priority over shuffle. Call again to disable loop.
    Args:
        index: The index number of the track from list_music to loop. If not provided or -1, loops the currently playing song.
    """
    if not bong_tools.caller_in_voice:
        return "The user needs to be in a voice channel to use music commands. This might be someone trolling from outside the voice channel."
    # Toggle: calling again disables loop
    if bong_tools.loop_enabled:
        bong_tools.loop_enabled = False
        bong_tools.loop_track = None
        return "Loop disabled."
    if index >= 0:
        files = bong_tools.music_library
        if not files:
            return "No music files available. Download some first."
        if index >= len(files):
            return f"Index {index} out of range. Use list_music to see available tracks (0-{len(files)-1})."
        bong_tools.loop_enabled = True
        bong_tools.loop_track = str(files[index])
        return f"Looping '{files[index].stem}'."
    # Loop the currently playing song
    bong_tools.loop_enabled = True
    bong_tools.loop_track = None
    return "Looping the current song."

@tool
def music_shuffle_enabled(enabled: bool) -> str:
    """Enable or disable shuffle mode for music playback. Only use this if the user is in a voice channel — if they are not, tell them to join one and do not call this tool. When enabled, after a song finishes a random mp3 from the saved_sounds folder will play next.
    Args:
        enabled: True to enable shuffle, False to disable shuffle.
    """
    if not bong_tools.caller_in_voice:
        return "The user needs to be in a voice channel to use music commands. This might be someone trolling from outside the voice channel."
    bong_tools.shuffle_enabled = enabled
    state = "enabled" if bong_tools.shuffle_enabled else "disabled"
    return f"Shuffle mode is now {state}."

# --- Image tools ---

@tool
def list_images() -> str:
    """List all saved images in the saved_images folder. Use this when the user wants to browse or view saved images. The full library is provided in context — use the index numbers to select images with send_image.
    """
    bong_tools.refresh_image_library()
    files = bong_tools.image_library
    if not files:
        return "No images found in the saved_images folder."
    return f"{len(files)} images available. The full library is in your context."

@tool
def send_image(index: int) -> str:
    """Send a saved image to the chat. Use this when the user wants to see a saved image. Use the index number from list_images to select the image.
    Args:
        index: The index number of the image from list_images (e.g. 0, 1, 2).
    """
    bong_tools.refresh_image_library()
    files = bong_tools.image_library
    if not files:
        return "No images available."
    if index < 0 or index >= len(files):
        return f"Index {index} out of range. Use list_images to see available images (0-{len(files)-1})."
    bong_tools.pending_send_image = str(files[index])
    return f"Sending '{files[index].stem}'."

# --- Text file tools ---

@tool
def list_texts() -> str:
    """List all saved text files in the saved_texts folder. Use this when the user wants to browse or view saved text files. The full library is provided in context — use the index numbers to select files with send_text.
    """
    bong_tools.refresh_text_library()
    files = bong_tools.text_library
    if not files:
        return "No text files found in the saved_texts folder."
    return f"{len(files)} text files available. The full library is in your context."

@tool
def send_text(index: int) -> str:
    """Send a saved text file to the chat. Use this when the user wants to see a saved text file. Use the index number from list_texts to select the file.
    Args:
        index: The index number of the text file from list_texts (e.g. 0, 1, 2).
    """
    bong_tools.refresh_text_library()
    files = bong_tools.text_library
    if not files:
        return "No text files available."
    if index < 0 or index >= len(files):
        return f"Index {index} out of range. Use list_texts to see available text files (0-{len(files)-1})."
    bong_tools.pending_send_text = str(files[index])
    return f"Sending '{files[index].name}'."

@tool
def read_text_file(index: int = 0) -> str:
    """Read the contents of a text file attached to the current message. Use this when the user wants you to read, summarize, or answer questions about an attached text file.
    Args:
        index: The 0-based index of the text file attachment to read (default 0 for the first text file).
    """
    # Actual processing happens in bong.py's _handle_read_text_file,
    # because it needs async access to the message attachment data
    return "Text file request handled by cog."

# --- Voice channel tools ---

@tool
def join_voice(userID: int) -> str:
    """Join the voice channel that the user is currently in. Use this when the user wants you to join their voice channel.
    Args:
        userID: User ID of the person you want to join voice chat with.
    """
    bong_tools.pending_join_voice = userID
    bong_tools.voice_connected = True
    return "Joining voice channel"

@tool
def leave_voice() -> str:
    """Disconnect from the current voice channel. Use this when the user wants you to leave their voice channel.
    """
    bong_tools.pending_leave_voice = True
    bong_tools.voice_connected = False
    return "Leaving voice channel"

# --- Vision tools ---

@tool
def describe_image(index: int = 0, question: str = "Briefly describe this image in 1-2 sentences. Be concise.") -> str:
    """Describe an image attached to the current message using the vision model. Use this when the user wants you to look at or describe an image, read text in an image, or answer questions about an image. The attachment list tells you which images are available.
    Args:
        index: The 0-based index of the image attachment to describe (default 0 for the first image).
        question: What to ask about the image (default: brief description). Use "Read all the text in this image." for OCR, or "Describe this image in detail." for a thorough description.
    """
    # Actual processing happens in bong.py's _handle_describe_image,
    # because it needs async access to the vision model and base64 image data
    return "Vision request handled by cog."

# --- System tools ---

@tool
def save_memory(fact: str) -> str:
    """Save an important fact to long-term memory. Use this to remember things about users, preferences, inside jokes, or any information worth recalling later. Be selective — only save things that are genuinely useful to remember. If the fact contradicts or updates something already remembered, the old memory will be replaced automatically.
    Args:
        fact: A concise fact or piece of information to remember (e.g. "Eve loves dubstep and skrillex", "Radon is an orange fox who likes cars").
    """
    try:
        clean_fact = bong_tools._clean_for_embedding(fact)

        # Check for contradictions — if this fact supersedes an existing one, replace it
        contradiction_id = bong_tools._find_contradiction(clean_fact, bong_tools.current_user_id)
        if contradiction_id:
            collection = bong_tools._vector_db._collection
            try:
                old = collection.get(ids=[contradiction_id], include=["documents", "metadatas"])
                old_text = old["documents"][0] if old["documents"] else "(unknown)"
                old_meta = dict(old["metadatas"][0]) if old["metadatas"] else {}
                collection.delete(ids=[contradiction_id])
                # Preserve the original metadata but update the fact text and timestamp
                old_meta["saved_at"] = datetime.now().timestamp()
                if bong_tools.current_username:
                    old_meta["username"] = bong_tools.current_username
                bong_tools._vector_db.add_texts([clean_fact], metadatas=[old_meta])
                return f"Updated memory: {old_text} → {clean_fact}"
            except Exception as e:
                bong_tools._vector_db.add_texts(
                    [clean_fact],
                    metadatas=[{"user_id": bong_tools.current_user_id, "saved_at": datetime.now().timestamp(), "username": bong_tools.current_username or ""}],
                )
                return f"Remembered: {clean_fact} (failed to replace old: {e})"

        # No contradiction — check for exact/near duplicates across all users
        similar = bong_tools._vector_db.similarity_search_with_relevance_scores(clean_fact, k=3)
        for doc, score in similar:
            if score >= 0.7:
                existing_uid = doc.metadata.get("user_id")
                if existing_uid == bong_tools.current_user_id:
                    return f"Already remembered something similar: {doc.page_content}"

        bong_tools._vector_db.add_texts(
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
    results = bong_tools.retrieve_memories(query, user_id=bong_tools.current_user_id)
    if not results:
        return "No relevant memories found for this user."
    return results

@tool
def recall_memories_general(query: str) -> str:
    """Search all long-term memories regardless of user. Use this when you need to recall something about someone other than the current user, or a general fact not tied to a specific person.
    Args:
        query: What to search for (e.g. "Radon's fursona", "inside jokes", "who likes dubstep").
    """
    results = bong_tools.retrieve_memories(query)
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
        clean_query = bong_tools._clean_for_embedding(query)
        results = bong_tools._vector_db.similarity_search_with_relevance_scores(clean_query, k=3, filter={"user_id": bong_tools.current_user_id})
        for doc, score in results:
            if score >= 0.5:
                doc_id = doc.id if hasattr(doc, 'id') else doc.metadata.get("id")
                if doc_id:
                    collection = bong_tools._vector_db._collection
                    collection.delete(ids=[doc_id])
                    return f"Forgot: {doc.page_content}"
        return "No similar memory found to forget. Try describing it differently."
    except Exception as e:
        return f"Failed to forget memory: {e}"

@tool
def shutdown() -> str:
    """Shut down the bot. Only use this when an authorized user explicitly asks you to shut down. If the user is not authorized (not in the allowed users list), do NOT call this tool — instead tell them they don't have permission.
    """
    if not bong_tools.authorized:
        return "Cannot shut down: the user is not authorized to do this. Tell them they don't have permission to shut down the bot."
    bong_tools.pending_shutdown = True
    return "Shutting down"

# All tools the model can call — this list is bound to the LLM so it knows what's available
tools = [react, describe_image, read_text_file, join_voice, leave_voice, current_time, web_search, youtube_search, download_music, list_music, search_music, play_audio, loop_audio, pause_audio, resume_audio, stop_audio, skip_audio, music_shuffle_enabled, list_images, send_image, list_texts, send_text, save_memory, recall_memories_by_userid, recall_memories_general, forget_memory, shutdown]

# Lookup dict from tool name to tool function — used by dispatch_tool in bong.py
tool_map = {t.name: t for t in tools}