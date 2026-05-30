# bong_tools.py — LangChain tool definitions and shared state for Bong

import os
import random
import re
from datetime import datetime
from pathlib import Path

from ddgs import DDGS
from langchain_core.tools import tool
from yt_dlp import YoutubeDL

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import bong_tools  # Self-reference so tools can access module-level vars reliably
import debug

# Directory where downloaded mp3 files are stored
DOWNLOAD_DIR = Path(__file__).parent / "saved_sounds"
DOWNLOAD_DIR.mkdir(exist_ok=True)
# Directory where saved images are stored
IMAGE_DIR = Path(__file__).parent / "saved_images"
IMAGE_DIR.mkdir(exist_ok=True)
# Directory where saved text files are stored
TEXT_DIR = Path(__file__).parent / "saved_texts"
TEXT_DIR.mkdir(exist_ok=True)

# --- Vector DB for long-term memory ---
DB_DIR = Path(__file__).parent / "chroma_db"
_embeddings = OllamaEmbeddings(model="nomic-embed-text", keep_alive=-1)
_vector_db = Chroma(
    collection_name="bong_memories",
    embedding_function=_embeddings,
    persist_directory=str(DB_DIR),
)

_BOILERPLATE = re.compile(r"\bbong\b['']?s?\b", re.IGNORECASE)
_USERID_TAG = re.compile(r"\s*\(userID:?\s*\d+\)", re.IGNORECASE)
# Percentage boost applied to user-specific memory scores (0.1 = 10% boost)
USER_MEMORY_SCORE_BOOST = 0.25

def _clean_for_embedding(text: str) -> str:
    """Remove bot boilerplate from text before embedding/search to reduce noise."""
    text = _BOILERPLATE.sub("", text)
    text = _USERID_TAG.sub("", text)
    return text.strip()

def retrieve_memories(query: str, username: str = "", user_id: int = None, k: int = 10) -> str:
    """Retrieve the k most relevant long-term memories for a given query.
    If user_id is provided, runs a filtered search for that user's memories first.
    If username is provided, runs a second search using the username and merges results.
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
                dedup_key = doc_id or norm
                if dedup_key in seen_ids:
                    continue
                seen_ids.add(dedup_key)
                adjusted_score = score * (1.0 + bong_tools.USER_MEMORY_SCORE_BOOST) if from_user_search else score
                all_results.append((doc.page_content, adjusted_score))

        if not all_results:
            debug.log("Memory", "No relevant memories found")
            return ""
        debug.log("Memory", f"Retrieved {len(all_results)} memories for query")
        return "\n".join(f"- {m}" for m, _ in sorted(all_results, key=lambda x: x[1], reverse=True))
    except Exception as e:
        debug.log("Memory", f"Retrieval error: {e}")
        return ""


# Shared state — tools write to these during the sync tool loop,
# and the cog reads/acts on them afterwards since Discord calls are async
pending_reactions = []       # Emoji reactions queued by the react tool
pending_join_voice = None    # User ID to join voice with, set by join_voice
pending_leave_voice = None    # Flag set by leave_voice
pending_shutdown = False     # Flag set by shutdown
pending_play_audio = None    # File path to play in voice chat, set by play_audio
pending_pause = False        # Flag set by pause_audio
pending_resume = False       # Flag set by resume_audio
pending_stop = False         # Flag set by stop_audio
pending_skip = False          # Flag set by skip_audio
pending_skip_target = None   # File path for skip's next track (independent of pending_play_audio)
pending_skip_info = ""      # Next track name set by skip_audio

pending_send_image = None    # File path to send as an image, set by send_image
pending_send_text = None     # File path to send as a text file, set by send_text

voice_connected = False    # Set by the cog before the tool loop; True if bot is in a voice channel
caller_in_voice = False    # Set by the cog before the tool loop; True if the user issuing commands is in a voice channel
current_user_id = None     # Set by the cog before the tool loop; Discord user ID of the current user

shuffle_enabled = False
loop_enabled = False
loop_track = None
current_track = None

image_library = []

def refresh_image_library():
    bong_tools.image_library = sorted(
        p for p in bong_tools.IMAGE_DIR.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
    )

text_library = []

def refresh_text_library():
    bong_tools.text_library = sorted(
        p for p in bong_tools.TEXT_DIR.iterdir()
        if p.suffix.lower() in (".txt", ".md", ".py", ".json", ".csv", ".xml", ".yaml", ".yml", ".cfg", ".ini", ".log", ".toml", ".rs", ".js", ".ts", ".html", ".css", ".sh", ".bat")
    )

music_library = []

def refresh_music_library():
    bong_tools.music_library = sorted(bong_tools.DOWNLOAD_DIR.glob("*.mp3"))

refresh_music_library()
refresh_image_library()
refresh_text_library()

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
def clear_console() -> str:
    """Clear the bot's console/terminal screen. Use this when the user asks you to clear the console or clean up the terminal.
    """
    os.system("cls" if os.name == "nt" else "clear")
    return "Console cleared"
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

    is_url = "youtube.com" in query or "youtu.be" in query
    url = query if is_url else None

    query_lower = query.lower()
    fuzzy_matches = [f for f in bong_tools.music_library if query_lower in f.stem.lower() or f.stem.lower() in query_lower]
    if fuzzy_matches:
        matched = ", ".join(f.stem for f in fuzzy_matches)
        return f"A similar song is already in the library: {matched}. Use play_audio to play it instead of downloading again."

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

    try:
        with YoutubeDL({"quiet": True, "no_warnings": True}) as probe:
            info = probe.extract_info(url, download=False)
            title = info.get("title", "unknown")
            clean_title = re.sub(r'[^\w\s\[\]\(\)\{\}]', '', title).strip()
            clean_title = re.sub(r' {2,}', ' ', clean_title)
            if not clean_title:
                clean_title = f"untitled_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            candidate = DOWNLOAD_DIR / f"{clean_title}.mp3"
            if candidate.exists():
                return f"'{clean_title}' is already in the library. Use play_audio with '{clean_title}' to play it."
        out_template = str(DOWNLOAD_DIR / f"{clean_title}.%(ext)s")
        ydl_opts = {
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
        with YoutubeDL(ydl_opts) as ydl:
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

# --- Audio playback tools ---
# These set pending flags that the cog reads and acts on asynchronously

@tool
def play_audio(index: int) -> str:
    """Play a downloaded mp3 file in the voice channel you are currently in. Only works if the user is in a voice channel — if they are not, tell them to join one and do not call this tool. Use the index number from list_music to select the track.
    Args:
        index: The index number of the track from list_music (e.g. 0, 1, 2).
    """
    if not bong_tools.caller_in_voice:
        return "The user needs to be in a voice channel to use music commands. This might be someone trolling from outside the voice channel."
    files = bong_tools.music_library
    if not files:
        return "No music files available. Download some first."
    if not bong_tools.voice_connected and not bong_tools.pending_join_voice:
        return "Not in a voice channel. Join a voice channel first using join_voice before playing music."
    if index < 0 or index >= len(files):
        return f"Index {index} out of range. Use list_music to see available tracks (0-{len(files)-1})."
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
    return "Vision request handled by cog."

# --- System tools ---

@tool
def save_memory(fact: str) -> str:
    """Save an important fact to long-term memory. Use this to remember things about users, preferences, inside jokes, or any information worth recalling later. Be selective — only save things that are genuinely useful to remember.
    Args:
        fact: A concise fact or piece of information to remember (e.g. "Eve loves dubstep and skrillex", "Radon is an orange fox who likes cars").
    """
    try:
        similar = bong_tools._vector_db.similarity_search_with_relevance_scores(fact, k=3, filter={"user_id": bong_tools.current_user_id})
        for doc, score in similar:
            if score >= 0.7:
                return f"Already remembered something similar: {doc.page_content}"
        clean_fact = bong_tools._clean_for_embedding(fact)
        bong_tools._vector_db.add_texts([clean_fact], metadatas=[{"user_id": bong_tools.current_user_id}])
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
def shutdown() -> str:
    """Shut down the bot. Only use this when an authorized user explicitly asks you to shut down.
    """
    bong_tools.pending_shutdown = True
    return "Shutting down"

# All tools the model can call
tools = [react, describe_image, read_text_file, join_voice, leave_voice, clear_console, current_time, web_search, youtube_search, download_music, list_music, play_audio, loop_audio, pause_audio, resume_audio, stop_audio, skip_audio, music_shuffle_enabled, list_images, send_image, list_texts, send_text, save_memory, recall_memories_by_userid, recall_memories_general, shutdown]

# Lookup dict from tool name to tool function for dispatching tool calls
tool_map = {t.name: t for t in tools}