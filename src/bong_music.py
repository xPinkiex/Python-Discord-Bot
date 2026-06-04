import re
from datetime import datetime
from functools import wraps
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.tools import tool
from ddgs import DDGS
from yt_dlp import YoutubeDL
import bong_tools
import bong_song_stats
import user_data


def _check_music():
    if not user_data.has_permission(bong_tools.current_user_id, "music"):
        return "You don't have permission to use music commands. Ask an admin to grant you the music tag."
    return None


def _fuzzy_match_music(files, name):
    return [(i, f) for i, f in enumerate(files) if name in f.stem.lower()]


def _requires_voice(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not bong_tools.caller_in_voice:
            return "The user needs to be in a voice channel to use music commands. This might be someone trolling from outside the voice channel."
        return func(*args, **kwargs)
    return wrapper


@tool
def download_music(query: str) -> str:
    """Download an mp3 audio file. Accepts either a YouTube URL or a song name. If given a song name, automatically searches YouTube for it. The current music library is listed above — check it before downloading, and if the song is already there use play_audio instead. Requires the music permission tag.
    Args:
        query: A YouTube URL or a song name to search for on YouTube (e.g. "Jersey by Mayday Parade" or "https://youtube.com/watch?v=...").
    """
    denied = _check_music()
    if denied:
        return denied
    bong_tools.refresh_music_library()

    is_url = "youtube.com" in query or "youtu.be" in query
    url = query if is_url else ""

    if url:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if "list" in qs:
            del qs["list"]
            if qs:
                parsed = parsed._replace(query=urlencode(qs, doseq=True))
                url = urlunparse(parsed)
            else:
                url = urlunparse(parsed._replace(query=""))

    query_lower = query.lower()
    fuzzy_matches = _fuzzy_match_music(bong_tools.music_library, query_lower)
    if fuzzy_matches:
        matched = ", ".join(f"{f.stem} (index {i})" for i, f in fuzzy_matches)
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
        with YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as probe:
            info = probe.extract_info(url, download=False)
            title = str(info.get("title", "unknown") or "unknown")
            clean_title = re.sub(r'[^\w\s\[\]\(\)\{\}]', '', title).strip()
            clean_title = re.sub(r' {2,}', ' ', clean_title)
            if not clean_title:
                clean_title = f"untitled_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            candidate = bong_tools.DOWNLOAD_DIR / f"{clean_title}.mp3"
            if candidate.exists():
                return f"'{clean_title}' is already in the library. Use play_audio with '{clean_title}' to play it."
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
    """List all downloaded mp3 files in the saved_sounds folder. Use this when the user wants to browse or play music. The full library is provided in context — use the index numbers to select tracks with play_audio. Requires the music permission tag.
    """
    denied = _check_music()
    if denied:
        return denied
    bong_tools.refresh_music_library()
    files = bong_tools.music_library
    if not files:
        return "No music files found in the saved_sounds folder."
    return f"{len(files)} songs available. The full library is in your context."


@tool
def search_music(query: str) -> str:
    """Search the music library by name. Use this when the user wants to play a song and you need to find its exact index. Always use this before play_audio if the user mentions a song by name instead of an index number. Requires the music permission tag.
    Args:
        query: Part of the song name to search for (e.g. "paradise", "mayday", "jersey").
    """
    denied = _check_music()
    if denied:
        return denied
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


@tool
@_requires_voice
def play_audio(index: int = -1, name: str = "") -> str:
    """Play a downloaded mp3 file in the voice channel the user is currently in. If something is already playing, the song is added to the queue. Only works if the user is in a voice channel. You can provide either an index number from list_music, or a song name to fuzzy-match against the library. Always use search_music first if the user gives a song name. Requires the music permission tag.
    Args:
        index: The index number of the track from list_music (e.g. 0, 1, 2). Use -1 if providing a name instead.
        name: A song name to fuzzy-match against the library. Only used if index is -1 or not provided.
    """
    denied = _check_music()
    if denied:
        return denied
    bong_tools.refresh_music_library()
    files = bong_tools.music_library
    if not files:
        return "No music files available. Download some first."
    if not bong_tools.voice_connected and not bong_tools.pending_join_voice:
        return "Not in a voice channel. Join a voice channel first using join_voice before playing music."
    track_path = None
    track_name = ""
    if name:
        name_lower = name.lower()
        exact = [(i, f) for i, f in enumerate(files) if f.stem.lower() == name_lower]
        if exact:
            f = exact[0][1]
            track_path = str(f)
            track_name = f.stem
        else:
            partial = _fuzzy_match_music(files, name_lower)
            if partial:
                f = partial[0][1]
                track_path = str(f)
                track_name = f.stem
            else:
                return f"No song matching '{name}' found. Use search_music to find the right track."
    else:
        if index < 0 or index >= len(files):
            return f"Index {index} out of range. Use list_music or search_music to find the right track (0-{len(files)-1})."
        track_path = str(files[index])
        track_name = files[index].stem
    if bong_tools.current_track or bong_tools.pending_play_audio:
        bong_tools.song_queue.append(track_path)
        pos = len(bong_tools.song_queue)
        if bong_tools.loop_enabled and bong_tools.loop_track and not bong_tools.queue_snapshot:
            bong_tools.queue_snapshot = [bong_tools.loop_track] + list(bong_tools.song_queue)
            bong_tools.loop_track = None
            return f"Added '{track_name}' to the queue (position {pos}). Queue loop activated ({len(bong_tools.queue_snapshot)} songs)."
        if bong_tools.loop_enabled and bong_tools.queue_snapshot:
            bong_tools.queue_snapshot.append(track_path)
            return f"Added '{track_name}' to the queue (position {pos}, loop has {len(bong_tools.queue_snapshot)} songs)."
        return f"Added '{track_name}' to the queue (position {pos})."
    bong_tools.pending_play_audio = track_path
    bong_song_stats._increment_song(track_name)
    return f"Playing '{track_name}'."


@tool
@_requires_voice
def pause_audio() -> str:
    """Pause the currently playing audio in voice chat. Requires the music permission tag.
    """
    denied = _check_music()
    if denied:
        return denied
    bong_tools.pending_pause = True
    return "Pausing audio."


@tool
@_requires_voice
def resume_audio() -> str:
    """Resume paused audio in voice chat. Requires the music permission tag.
    """
    denied = _check_music()
    if denied:
        return denied
    bong_tools.pending_resume = True
    return "Resuming audio."


@tool
@_requires_voice
def stop_audio() -> str:
    """Stop audio playback in voice chat entirely and clear the song queue, loop, and shuffle state. Requires the music permission tag.
    """
    denied = _check_music()
    if denied:
        return denied
    bong_tools.pending_stop = True
    bong_tools.song_queue.clear()
    bong_tools.loop_enabled = False
    bong_tools.loop_track = None
    bong_tools.queue_snapshot = []
    bong_tools.shuffle_enabled = False
    return "Stopping audio and clearing the queue."


@tool
@_requires_voice
def skip_audio() -> str:
    """Skip the currently playing song and play the next one in the queue. Requires the music permission tag.
    """
    denied = _check_music()
    if denied:
        return denied
    bong_tools.pending_skip = True
    next_track, _desc = bong_tools.advance_queue()
    if next_track:
        bong_tools.pending_skip_target = next_track
        bong_song_stats._increment_song(Path(next_track).stem)
        return f"Skipping to '{Path(next_track).stem}'."
    bong_tools.pending_skip_target = None
    return "Queue is empty. Add songs to the queue, enable loop, or enable shuffle."


@tool
@_requires_voice
def loop_audio(enabled: bool) -> str:
    """Toggle loop mode. Requires the music permission tag.
    Args:
        enabled: True to enable loop, False to disable.
    """
    denied = _check_music()
    if denied:
        return denied
    if not enabled:
        bong_tools.loop_enabled = False
        bong_tools.loop_track = None
        bong_tools.queue_snapshot = []
        return "Loop disabled."
    bong_tools.loop_enabled = True
    if (bong_tools.current_track or bong_tools.pending_play_audio) and (bong_tools.queue_snapshot or bong_tools.song_queue):
        songs = []
        if bong_tools.current_track:
            songs.append(bong_tools.current_track)
        elif bong_tools.pending_play_audio:
            songs.append(bong_tools.pending_play_audio)
        songs.extend(bong_tools.song_queue)
        bong_tools.queue_snapshot = list(songs)
        bong_tools.loop_track = None
        return f"Looping the queue ({len(songs)} song(s))."
    if bong_tools.current_track or bong_tools.pending_play_audio:
        bong_tools.loop_track = bong_tools.current_track or bong_tools.pending_play_audio
        bong_tools.queue_snapshot = []
        if bong_tools.loop_track:
            return f"Looping '{Path(bong_tools.loop_track).stem}'."
        return "Looping the current song (will bind when playback starts)."
    bong_tools.loop_enabled = False
    return "Nothing is playing. Play a song first before enabling loop."


@tool
@_requires_voice
def music_shuffle_enabled(enabled: bool) -> str:
    """Enable or disable shuffle mode. Requires the music permission tag.
    Args:
        enabled: True to enable shuffle, False to disable shuffle.
    """
    denied = _check_music()
    if denied:
        return denied
    bong_tools.shuffle_enabled = enabled
    state = "enabled" if enabled else "disabled"
    return f"Shuffle mode is now {state}."


@tool
def queue() -> str:
    """Show the current song queue. Requires the music permission tag."""
    denied = _check_music()
    if denied:
        return denied
    lines = []
    if bong_tools.current_track:
        lines.append(f"Now playing: {Path(bong_tools.current_track).stem}")
    elif bong_tools.pending_play_audio:
        lines.append(f"Now playing: {Path(bong_tools.pending_play_audio).stem}")
    else:
        lines.append("Nothing is currently playing.")
    if bong_tools.song_queue:
        for i, path in enumerate(bong_tools.song_queue, 1):
            lines.append(f"  {i}. {Path(path).stem}")
    else:
        lines.append("Queue is empty.")
    state_parts = []
    if bong_tools.loop_enabled and bong_tools.queue_snapshot:
        state_parts.append(f"loop: queue ({len(bong_tools.queue_snapshot)} songs)")
    elif bong_tools.loop_enabled and bong_tools.loop_track:
        state_parts.append("loop: track")
    elif bong_tools.loop_enabled:
        state_parts.append("loop: on")
    if bong_tools.shuffle_enabled:
        state_parts.append("shuffle on")
    if state_parts:
        lines.append(f"({', '.join(state_parts)})")
    return "\n".join(lines)


@tool
@_requires_voice
def clear_queue() -> str:
    """Clear all songs from the queue without stopping the currently playing song. Requires the music permission tag.
    """
    denied = _check_music()
    if denied:
        return denied
    count = len(bong_tools.song_queue)
    bong_tools.song_queue.clear()
    if bong_tools.loop_enabled and bong_tools.queue_snapshot:
        bong_tools.loop_enabled = False
        bong_tools.queue_snapshot = []
    if count:
        return f"Cleared {count} song(s) from the queue."
    return "The queue is already empty."


tools = [download_music, list_music, search_music, play_audio, loop_audio, pause_audio,
        resume_audio, stop_audio, skip_audio, music_shuffle_enabled, queue, clear_queue]