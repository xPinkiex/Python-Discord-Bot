import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.tools import tool

import reminders
import user_data
import bong_tools
import bong_memory_helpers
import bong_song_stats


def _format_utc_offset(offset: float) -> str:
    sign = "+" if offset >= 0 else "-"
    hours = int(abs(offset))
    minutes = int((abs(offset) - hours) * 60)
    if minutes:
        return f"UTC{sign}{hours}:{minutes:02d}"
    return f"UTC{sign}{hours}"


def _check_llm():
    if not user_data.has_permission(bong_tools.current_user_id, "llm"):
        return "You don't have permission to use this feature. Ask an admin to grant you the llm tag."
    return None


def _check_music():
    if not user_data.has_permission(bong_tools.current_user_id, "music"):
        return "You don't have permission to use music commands. Ask an admin to grant you the music tag."
    return None


def _check_vc():
    if not user_data.has_permission(bong_tools.current_user_id, "vc_commands"):
        return "You don't have permission to use voice commands. Ask an admin to grant you the vc_commands tag."
    return None


@tool
def react(emojis: str) -> str:
    """React to the user's message with one or more emojis. Use this to express emotion or acknowledge the message.
    Args:
        emojis: One or more emoji characters to react with, separated by spaces (e.g. ❤️, 👍 😂, 🤔 💡 🎉)
    """
    denied = _check_llm()
    if denied:
        return denied
    for emoji in emojis.split():
        bong_tools.pending_reactions.append(emoji)
    return f"Reacted with {emojis}"


@tool
def join_voice(userID: int) -> str:
    """Join the voice channel that the user is currently in. Use this when the user wants you to join their voice channel.
    Args:
        userID: User ID of the person you want to join voice chat with.
    """
    denied = _check_music()
    if denied:
        return denied
    bong_tools.pending_join_voice = userID
    return "Joining voice channel"


@tool
def leave_voice() -> str:
    """Disconnect from the current voice channel. Use this when the user wants you to leave their voice channel.
    """
    denied = _check_music()
    if denied:
        return denied
    bong_tools.pending_leave_voice = True
    return "Leaving voice channel"


@tool
def list_images() -> str:
    """List all saved images in the saved_images folder. Use this when the user wants to browse or view saved images. The full library is provided in context — use the index numbers to select images with send_image.
    """
    denied = _check_llm()
    if denied:
        return denied
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
    denied = _check_llm()
    if denied:
        return denied
    bong_tools.refresh_image_library()
    files = bong_tools.image_library
    if not files:
        return "No images available."
    if index < 0 or index >= len(files):
        return f"Index {index} out of range. Use list_images to see available images (0-{len(files)-1})."
    bong_tools.pending_send_image = str(files[index])
    return f"Sending '{files[index].stem}'."


@tool
def list_texts() -> str:
    """List all saved text files in the saved_texts folder. Use this when the user wants to browse or view saved text files. The full library is provided in context — use the index numbers to select files with send_text.
    """
    denied = _check_llm()
    if denied:
        return denied
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
    denied = _check_llm()
    if denied:
        return denied
    bong_tools.refresh_text_library()
    files = bong_tools.text_library
    if not files:
        return "No text files available."
    if index < 0 or index >= len(files):
        return f"Index {index} out of range. Use list_texts to see available text files (0-{len(files)-1})."
    bong_tools.pending_send_text = str(files[index])
    return f"Sending '{files[index].stem}'."


@tool
def read_text_file(index: int = 0) -> str:
    """Read the contents of a text file attached to the current message. Use this when the user wants you to read, summarize, or answer questions about an attached text file.
    Args:
        index: The 0-based index of the text file attachment to read (default 0 for the first text file).
    """
    return "Text file request handled by cog."


@tool
def describe_image(index: int = 0, question: str = "Briefly describe this image in 1-2 sentences. Be concise.") -> str:
    """Describe an image attached to the current message using the vision model. Use this when the user wants you to look at or describe an image, read text in an image, or answer questions about an image. The attachment list tells you which images are available.
    Args:
        index: The 0-based index of the image attachment to describe (default 0 for the first image).
        question: What to ask about the image (default: brief description). Use "Read all the text in this image." for OCR, or "Describe this image in detail." for a thorough description.
    """
    return "Vision request handled by cog."


@tool
def current_time() -> str:
    """Get the current time. If the user has a stored timezone, returns their local time and UTC. If not, returns UTC time and suggests setting a timezone.
    """
    utc_now = datetime.now(timezone.utc)
    offset = user_data.get_timezone(bong_tools.current_user_id)
    if offset is not None:
        local_now = utc_now + timedelta(hours=offset)
        tz_str = _format_utc_offset(offset)
        name = bong_tools.current_username or "you"
        return f"Current time for {name} ({tz_str}): {local_now.strftime('%H:%M on %A, %B %d')}\n(For reference, UTC is {utc_now.strftime('%H:%M on %A, %B %d')})"
    return f"UTC time: {utc_now.strftime('%H:%M on %A, %B %d')}\nNo timezone set — use set_timezone if the user mentions their timezone."


@tool
def set_timezone(timezone: str) -> str:
    """Set your timezone so Bong can tell you the time in your local time. Use this when someone mentions their timezone or when you need to know their local time. Supported formats: 'UTC+2', 'GMT-5', 'EST', 'PST', 'CET', 'New York', 'London', 'Tokyo', etc.
    Args:
        timezone: A timezone name or UTC offset (e.g. "UTC+2", "EST", "London", "+5:30", "PST").
    """
    offset = user_data.parse_timezone(timezone)
    if offset is None:
        return f"Could not understand timezone '{timezone}'. Try formats like 'UTC+2', 'EST', 'PST', 'London', or 'GMT-5'."
    user_data.set_timezone(bong_tools.current_user_id, offset)
    return f"Timezone set to {_format_utc_offset(offset)}."


@tool
def get_timezone() -> str:
    """Get the current user's timezone. Returns their UTC offset or says they haven't set one."""
    offset = user_data.get_timezone(bong_tools.current_user_id)
    if offset is None:
        return "No timezone set. Ask the user for their timezone and use set_timezone."
    return _format_utc_offset(offset)


@tool
def set_reminder(message: str, time: str = "", time_delta: str = "") -> str:
    """Set a reminder for the current user. Bong will DM them when the time is up.

    ALWAYS use the 'time' parameter when the user specifies a specific date or time, such as "tomorrow at 3pm", "Friday at noon", "on June 5th", "31.5 at 10:22", etc. Pass the user's exact words into 'time' — do NOT calculate a duration yourself. The system will handle timezone conversion automatically.

    ONLY use 'time_delta' for simple relative durations like "in 30 minutes" or "2 hours from now" when no specific date/time is mentioned.

    The 'time' parameter requires the user to have a timezone set. If they don't, ask them to set one with set_timezone first.
    Args:
        message: What to remind the user about (e.g. "feed the cat", "take out the trash").
        time: A specific date/time in the user's timezone. Pass the user's words directly — e.g. "tomorrow at 3pm", "Friday at 12:00", "31.5.2026 at 10:22", "next monday at 9am". Do NOT calculate durations — just pass the natural language time here. Dates with dots use day.month format (31.5 = May 31st). Dates with slashes default to month/day (5/6 = June 5th).
        time_delta: A relative duration from now — e.g. "30 minutes", "2 hours". ONLY use this when the user says something like "remind me in 30 minutes" with no specific date/time.
    """
    if time:
        utc_offset = user_data.get_timezone(bong_tools.current_user_id)
        if utc_offset is None:
            return "The user hasn't set their timezone yet. Ask them for their timezone and use set_timezone to set it before setting absolute-time reminders. You can still use the time_delta parameter for relative reminders like 'in 2 hours'."
        try:
            ts = reminders.parse_absolute_time(time, utc_offset)
        except reminders.PastDateError as e:
            return f"{e} Please choose a future time."
        if ts is None:
            return f"Could not understand the time '{time}'. Try formats like 'tomorrow at 3pm', 'Friday at 12:00', 'June 5 at 3pm', or 'next monday at 9am'."
        reminders.add_reminder(
            user_id=bong_tools.current_user_id,
            username=bong_tools.current_username or "",
            message=message,
            due_at=ts,
        )
        when_local = datetime.fromtimestamp(ts, tz=timezone(timedelta(hours=utc_offset))).strftime("%H:%M on %Y-%m-%d")
        tz_str = _format_utc_offset(utc_offset)
        return f"Reminder set: '{message}' at {when_local} ({tz_str})."

    if time_delta:
        seconds = reminders.parse_time_delta(time_delta)
        if seconds is None:
            return f"Could not understand the time '{time_delta}'. Use formats like '30 minutes', '2 hours', '1 day'."
        due_at = datetime.now(timezone.utc).timestamp() + seconds
        reminders.add_reminder(
            user_id=bong_tools.current_user_id,
            username=bong_tools.current_username or "",
            message=message,
            due_at=due_at,
        )
        when = reminders._format_delta(seconds)
        return f"Reminder set: '{message}' in {when}."

    return "Please provide either a 'time' (absolute, e.g. 'tomorrow at 3pm') or 'time_delta' (relative, e.g. '2 hours') for when to remind."


@tool
def cancel_reminder(query: str = "") -> str:
    """Cancel a pending reminder. If a query is given, cancels the most recent matching reminder. If no query, cancels the most recent reminder.
    Args:
        query: Part of the reminder message to match (e.g. "cat"). Leave empty to cancel the most recent one.
    """
    return reminders.cancel_reminder(bong_tools.current_user_id, query)


@tool
def list_reminders_tool() -> str:
    """List all pending reminders for the current user. Use this when the user asks what reminders they have set."""
    return reminders.list_reminders(bong_tools.current_user_id)


@tool
def bot_stats() -> str:
    """Get statistics about the bot: uptime, memory count, known users, reminders, top 3 most-played songs, and top 3 users by tokens."""
    lines = []
    now = datetime.now()
    if bong_tools.start_time:
        delta = now - bong_tools.start_time
        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        lines.append(f"Uptime: {days}d {hours}h {minutes}m")
    else:
        lines.append("Uptime: unknown")
    try:
        mem_count = bong_memory_helpers.memory_count()
        lines.append(f"Memories: {mem_count}")
    except Exception:
        lines.append("Memories: unavailable")
    known = user_data.known_user_count()
    lines.append(f"Known users: {known}")
    reminder_count = len(reminders.reminders) if reminders.reminders else 0
    lines.append(f"Pending reminders: {reminder_count}")
    total_plays = bong_song_stats.get_total_plays()
    lines.append(f"Total song plays: {total_plays}")
    top = bong_song_stats.get_top_songs(3)
    if top:
        top_lines = [f"  {i+1}. {name} ({count} plays)" for i, (name, count) in enumerate(top)]
        lines.append("Top songs:\n" + "\n".join(top_lines))
    else:
        lines.append("Top songs: none yet")
    total_tokens = sum(d.get("tokens", 0) for d in user_data._user_data.values())
    lines.append(f"Total tokens used: {total_tokens:,}")
    top_users = user_data.get_top_users_by_tokens(3)
    if top_users:
        top_user_lines = [f"  {i+1}. {name or uid} ({count:,} tokens)" for i, (uid, name, count) in enumerate(top_users)]
        lines.append("Top users:\n" + "\n".join(top_user_lines))
    else:
        lines.append("Top users: none yet")
    return "\n".join(lines)


@tool
def shutdown() -> str:
    """Shut down the bot. Only an admin user can request this.
    """
    if not user_data.is_admin(bong_tools.current_user_id):
        return "Cannot shut down: you don't have permission (requires admin tag)."
    bong_tools.pending_shutdown = True
    return "Shutting down"


@tool
def start_listening(userID: int) -> str:
    """Start listening for voice commands in the voice channel. The bot will listen for the wake word 'hey bong' and process voice commands from users with the vc_commands permission.
    Args:
        userID: The Discord user ID of the person who wants to start voice commands (used to confirm authorization).
    """
    denied = _check_vc()
    if denied:
        return denied
    if bong_tools.pending_start_listening is not None:
        return "Voice command listener is already being started. Do not call this tool again."
    bong_tools.pending_start_listening = userID
    return "Starting voice command listener"


@tool
def stop_listening() -> str:
    """Stop listening for voice commands in the voice channel. Use this ONLY when the user wants to disable voice command mode — the bot will stop detecting wake words and processing voice input. Do NOT use this to stop music playback or disconnect from voice.
    """
    denied = _check_vc()
    if denied:
        return denied
    bong_tools.pending_stop_listening = True
    return "Stopping voice command listener"


tools = [react, describe_image, read_text_file, join_voice, leave_voice, current_time,
         list_images, send_image, list_texts, send_text,
         set_reminder, cancel_reminder, list_reminders_tool, set_timezone, get_timezone,
         bot_stats, shutdown, start_listening, stop_listening]