# bong.py — Bong Discord bot cog: handles message events, LLM invocation, and tool dispatch

import discord
import asyncio
import base64
import random
import re
from datetime import datetime
from pathlib import Path

from discord.ext import commands

from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage

import bong_tools
import debug

# Classifier model for determining if user is talking to Bong
classifier_model = ChatOllama(model="gemma3:12b-cloud", temperature=0.1, num_predict=50, keep_alive=-1)
# Vision model for image descriptions
description_model = ChatOllama(model="gemma3:12b-cloud", temperature=0.3, num_predict=800, keep_alive=-1)

# Base model without tools bound (used for re-invocation after tool calls)
base_model = ChatOllama(model="glm-5.1:cloud", temperature=0.5, num_predict=2000, repeat_penalty=1.6, keep_alive=-1)
# Model with tools bound for the initial invocation
model = base_model.bind_tools(bong_tools.tools)

# Maximum number of messages kept per channel in chat history
MAX_MEMORY_SIZE = 30
# Per-channel chat history: channel_id -> list of history entry strings
chat_memories = {}
# Set of channel IDs where Bong is currently active
active_channels = set()

# Discord user IDs allowed to use restricted commands (e.g. shutdown, toggle)
ALLOWED_USERS = {
    273761843544064000,  #Eve
    773961674314219530,  #Radon
    694228585371926572   #Erich
}

# Channels to preload chat history from on startup
DEBUG_CHANNEL_IDS = [698924302594211883]

# Load the response prompt templates from files
TEMPLATE_DIR = Path(__file__).parent / "Response Templates"
prompt_template = (TEMPLATE_DIR / "Bong.txt").read_text(encoding="utf-8")
classifier_template = (TEMPLATE_DIR / "Spoken_To_Classifier.txt").read_text(encoding="utf-8")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".webm", ".mkv")
TEXT_EXTS = (".txt", ".md", ".py", ".json", ".csv", ".xml", ".yaml", ".yml", ".cfg", ".ini", ".log", ".toml", ".rs", ".js", ".ts", ".html", ".css", ".sh", ".bat")

MUSIC_TOOLS = {"list_music", "download_music", "play_audio", "skip_audio", "loop_audio", "music_shuffle_enabled"}
IMAGE_TOOLS = {"list_images", "send_image"}
TEXT_TOOLS = {"list_texts", "send_text"}

MAX_TOOL_ITERATIONS = 10


async def is_talking_to_bong(message_content: str, recent_messages: list, reply_context: str = "", bot_display_name: str = "Bong") -> bool:
    """Check if the user is talking to Bong — fast path for obvious mentions, LLM for ambiguous cases."""
    if "bong" in message_content.lower() or "<@698627881760456724>" in message_content:
        return True

    tagged_name = f"{bot_display_name} (Bong)"
    recent_context = "\n".join(
        msg.replace(bot_display_name, tagged_name) if bot_display_name in msg else msg
        for msg in recent_messages[-7:]
    ) if recent_messages else "No recent messages"

    actual_reply_context = reply_context if reply_context else "None"

    prompt = classifier_template.replace("{recent_context}", recent_context).replace("{message_content}", message_content).replace("{reply_context}", actual_reply_context)

    try:
        response = await asyncio.to_thread(classifier_model.invoke, [HumanMessage(content=prompt)])
        response_text = response.content.strip().upper() if response.content else "NO"
        return "YES" in response_text
    except Exception:
        return False


async def process_attachments(message):
    """Categorize and read message attachments. Returns (attachment_desc, image_attachments, text_attachments)."""
    attachment_parts = []
    image_attachments = []
    text_attachments = []

    for a in message.attachments:
        is_image = (a.content_type and a.content_type.startswith("image/")) or (a.filename and any(a.filename.lower().endswith(ext) for ext in IMAGE_EXTS))
        is_video = (a.content_type and a.content_type.startswith("video/")) or (a.filename and any(a.filename.lower().endswith(ext) for ext in VIDEO_EXTS))
        is_text = (a.content_type and a.content_type.startswith("text/")) or (a.filename and any(a.filename.lower().endswith(ext) for ext in TEXT_EXTS))

        if is_image:
            size_info = f" ({a.width}x{a.height})" if a.width and a.height else ""
            attachment_parts.append(f"[Image {len(image_attachments)}: {a.filename}{size_info}]")
            try:
                img_bytes = await a.read()
                image_attachments.append({
                    "filename": a.filename,
                    "content_type": a.content_type or "image/png",
                    "base64": base64.b64encode(img_bytes).decode("utf-8"),
                })
            except Exception as e:
                debug.log("AI", f"Failed to read image {a.filename}: {e}")
        elif is_video:
            attachment_parts.append(f"[Video: {a.filename}]")
        elif is_text:
            attachment_parts.append(f"[Text file {len(text_attachments)}: {a.filename}]")
            try:
                text_bytes = await a.read()
                text_content = text_bytes.decode("utf-8", errors="replace")
                text_attachments.append({
                    "filename": a.filename,
                    "content": text_content,
                })
            except Exception as e:
                debug.log("AI", f"Failed to read text file {a.filename}: {e}")

    attachment_desc = ", ".join(attachment_parts) if attachment_parts else "None"
    return attachment_desc, image_attachments, text_attachments


def build_voice_status(guild):
    """Build the voice status string and return (vc, voice_status)."""
    if guild:
        vc = guild.voice_client
        if vc and vc.is_connected():
            voice_status = f"Connected to voice channel '{vc.channel.name}'. Do NOT use join_voice — you are already in voice."
            if bong_tools.current_track and (vc.is_playing() or vc.is_paused()):
                voice_status += f"\nCurrently playing '{Path(bong_tools.current_track).stem}'."
        else:
            voice_status = "Not in any voice channel."
    else:
        vc = None
        voice_status = "DM — no voice channels available."
    return vc, voice_status


def build_system_prompt(message, history, voice_status, attachment_desc, image_attachments, text_attachments, replied, replied_to):
    """Fill in the prompt template with all context variables and return the system message string."""
    history_str = "\n".join(history)
    user_msg = message.content.replace("\n", ". ")
    memories_str = bong_tools.retrieve_memories(f"{message.author.display_name}: {user_msg}", username=message.author.display_name, user_id=message.author.id)

    replied_content = replied_to.content.replace("\n", ". ") if replied else ""
    reply_block = ""
    if replied:
        reply_block = f"\nThe user is replying to a message. Take into account what they're replying to when responding.\n\nReplied-to user ID: {replied_to.author.id}\nThe user replied to: {replied_to.author.display_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {replied_content}\n"

    attachment_block = ""
    if attachment_desc != "None":
        attachment_block = f"\nAttachments included with this message: {attachment_desc}\n"
    if image_attachments:
        attachment_block += "\nYou have image attachments on this message. ALWAYS call describe_image to look at them before responding. Never ignore an image.\n"
    if text_attachments:
        attachment_block += "\nYou have text file attachments on this message. ALWAYS call read_text_file to read them before responding. Never ignore a text file.\n"

    return prompt_template.replace("{username}", message.author.display_name).replace("{userID}", str(message.author.id)).replace("{message}", user_msg).replace("{voice_status}", voice_status).replace("{attachments}", attachment_block).replace("{history}", history_str).replace("{reply_context}", reply_block).replace("{memories}", memories_str)


async def update_voice_state(guild, author_id):
    """Set bong_tools.voice_connected and caller_in_voice based on current guild state."""
    bong_tools.voice_connected = (guild is not None and guild.voice_client is not None and guild.voice_client.is_connected())
    if guild:
        member = guild.get_member(author_id)
        if member and member.voice and member.voice.channel:
            bong_tools.caller_in_voice = True
        else:
            member = await guild.fetch_member(author_id)
            bong_tools.caller_in_voice = bool(member and member.voice and member.voice.channel)
    else:
        bong_tools.caller_in_voice = False


def inject_library_context(messages, tool_calls):
    """If any tool calls reference music/image/text libraries, inject the current library listing into messages."""
    names = {tc["name"] for tc in tool_calls}
    if names & MUSIC_TOOLS:
        bong_tools.refresh_music_library()
        library_lines = "\n".join(f"{i}: {f.stem}" for i, f in enumerate(bong_tools.music_library))
        messages.append(HumanMessage(content=f"[Music Library]\n{library_lines}"))
    if names & IMAGE_TOOLS:
        bong_tools.refresh_image_library()
        image_lines = "\n".join(f"{i}: {f.stem}" for i, f in enumerate(bong_tools.image_library))
        messages.append(HumanMessage(content=f"[Image Library]\n{image_lines}"))
    if names & TEXT_TOOLS:
        bong_tools.refresh_text_library()
        text_lines = "\n".join(f"{i}: {f.name}" for i, f in enumerate(bong_tools.text_library))
        messages.append(HumanMessage(content=f"[Text File Library]\n{text_lines}"))


async def _handle_describe_image(tool_args, image_attachments):
    """Execute the describe_image tool inline (needs async access to vision model and attachments)."""
    idx = tool_args.get("index", 0)
    question = tool_args.get("question", "Briefly describe this image in 1-2 sentences. Be concise.")
    if not image_attachments:
        return "No image attachments found on this message."
    if idx < 0 or idx >= len(image_attachments):
        return f"Image index {idx} out of range. There are {len(image_attachments)} image(s) attached (indexes 0-{len(image_attachments)-1})."
    img = image_attachments[idx]
    try:
        label_msg = HumanMessage(content=[
            {"type": "text", "text": "Describe this image in exactly 5 words as a short label suitable for a filename. Use only letters, numbers, and underscores instead of spaces. Examples: Dog_wearing_small_hat_meme, Orange_cat_sleeping_on_couch, Beautiful_sunset_over_the_ocean. Reply with ONLY the label, nothing else."},
            {"type": "image_url", "image_url": f"data:{img['content_type']};base64,{img['base64']}"},
        ])
        label_response = await asyncio.to_thread(description_model.invoke, [label_msg])
        raw_label = label_response.content.strip() if label_response.content else "image"
        label = re.sub(r'[^\w]', '', raw_label.replace(" ", "_"))[:50] or "image"

        save_dir = Path(__file__).parent / "saved_images"
        save_dir.mkdir(exist_ok=True)
        ext = Path(img['filename']).suffix or ".png"
        save_path = save_dir / f"{label}{ext}"
        save_path.write_bytes(base64.b64decode(img["base64"]))
        debug.log("AI", f"Saved image to {save_path}")

        vision_msg = HumanMessage(content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": f"data:{img['content_type']};base64,{img['base64']}"},
        ])
        vision_response = await asyncio.to_thread(description_model.invoke, [vision_msg])
        vision_text = vision_response.content.strip() if vision_response.content else "Could not describe image"
        return f"Description of '{img['filename']}': {vision_text}"
    except Exception as e:
        debug.log("AI", f"Vision error for {img['filename']}: {e}")
        return f"Vision model error for '{img['filename']}': {e}"


async def _handle_read_text_file(tool_args, text_attachments):
    """Execute the read_text_file tool inline (needs access to text attachment contents)."""
    idx = tool_args.get("index", 0)
    if not text_attachments:
        return "No text file attachments found on this message."
    if idx < 0 or idx >= len(text_attachments):
        return f"Text file index {idx} out of range. There are {len(text_attachments)} text file(s) attached (indexes 0-{len(text_attachments)-1})."
    txt = text_attachments[idx]
    try:
        save_dir = Path(__file__).parent / "saved_texts"
        save_dir.mkdir(exist_ok=True)
        ext = Path(txt['filename']).suffix or ".txt"
        clean_name = re.sub(r'[^\w]', '_', Path(txt['filename']).stem)[:50] or "text_file"
        save_path = save_dir / f"{clean_name}{ext}"
        save_path.write_text(txt['content'], encoding="utf-8")
        debug.log("AI", f"Saved text file to {save_path}")
        content = txt['content']
        if len(content) > 8000:
            content = content[:8000] + f"\n\n... [truncated, {len(txt['content'])} total characters]"
        return f"Contents of '{txt['filename']}':\n{content}"
    except Exception as e:
        debug.log("AI", f"Text file error for {txt['filename']}: {e}")
        return f"Error reading text file '{txt['filename']}': {e}"


async def dispatch_tool(tool_name, tool_args, image_attachments, text_attachments):
    """Execute a single tool call, returning the string result.
    Handles describe_image and read_text_file inline; everything else goes through bong_tools.tool_map.
    """
    if tool_name == "describe_image":
        result = await _handle_describe_image(tool_args, image_attachments)
        debug.log("AI", f"Tool result: {result}")
        return result
    if tool_name == "read_text_file":
        result = await _handle_read_text_file(tool_args, text_attachments)
        debug.log("AI", f"Tool result: {result}")
        return result
    if tool_name not in bong_tools.tool_map:
        debug.log("AI", f"Unknown tool: {tool_name}")
        return f"Error: '{tool_name}' is not a valid tool name. You may have put the arguments inside the tool name by mistake. Call the tool again with just the tool name and the arguments as separate parameters. Available tools: {', '.join(bong_tools.tool_map.keys())}"
    try:
        result = await asyncio.to_thread(bong_tools.tool_map[tool_name].invoke, tool_args)
        debug.log("AI", f"Tool result: {result}")
        return result
    except Exception as e:
        debug.log("AI", f"Tool error: {e}")
        return f"Error: {e}"


def _extract_response_text(response):
    """Extract plain text from an LLM response (content may be a list of chunks or a string)."""
    if isinstance(response.content, list):
        return "".join(chunk.text if hasattr(chunk, "text") else str(chunk) for chunk in response.content)
    return response.content or ""


async def run_tool_loop(bound_model, messages, image_attachments, text_attachments, last_prompt_path):
    """Run the LLM invocation + tool-call loop. Returns (result_text, tool_summaries)."""
    ai_response = await asyncio.to_thread(bound_model.invoke, messages)
    messages.append(ai_response)

    iteration = 0
    tool_summaries = []

    while ai_response.tool_calls:
        iteration += 1
        if iteration > MAX_TOOL_ITERATIONS:
            messages.append(HumanMessage(content="You have exceeded the maximum number of tool calls. Please respond to the user now without making any more tool calls."))
            ai_response = await asyncio.to_thread(bound_model.invoke, messages)
            messages.append(ai_response)
            break

        inject_library_context(messages, ai_response.tool_calls)

        for tc in ai_response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            debug.log("AI", f"Tool call: {tool_name}({tool_args})")
            debug.log_to_file("AI", f"TOOL CALL: {tool_name}({tool_args})")
            last_prompt_path.write_text(
                last_prompt_path.read_text(encoding="utf-8") + f"\nTOOL CALL: {tool_name}({tool_args})\n",
                encoding="utf-8",
            )

            tool_result = await dispatch_tool(tool_name, tool_args, image_attachments, text_attachments)
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tc["id"]))

            summary_args = ', '.join(str(v) for v in tool_args.values())
            tool_summaries.append(f"{tool_name}({summary_args})")
            debug.log_to_file("AI", f"TOOL RESULT ({tool_name}): {tool_result}")
            last_prompt_path.write_text(
                last_prompt_path.read_text(encoding="utf-8") + f"TOOL RESULT ({tool_name}): {tool_result}\n",
                encoding="utf-8",
            )

        ai_response = await asyncio.to_thread(bound_model.invoke, messages)
        messages.append(ai_response)

    result = _extract_response_text(ai_response)

    if not result.strip():
        debug.log("AI", "Empty response, retrying with thinking model")
        retry_response = await asyncio.to_thread(base_model.invoke, messages)
        result = _extract_response_text(retry_response)

    debug.log("AI", "Generating final response")
    last_prompt_path.write_text(
        last_prompt_path.read_text(encoding="utf-8") + f"\nFINAL RESPONSE:\n{result}\n",
        encoding="utf-8",
    )

    return result, tool_summaries


def _make_after_play_callback(guild):
    """Create a closure that auto-continues playback after a track finishes (loop/shuffle)."""
    def after_play(err):
        if err:
            debug.log("Audio", f"Playback error: {err}")
        vc = guild.voice_client
        if not vc or not vc.is_connected():
            return
        try:
            if vc.is_playing() or vc.is_paused():
                return
        except Exception:
            return
        if bong_tools.pending_play_audio or bong_tools.pending_skip or bong_tools.pending_stop:
            return
        try:
            if bong_tools.loop_enabled and bong_tools.current_track:
                vc.play(discord.FFmpegPCMAudio(bong_tools.current_track, options="-filter:a volume=0.3"), after=after_play)
                return
            if bong_tools.shuffle_enabled:
                files = list(bong_tools.DOWNLOAD_DIR.glob("*.mp3"))
                if files:
                    next_track = random.choice(files)
                    bong_tools.current_track = str(next_track)
                    vc.play(discord.FFmpegPCMAudio(bong_tools.current_track, options="-filter:a volume=0.3"), after=after_play)
                    return
            bong_tools.current_track = None
        except Exception as e:
            debug.log("Audio", f"Auto-continue error: {e}")
    return after_play


async def _dispatch_join_voice(guild):
    """Handle pending_join_voice flag. Returns error string or None."""
    if not bong_tools.pending_join_voice:
        return None
    error = None
    if guild:
        target_member = guild.get_member(bong_tools.pending_join_voice)
        if target_member and target_member.voice and target_member.voice.channel:
            try:
                await target_member.voice.channel.connect()
                debug.log("AI", f"Joined voice channel: {target_member.voice.channel.name}")
            except Exception as e:
                debug.log("AI", f"Failed to join voice channel: {e}")
                error = f"Failed to join voice channel: {e}"
        else:
            debug.log("AI", "Target user is not in a voice channel")
            error = "Target user is not in a voice channel"
    else:
        error = "Cannot join voice channels in DMs."
    bong_tools.pending_join_voice = None
    return error


async def _dispatch_leave_voice(guild):
    """Handle pending_leave_voice flag. Returns error string or None."""
    if not bong_tools.pending_leave_voice:
        return None
    error = None
    if guild and guild.voice_client:
        try:
            await guild.voice_client.disconnect()
            debug.log("AI", "Left voice channel")
        except Exception as e:
            debug.log("AI", f"Failed to leave voice channel: {e}")
            error = f"Failed to leave voice channel: {e}"
    elif guild:
        debug.log("AI", "Not in a voice channel")
        error = "Not in a voice channel"
    else:
        error = "Cannot leave voice channels in DMs."
    bong_tools.pending_leave_voice = None
    return error


async def _dispatch_play_audio(guild):
    """Handle pending_play_audio flag. Returns error string or None."""
    if not bong_tools.pending_play_audio:
        return None
    vc = guild.voice_client if guild else None
    if not vc or not vc.is_connected():
        bong_tools.pending_play_audio = None
        return "Not in a voice channel, can't play audio. Join a voice channel first."
    try:
        track_path = bong_tools.pending_play_audio
        bong_tools.current_track = track_path
        after_play = _make_after_play_callback(guild)
        if vc.is_playing() or vc.is_paused():
            vc.stop()
            await asyncio.sleep(0.5)
        source = discord.FFmpegPCMAudio(track_path, options="-filter:a volume=0.3")
        vc.play(source, after=after_play)
    except Exception as e:
        debug.log("AI", f"Failed to play audio: {e}")
        return f"Failed to play audio: {e}"
    finally:
        bong_tools.pending_play_audio = None
    return None


async def _dispatch_loop_audio(guild):
    """Handle loop track start by stopping current playback and queuing the loop track."""
    if not (bong_tools.loop_enabled and bong_tools.loop_track):
        return None
    vc = guild.voice_client if guild else None
    if not vc or not vc.is_connected():
        bong_tools.loop_track = None
        return None
    if vc.is_playing() or vc.is_paused():
        vc.stop()
    bong_tools.current_track = bong_tools.loop_track
    bong_tools.pending_play_audio = bong_tools.loop_track
    bong_tools.loop_track = None
    return None


async def _dispatch_pause_audio(guild):
    """Handle pending_pause flag. Returns error string or None."""
    if not bong_tools.pending_pause:
        return None
    vc = guild.voice_client if guild else None
    error = None
    if vc and vc.is_playing():
        vc.pause()
    else:
        error = "Nothing is playing to pause."
    bong_tools.pending_pause = False
    return error


async def _dispatch_resume_audio(guild):
    """Handle pending_resume flag. Returns error string or None."""
    if not bong_tools.pending_resume:
        return None
    vc = guild.voice_client if guild else None
    error = None
    if vc and vc.is_paused():
        vc.resume()
    else:
        error = "Nothing is paused to resume."
    bong_tools.pending_resume = False
    return error


async def _dispatch_stop_audio(guild):
    """Handle pending_stop flag. Returns error string or None."""
    if not bong_tools.pending_stop:
        return None
    bong_tools.loop_enabled = False
    bong_tools.loop_track = None
    bong_tools.current_track = None
    vc = guild.voice_client if guild else None
    error = None
    if vc and (vc.is_playing() or vc.is_paused()):
        vc.stop()
    else:
        error = "Nothing is playing to stop."
    bong_tools.pending_stop = False
    return error


async def _dispatch_skip_audio(guild):
    """Handle pending_skip flag by stopping current playback and queuing the next track."""
    if not bong_tools.pending_skip:
        return None
    vc = guild.voice_client if guild else None
    if not vc or not (vc.is_playing() or vc.is_paused()):
        bong_tools.pending_skip = False
        bong_tools.pending_skip_target = None
        bong_tools.pending_skip_info = ""
        return "Nothing is playing to skip."
    if not bong_tools.pending_skip_target:
        bong_tools.pending_skip = False
        bong_tools.pending_skip_info = ""
        return "No music files to skip to."
    bong_tools.current_track = str(bong_tools.pending_skip_target)
    bong_tools.pending_play_audio = str(bong_tools.pending_skip_target)
    vc.stop()
    bong_tools.pending_skip = False
    bong_tools.pending_skip_target = None
    bong_tools.pending_skip_info = ""
    return None


async def dispatch_voice_actions(guild, message):
    """Dispatch all pending voice/audio/file-send actions. Returns first error or None."""
    try:
        voice_dispatchers = [
            _dispatch_join_voice,
            _dispatch_leave_voice,
            _dispatch_stop_audio,
            _dispatch_skip_audio,
            _dispatch_loop_audio,
            _dispatch_play_audio,
            _dispatch_pause_audio,
            _dispatch_resume_audio,
        ]
        results = []
        for fn in voice_dispatchers:
            results.append(await fn(guild))

        if bong_tools.pending_send_image:
            img_path = Path(bong_tools.pending_send_image)
            if img_path.exists():
                try:
                    await message.channel.send(file=discord.File(str(img_path)))
                except Exception as e:
                    debug.log("AI", f"Failed to send image: {e}")
            else:
                debug.log("AI", f"Image not found: {img_path}")
            bong_tools.pending_send_image = None

        if bong_tools.pending_send_text:
            txt_path = Path(bong_tools.pending_send_text)
            if txt_path.exists():
                try:
                    await message.channel.send(file=discord.File(str(txt_path)))
                except Exception as e:
                    debug.log("AI", f"Failed to send text file: {e}")
            else:
                debug.log("AI", f"Text file not found: {txt_path}")
            bong_tools.pending_send_text = None

        for error in results:
            if error:
                return error
        return None

    except Exception as e:
        debug.log("AI", f"Error during voice/audio dispatch: {e}")
        debug.log_to_file("AI", f"Error during voice/audio dispatch: {e}")
        if guild and guild.voice_client:
            try:
                await guild.voice_client.disconnect(force=True)
                debug.log("AI", "Force-disconnected voice client to reset Opus state")
            except Exception:
                pass
        return f"An error occurred: {e}"


async def apply_reactions(message):
    """Apply any pending emoji reactions and clear the queue."""
    for emoji in bong_tools.pending_reactions:
        try:
            await message.add_reaction(emoji)
        except Exception as e:
            debug.log("AI", f"Failed to react with {emoji}: {e}")
    bong_tools.pending_reactions.clear()


def record_history(history, message, result, attachment_desc, tool_summaries):
    """Store the exchange in channel history, evicting oldest if at capacity."""
    attachment_suffix = f" {attachment_desc}" if attachment_desc != "None" else ""
    tool_summary_str = f" [Already completed: {'; '.join(tool_summaries)}]" if tool_summaries else ""
    history_entry = f"{message.author.display_name} at {datetime.now().strftime('%H:%M')}: {message.content.replace(chr(10), '. ')}{attachment_suffix}\nBong's response: {result.replace(chr(10), '. ')}{tool_summary_str}"
    if len(history) >= MAX_MEMORY_SIZE:
        history.pop()
    history.insert(0, history_entry)
    debug.log("AI", f"response {len(history)}")
    debug.log_to_file("AI", f"RESPONSE: {result}")


def record_passive_message(history, message, attachment_desc):
    """Store a non-Bong-targeted message in channel history for context."""
    attachment_suffix = f" {attachment_desc}" if attachment_desc != "None" else ""
    history_entry = f"{message.author.display_name} at {datetime.now().strftime('%H:%M')}: {message.content.replace(chr(10), '. ')}{attachment_suffix}"
    history.insert(0, history_entry)
    if len(history) >= MAX_MEMORY_SIZE:
        history.pop()
    debug.log("AI", f"remembered {len(history)}")


class BongCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.bot.loop.create_task(self._preload_channel())
    
    async def _preload_channel(self):
        """Load recent message history from debug channels on startup so Bong has context."""
        await self.bot.wait_until_ready()
        for d_channel in DEBUG_CHANNEL_IDS:
            channel = self.bot.get_channel(d_channel)
            if channel is None:
                dm_user = self.bot.get_user(d_channel)
                if dm_user:
                    channel = await dm_user.create_dm()
                if channel is None:
                    debug.log("AI", f"Preload failed: channel {d_channel} not found")
                    continue

            active_channels.add(d_channel)
            history = chat_memories.setdefault(d_channel, [])
            async for msg in channel.history(limit=MAX_MEMORY_SIZE):
                history.append(f"{msg.author.display_name} at {msg.created_at.strftime('%H:%M')}: {msg.content}")
            channel_name = channel.name if hasattr(channel, "name") else f"DM with {channel.recipient}"
            debug.log("AI", f"Preloaded {len(history)} messages from channel #{channel_name}")

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.bot.user:
            return
        if message.channel.id not in active_channels:
            return
        if message.content.startswith(self.bot.command_prefix):
            after_prefix = message.content[len(self.bot.command_prefix):].strip()
            command_name = after_prefix.split()[0] if after_prefix else ""
            if command_name in self.bot.all_commands:
                return

        attachment_desc, image_attachments, text_attachments = await process_attachments(message)
        history = chat_memories.setdefault(message.channel.id, [])

        recent_messages = history[:7]

        replied_to = ""
        replied = False
        reply_context = ""
        if message.reference is not None:
            replied_to = await message.channel.fetch_message(message.reference.message_id)
            replied = True
            reply_context = f"User is replying to {replied_to.author.display_name}: {replied_to.content.replace(chr(10), '. ')}"

        if not await is_talking_to_bong(message.content.replace(chr(10), '. '), recent_messages, reply_context, self.bot.user.display_name):
            record_passive_message(history, message, attachment_desc)
            return

        try:
            guild = message.guild
            _, voice_status = build_voice_status(guild)
            system_msg = build_system_prompt(message, history, voice_status, attachment_desc, image_attachments, text_attachments, replied, replied_to)

            messages = [HumanMessage(content=system_msg)]

            last_prompt_path = Path(__file__).parent / "logs" / "last_prompt.log"
            last_prompt_path.write_text(system_msg + "\n\n========\n", encoding="utf-8")

            debug.log_to_file("AI", f"QUERY from {message.author.display_name} ({message.author.id}): {message.content}")

            await update_voice_state(guild, message.author.id)
            bong_tools.authorized = message.author.id in ALLOWED_USERS
            bong_tools.current_user_id = message.author.id

            bound_model = base_model.bind_tools(bong_tools.tools)
            result, tool_summaries = await run_tool_loop(bound_model, messages, image_attachments, text_attachments, last_prompt_path)

            await apply_reactions(message)
            record_history(history, message, result, attachment_desc, tool_summaries)

            voice_error = await dispatch_voice_actions(guild, message)

            async with message.channel.typing():
                if voice_error:
                    messages.append(HumanMessage(content=f"System: The voice/audio action failed with this error: {voice_error}. Please let the user know and suggest what they can do."))
                    error_response = await asyncio.to_thread(model.invoke, messages)
                    error_text = _extract_response_text(error_response)
                    await message.channel.send(error_text)
                else:
                    await message.channel.send(result)

            if bong_tools.pending_shutdown:
                if message.author.id in ALLOWED_USERS:
                    await message.add_reaction("🫡")
                    await self.bot.close()
                else:
                    debug.log("AI", "Unauthorized shutdown attempt")
                bong_tools.pending_shutdown = False

        except Exception as e:
            error_message = f"Error generating response: {e}"
            await message.channel.send(error_message)
            history.insert(0, error_message)

    @commands.command(name="llm", help="Toggle Bong's activity in the current channel")
    async def llm(self, ctx):
        """Toggle Bong's activity in the current channel. If active, it will respond
        to messages containing 'bong'. If inactive, it will stop responding."""
        if ctx.author.id not in ALLOWED_USERS:
            await ctx.send("You are not authorized to use this command.")
            return
        if ctx.channel.id in active_channels:
            active_channels.discard(ctx.channel.id)
            await ctx.send(f"No more Bonging in this channel! (ID: {ctx.channel.id})")
        else:
            active_channels.add(ctx.channel.id)
            history = chat_memories.setdefault(ctx.channel.id, [])
            async for msg in ctx.channel.history(limit=MAX_MEMORY_SIZE):
                history.append(f"{msg.author.display_name} at {msg.created_at.strftime('%H:%M')}: {msg.content}")
            await ctx.send(f"Bong is now active in this channel! (ID: {ctx.channel.id})")
            debug.log("AI", history)

async def setup(bot):
    """Entry point for discord.py to load this cog."""
    await bot.add_cog(BongCog(bot))