# bong.py — Bong Discord bot cog: handles message events, LLM invocation, and tool dispatch
#
# This is the main "brain" of the bot. It:
#   1. Listens for messages in active channels
#   2. Uses a classifier LLM to decide if the user is talking to Bong
#   3. Builds a system prompt with context (history, voice status, memories, attachments)
#   4. Runs the main LLM in a tool-call loop (up to 10 iterations)
#   5. Dispatches async actions (voice, reactions, file sends) that the sync tools queued
#   6. Records the exchange in channel history for future context

import discord
import asyncio
import base64

import re
import time
from datetime import datetime
from pathlib import Path
import httpx
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BONG_DATA = PROJECT_ROOT / "bong_data"
BONG_USER_DATA = PROJECT_ROOT / "bong_user_data"

# discord.py's command framework — this file is loaded as a "cog" (plugin)
from discord.ext import commands

# LangChain LLM wrappers for Ollama-hosted models
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage

import bong_tools
import bong_song_stats
import bong_memory_helpers
import debug
import dm_approval
import persist
import reminders
import user_data
import voice_commands
import bong_e621

# --- LLM Models ---
# The classifier model is fast and cheap — it only needs to output YES/NO
classifier_model = ChatOllama(model="gemma3:12b-cloud", temperature=0.1, num_predict=50, keep_alive=-1)
# The vision model describes images and generates filenames for saved images
description_model = ChatOllama(model="gemma3:12b-cloud", temperature=0.3, num_predict=800, keep_alive=-1)

# The main conversation model — used for generating Bong's responses
# base_model is without tools (used for retrying empty responses)
base_model = ChatOllama(model="glm-5.1:cloud", temperature=0.5, num_predict=2000, repeat_penalty=1.6, keep_alive=-1)
# model has tools bound — used for retrying after voice errors
model = base_model.bind_tools(bong_tools.tools)

# --- Chat history settings ---
# Maximum number of messages kept per channel in the rolling history
MAX_MEMORY_SIZE = 30
# Per-channel chat history: channel_id -> list of history entry strings (newest first)
chat_memories = {}
# Per-channel conversation summaries (in-memory, not persisted)
channel_summaries: dict[int, list[str]] = {}
MAX_SUMMARIES_PER_CHANNEL = 5
# How many oldest messages to summarize when we hit the threshold
SUMMARIZE_CHUNK_SIZE = 10
# Set of channel IDs where Bong is currently active (toggled with the @llm command)
active_channels = set()
# Per-user cooldown for voice commands (user_id -> timestamp of last processed command)
_voice_cooldowns: dict[int, float] = {}
VOICE_COOLDOWN_SECONDS = 5
# Channels with an active background summarization task (prevents double-summarization)
_summarization_in_progress: set[int] = set()

# Permission tags are managed in user_data (users.json + OWNER_ID)
# Tags: llm (chat), llm_fast (no cooldown), music, vc_commands, e621, admin (implies all)
# - admin:    full access (@llm, @tags, @reload, @poweroff, shutdown)
# - llm:      talk to Bong, use chat-tier tools (memories, images, texts, web, reminders, timezone, stats, react)
# - llm_fast:  same as llm + no 60s cooldown
# - music:    all music tools + join/leave voice
# - vc_commands: voice command wake word + start/stop listening
# - e621:     e621 search, subscribe, unsubscribe, DM notifications

# Channels whose history is automatically loaded on startup so Bong has context immediately
DEBUG_CHANNEL_IDS = [ 698924302594211883, 698633099591942199 ]

# Load the system prompt and classifier prompt from template files
TEMPLATE_DIR = BONG_DATA / "Response Templates"
prompt_template = (TEMPLATE_DIR / "Bong.txt").read_text(encoding="utf-8")
classifier_template = (TEMPLATE_DIR / "Spoken_To_Classifier.txt").read_text(encoding="utf-8")

# Recognized file extensions for categorizing attachments
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".webm", ".mkv")
TEXT_EXTS = (".txt", ".md", ".py", ".json", ".csv", ".xml", ".yaml", ".yml", ".cfg", ".ini", ".log", ".toml", ".rs", ".js", ".ts", ".html", ".css", ".sh", ".bat")

# Tool names that need the music/image/text library injected as context when called
MUSIC_TOOLS = {"list_music", "download_music", "play_audio", "skip_audio", "loop_audio", "music_shuffle_enabled"}
IMAGE_TOOLS = {"list_images", "send_image"}
TEXT_TOOLS = {"list_texts", "send_text"}

# Maximum number of tool-call iterations before forcing a final response
MAX_TOOL_ITERATIONS = 10
# LLM retry settings
LLM_MAX_RETRIES = 2
LLM_RETRY_DELAYS = [2, 4]  # seconds between retries


# Exceptions that indicate a transient network/timeout issue worth retrying
RETRYABLE_EXCEPTIONS = (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError, ConnectionError, OSError)


async def invoke_with_retry(model, messages, max_retries=LLM_MAX_RETRIES):
    """Invoke an LLM model with automatic retries on transient errors.
    
    Retries up to max_retries times with exponential backoff.
    Only retries on network/timeout errors, not on model errors or bad output.
    """
    for attempt in range(max_retries + 1):
        try:
            return await asyncio.to_thread(model.invoke, messages)
        except RETRYABLE_EXCEPTIONS as e:
            if attempt >= max_retries:
                raise
            delay = LLM_RETRY_DELAYS[attempt] if attempt < len(LLM_RETRY_DELAYS) else LLM_RETRY_DELAYS[-1]
            debug.error("AI", f"LLM timeout/connection error (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay}s: {e}")
            await asyncio.sleep(delay)
        except Exception:
            raise


async def is_talking_to_bong(message_content: str, recent_messages: list, reply_context: str = "", bot_display_name: str = "Bong") -> bool:
    """Check if the user is talking to Bong.
    
    Fast path: if the message contains "bong" or a bot mention, return True immediately.
    Slow path: run the classifier LLM to decide for ambiguous messages.
    """
    # Fast path: obvious mentions of the bot
    if "bong" in message_content.lower() or f"<@{bong_tools.BOT_USER_ID}>" in message_content:
        return True

    # Slow path: ask the classifier model
    tagged_name = f"{bot_display_name} (Bong)"
    recent_context = "\n".join(
        msg.replace(bot_display_name, tagged_name) if bot_display_name in msg else msg
        for msg in recent_messages[-7:]
    ) if recent_messages else "No recent messages"

    actual_reply_context = reply_context if reply_context else "None"

    # Fill in the classifier prompt template with recent context and the current message
    prompt = classifier_template.replace("{recent_context}", recent_context).replace("{message_content}", message_content).replace("{reply_context}", actual_reply_context)

    try:
        # Run the classifier in a thread to avoid blocking the async event loop
        response = await invoke_with_retry(classifier_model, [HumanMessage(content=prompt)])
        response_text = _extract_response_text(response).upper()
        return "YES" in response_text
    except Exception:
        return False


async def process_attachments(message):
    """Categorize and read message attachments.
    
    Scans all attachments for images, videos, and text files. Images are base64-encoded
    for the vision model. Text files are read into strings. Returns a description string
    plus the raw data needed by describe_image and read_text_file tools.
    """
    attachment_parts = []
    image_attachments = []
    text_attachments = []

    for a in message.attachments:
        # Check content type first (reliable), then fall back to file extension
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
    """Build the voice status string describing Bong's current voice channel state.
    
    Returns (vc, voice_status) where vc is the voice client object (or None)
    and voice_status is a human-readable string for the system prompt.
    """
    if guild:
        vc = guild.voice_client
        if vc and vc.is_connected():
            voice_status = f"Connected to voice channel '{vc.channel.name}'. Do NOT use join_voice — you are already in voice."
            if bong_tools.current_track and (vc.is_playing() or vc.is_paused()):
                voice_status += f"\nCurrently playing '{Path(bong_tools.current_track).stem}'."
            if voice_commands.is_listening(guild.id):
                voice_status += "\nVoice command listener is ACTIVE — listening for 'hey bong' wake word."
        else:
            voice_status = "Not in any voice channel."
    else:
        vc = None
        voice_status = "DM — no voice channels available."
    return vc, voice_status


def build_system_prompt(message, history, voice_status, attachment_desc, image_attachments, text_attachments, replied, replied_to):
    """Fill in the prompt template with all context variables and return the complete system message.
    
    This is the main prompt that the LLM receives. It injects:
      - User name and ID
      - Voice channel status
      - Attachment descriptions (with instructions to use describe_image/read_text_file)
      - Chat history
      - Reply context (if the user is replying to another message)
      - Long-term memories retrieved from ChromaDB
    """
    history_str = "\n".join(history)
    user_msg = message.content.replace("\n", ". ")
    # Retrieve relevant memories based on the user's message and display name
    memories_str = bong_memory_helpers.retrieve_memories(f"{message.author.display_name}: {user_msg}", user_id=message.author.id)

    # Build the reply context block if the user is replying to another message
    replied_content = replied_to.content.replace("\n", ". ") if replied else ""
    reply_block = ""
    if replied:
        reply_block = f"\nThe user is replying to a message. Take into account what they're replying to when responding.\n\nReplied-to user ID: {replied_to.author.id}\nThe user replied to: {replied_to.author.display_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {replied_content}\n"

    # Build the attachment description block with instructions for the LLM
    attachment_block = ""
    if attachment_desc != "None":
        attachment_block = f"\nAttachments included with this message: {attachment_desc}\n"
    if image_attachments:
        attachment_block += "\nYou have image attachments on this message. ALWAYS call describe_image to look at them before responding. Never ignore an image.\n"
    if text_attachments:
        attachment_block += "\nYou have text file attachments on this message. ALWAYS call read_text_file to read them before responding. Never ignore a text file.\n"

    # Build the summaries block from per-channel summaries
    summaries = channel_summaries.get(message.channel.id, [])
    if summaries:
        summaries_str = "\n\nThe older conversation above has been summarized for context. These are compressed versions of what was discussed before the chat history you can see:\n" + "\n".join(f"- {s}" for s in summaries)
    else:
        summaries_str = ""

    # Replace all {placeholders} in the template with the actual values
    return prompt_template.replace("{username}", message.author.display_name).replace("{userID}", str(message.author.id)).replace("{message}", user_msg).replace("{voice_status}", voice_status).replace("{attachments}", attachment_block).replace("{history}", history_str).replace("{reply_context}", reply_block).replace("{memories}", memories_str).replace("{summaries}", summaries_str)


async def update_voice_state(guild, author_id):
    """Update bong_tools.voice_connected and caller_in_voice based on the current guild state.
    
    Tries the cache first (guild.get_member), then falls back to an API call (guild.fetch_member).
    """
    bong_tools.voice_connected = (guild is not None and guild.voice_client is not None and guild.voice_client.is_connected())
    if guild:
        # Try cache first to avoid unnecessary API calls
        member = guild.get_member(author_id)
        if member and member.voice and member.voice.channel:
            bong_tools.caller_in_voice = True
        else:
            # Fall back to API call if not in cache
            member = await guild.fetch_member(author_id)
            bong_tools.caller_in_voice = bool(member and member.voice and member.voice.channel)
    else:
        bong_tools.caller_in_voice = False


def inject_library_context(messages, tool_calls):
    """If any tool calls reference music/image/text libraries, inject the current library listing into the message history.
    
    This gives the LLM the index numbers it needs to select files by name.
    """
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
    """Execute the describe_image tool inline (needs async access to vision model and attachment data).
    
    This can't go through bong_tools because it needs:
      1. The base64 image data from the message attachment
      2. Async invocation of the vision model
    The tool function in bong_tools just returns a placeholder string; the actual work happens here.
    """
    idx = tool_args.get("index", 0)
    question = tool_args.get("question", "Briefly describe this image in 1-2 sentences. Be concise.")
    if not image_attachments:
        return "No image attachments found on this message."
    if idx < 0 or idx >= len(image_attachments):
        return f"Image index {idx} out of range. There are {len(image_attachments)} image(s) attached (indexes 0-{len(image_attachments)-1})."
    img = image_attachments[idx]
    try:
        # First, generate a short label for the filename
        label_msg = HumanMessage(content=[
            {"type": "text", "text": "Describe this image in exactly 5 words as a short label suitable for a filename. Use only letters, numbers, and underscores instead of spaces. Examples: Dog_wearing_small_hat_meme, Orange_cat_sleeping_on_couch, Beautiful_sunset_over_the_ocean. Reply with ONLY the label, nothing else."},
            {"type": "image_url", "image_url": f"data:{img['content_type']};base64,{img['base64']}"},
        ])
        label_response = await invoke_with_retry(description_model, [label_msg])
        raw_label = _extract_response_text(label_response) or "image"
        label = re.sub(r'[^\w]', '', raw_label.replace(" ", "_"))[:50] or "image"

        # Save the image to saved_images/ with the generated label as filename
        save_dir = BONG_DATA / "saved_images"
        save_dir.mkdir(exist_ok=True)
        ext = Path(img['filename']).suffix or ".png"
        save_path = save_dir / f"{label}{ext}"
        save_path.write_bytes(base64.b64decode(img["base64"]))
        debug.log("AI", f"Saved image to {save_path}")

        # Now generate the actual description of the image
        vision_msg = HumanMessage(content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": f"data:{img['content_type']};base64,{img['base64']}"},
        ])
        vision_response = await invoke_with_retry(description_model, [vision_msg])
        vision_text = _extract_response_text(vision_response) or "Could not describe image"
        return f"Description of '{img['filename']}': {vision_text}"
    except Exception as e:
        debug.log("AI", f"Vision error for {img['filename']}: {e}")
        return f"Vision model error for '{img['filename']}': {e}"


async def _handle_read_text_file(tool_args, text_attachments):
    """Execute the read_text_file tool inline (needs access to text attachment contents).
    
    Similar to describe_image — the actual data lives in the async cog context, not in the sync tool.
    """
    idx = tool_args.get("index", 0)
    if not text_attachments:
        return "No text file attachments found on this message."
    if idx < 0 or idx >= len(text_attachments):
        return f"Text file index {idx} out of range. There are {len(text_attachments)} text file(s) attached (indexes 0-{len(text_attachments)-1})."
    txt = text_attachments[idx]
    try:
        # Save the text file to saved_texts/ for future reference
        save_dir = BONG_DATA / "saved_texts"
        save_dir.mkdir(exist_ok=True)
        ext = Path(txt['filename']).suffix or ".txt"
        clean_name = re.sub(r'[^\w]', '_', Path(txt['filename']).stem)[:50] or "text_file"
        save_path = save_dir / f"{clean_name}{ext}"
        save_path.write_text(txt['content'], encoding="utf-8")
        debug.log("AI", f"Saved text file to {save_path}")
        # Return the content (truncated at 8000 chars to stay within LLM context limits)
        content = txt['content']
        if len(content) > 8000:
            content = content[:8000] + f"\n\n... [truncated, {len(txt['content'])} total characters]"
        return f"Contents of '{txt['filename']}':\n{content}"
    except Exception as e:
        debug.log("AI", f"Text file error for {txt['filename']}: {e}")
        return f"Error reading text file '{txt['filename']}': {e}"


async def dispatch_tool(tool_name, tool_args, image_attachments, text_attachments):
    """Execute a single tool call, returning the string result.
    
    describe_image and read_text_file are handled inline because they need
    async access to attachment data. Everything else goes through bong_tools.tool_map.
    """
    if tool_name == "describe_image":
        result = await _handle_describe_image(tool_args, image_attachments)
        debug.log("AI", f"Tool result: {result}")
        return result
    if tool_name == "read_text_file":
        result = await _handle_read_text_file(tool_args, text_attachments)
        debug.log("AI", f"Tool result: {result}")
        return result
    # Unknown tool — tell the LLM it made a mistake
    if tool_name not in bong_tools.tool_map:
        debug.log("AI", f"Unknown tool: {tool_name}")
        return f"Error: '{tool_name}' is not a valid tool name. You may have put the arguments inside the tool name by mistake. Call the tool again with just the tool name and the arguments as separate parameters. Available tools: {', '.join(bong_tools.tool_map.keys())}"
    try:
        # Run the sync tool function in a thread to avoid blocking the event loop
        result = await asyncio.to_thread(bong_tools.tool_map[tool_name].invoke, tool_args)
        debug.log("AI", f"Tool result: {result}")
        return result
    except Exception as e:
        debug.log("AI", f"Tool error: {e}")
        return f"Error: {e}"


from llm_utils import _extract_response_text


async def send_chunked(channel, text, max_len=2000):
    """Send text to a Discord channel, splitting into multiple messages if it exceeds max_len.
    
    Splits at newlines when possible to avoid breaking mid-sentence.
    """
    if not text:
        return
    while text:
        if len(text) <= max_len:
            await channel.send(text)
            return
        # Find a good split point — prefer newline, then space, then hard cut
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1 or split_at < max_len // 2:
            split_at = text.rfind(" ", 0, max_len)
        if split_at == -1:
            split_at = max_len
        await channel.send(text[:split_at])
        text = text[split_at:].lstrip("\n ")


async def run_tool_loop(bound_model, messages, image_attachments, text_attachments, last_prompt_path):
    """Run the LLM invocation + tool-call loop.
    
    The loop works like this:
      1. Send the full conversation (system prompt + history) to the LLM
      2. If the LLM responds with tool calls, execute them and feed results back
      3. Repeat until the LLM gives a plain text response (or hit MAX_TOOL_ITERATIONS)
      4. Stream the final response token by token for faster perceived output
    
    Returns (result_text, tool_summaries) where tool_summaries is a list of
    short descriptions like "web_search(query)" for inclusion in chat history.
    """
    total_input = 0
    total_output = 0

    def _tally(response):
        nonlocal total_input, total_output
        if response and response.usage_metadata:
            total_input += response.usage_metadata.get("input_tokens", 0)
            total_output += response.usage_metadata.get("output_tokens", 0)

    ai_response = await invoke_with_retry(bound_model, messages)
    _tally(ai_response)
    messages.append(ai_response)

    iteration = 0
    tool_summaries = []

    while ai_response.tool_calls:
        iteration += 1
        if iteration > MAX_TOOL_ITERATIONS:
            # Force the LLM to stop tool-calling and give a final response
            messages.append(HumanMessage(content="You have exceeded the maximum number of tool calls. Please respond to the user now without making any more tool calls."))
            ai_response = await invoke_with_retry(bound_model, messages)
            _tally(ai_response)
            messages.append(ai_response)
            break

        # If any tool calls need library listings, inject them as context
        inject_library_context(messages, ai_response.tool_calls)

        for tc in ai_response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            debug.log("AI", f"Tool call: {tool_name}({tool_args})")
            debug.log_to_file("AI", f"TOOL CALL: {tool_name}({tool_args})")
            if last_prompt_path:
                last_prompt_path.write_text(
                    last_prompt_path.read_text(encoding="utf-8") + f"\nTOOL CALL: {tool_name}({tool_args})\n",
                    encoding="utf-8",
                )

            tool_result = await dispatch_tool(tool_name, tool_args, image_attachments, text_attachments)
            # Feed the tool result back to the LLM as a ToolMessage
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tc["id"]))

            # Build a short summary for the chat history record
            summary_args = ', '.join(str(v) for v in tool_args.values())
            tool_summaries.append(f"{tool_name}({summary_args})")
            debug.log_to_file("AI", f"TOOL RESULT ({tool_name}): {tool_result}")
            if last_prompt_path:
                last_prompt_path.write_text(
                    last_prompt_path.read_text(encoding="utf-8") + f"TOOL RESULT ({tool_name}): {tool_result}\n",
                    encoding="utf-8",
                )

        # Re-invoke the LLM with the tool results so it can decide what to do next
        ai_response = await invoke_with_retry(bound_model, messages)
        _tally(ai_response)
        messages.append(ai_response)

    # Extract the final text from the last AI response
    result_text = _extract_response_text(ai_response)

    # If the LLM returned an empty response, retry once without tool binding
    if not result_text.strip():
        debug.log("AI", "Empty response, retrying with thinking model")
        retry_response = await invoke_with_retry(base_model, messages)
        _tally(retry_response)
        result_text = _extract_response_text(retry_response)

    debug.log("Tokens", f"in={total_input} out={total_output} total={total_input + total_output}")
    if bong_tools.current_user_id:
        user_data.add_tokens(bong_tools.current_user_id, total_input + total_output, bong_tools.current_username or "")

    debug.log("AI", "Generating final response")
    if last_prompt_path:
        last_prompt_path.write_text(
            last_prompt_path.read_text(encoding="utf-8") + f"\nFINAL RESPONSE:\n{result_text}\n",
            encoding="utf-8",
        )

    return result_text, tool_summaries


def _make_after_play_callback(guild, loop):
    """Create a closure that auto-continues playback after a track finishes.
    
    This is attached to discord.py's FFmpegPCMAudio as the `after` callback.
    When a song ends naturally, this function checks the queue, then loop,
    then shuffle, and starts the next track automatically.
    """
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
            next_track, _desc = bong_tools.advance_queue()
            if next_track:
                bong_song_stats._increment_song(Path(next_track).stem)
                vc.play(discord.FFmpegPCMAudio(next_track, options="-filter:a volume=0.3"), after=after_play)
                asyncio.run_coroutine_threadsafe(_set_voice_status(guild, _build_now_playing_status()), loop)
            else:
                asyncio.run_coroutine_threadsafe(_set_voice_status(guild, None), loop)
        except Exception as e:
            debug.log("Audio", f"Auto-continue error: {e}")
    return after_play


# ========== Voice/audio action dispatchers ==========
# These functions read the pending_* flags set by the sync tool functions
# and perform the actual async Discord API calls. Each returns an error string
# or None on success.

async def _set_voice_status(guild, status: str | None):
    """Set or clear the voice channel status text. Silently ignores permission errors."""
    if not guild or not guild.voice_client or not guild.voice_client.channel:
        return
    try:
        await guild.voice_client.channel.edit(status=status)
    except discord.Forbidden:
        pass
    except Exception as e:
        debug.log("Audio", f"Failed to set voice status: {e}")


def _build_now_playing_status():
    """Build the voice channel status string from current playback state."""
    if not bong_tools.current_track:
        return None
    name = Path(bong_tools.current_track).stem
    status = f"🎵 {name}"
    if bong_tools.loop_enabled and bong_tools.queue_snapshot:
        status += " 🔁Q"
    elif bong_tools.loop_enabled and bong_tools.loop_track:
        status += " 🔁"
    elif bong_tools.loop_enabled:
        status += " 🔁"
    if bong_tools.shuffle_enabled:
        status += " 🔀"
    return status

async def _dispatch_join_voice(guild):
    """Handle pending_join_voice flag. Connects to the target user's voice channel."""
    if not bong_tools.pending_join_voice:
        return None
    error = None
    if guild:
        target_member = guild.get_member(bong_tools.pending_join_voice)
        if target_member and target_member.voice and target_member.voice.channel:
            try:
                # If voice commands are active, use VoiceRecvClient to maintain receive capability
                if voice_commands.is_listening(guild.id):
                    from discord.ext.voice_recv import VoiceRecvClient
                    await target_member.voice.channel.connect(cls=VoiceRecvClient)
                else:
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


async def _stop_playback(vc):
    """Stop audio playback without killing the voice listener.

    VoiceRecvClient.stop() kills BOTH playback and the voice command
    listener, so we use stop_playing() instead when available.
    """
    from discord.ext.voice_recv import VoiceRecvClient
    if isinstance(vc, VoiceRecvClient):
        vc.stop_playing()
    else:
        vc.stop()


async def _dispatch_leave_voice(guild):
    """Handle pending_leave_voice flag. Disconnects from the current voice channel."""
    if not bong_tools.pending_leave_voice:
        return None
    error = None
    if guild and guild.voice_client:
        # Stop voice command listener before disconnecting
        await voice_commands.stop_listening(guild)
        await _set_voice_status(guild, None)
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
    """Handle pending_play_audio flag. Starts playback of the queued track."""
    if not bong_tools.pending_play_audio:
        return None
    vc = guild.voice_client if guild else None
    if not vc or not vc.is_connected():
        bong_tools.pending_play_audio = None
        return "Not in a voice channel, can't play audio. Join a voice channel first."
    try:
        track_path = bong_tools.pending_play_audio
        bong_tools.current_track = track_path
        # Bind deferred loop track now that we have a current track
        if bong_tools.loop_enabled and not bong_tools.loop_track and not bong_tools.queue_snapshot:
            bong_tools.loop_track = track_path
        after_play = _make_after_play_callback(guild, asyncio.get_running_loop())
        # If something is already playing, stop it first and wait briefly
        if vc.is_playing() or vc.is_paused():
            await _stop_playback(vc)
            await asyncio.sleep(0.5)
        source = discord.FFmpegPCMAudio(track_path, options="-filter:a volume=0.3")
        vc.play(source, after=after_play)
        await _set_voice_status(guild, _build_now_playing_status())
    except Exception as e:
        debug.log("AI", f"Failed to play audio: {e}")
        return f"Failed to play audio: {e}"
    finally:
        bong_tools.pending_play_audio = None
    return None


async def _dispatch_loop_audio(guild):
    """Handle loop track start when nothing is currently playing.
    
    If audio is already playing, after_play will handle loop continuation — 
    stopping and restarting here would cause the current song to replay from
    the beginning on every dispatch cycle.
    """
    if not bong_tools.loop_enabled or not bong_tools.loop_track:
        return None
    vc = guild.voice_client if guild else None
    if not vc or not vc.is_connected():
        bong_tools.loop_track = None
        return None
    if vc.is_playing() or vc.is_paused():
        return None
    bong_tools.current_track = bong_tools.loop_track
    bong_tools.pending_play_audio = bong_tools.loop_track
    bong_tools.loop_track = None
    return None


async def _dispatch_pause_audio(guild):
    """Handle pending_pause flag."""
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
    """Handle pending_resume flag."""
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
    """Handle pending_stop flag. Stops playback and clears loop/shuffle state."""
    if not bong_tools.pending_stop:
        return None
    # Reset all playback state when stopping
    bong_tools.loop_enabled = False
    bong_tools.loop_track = None
    bong_tools.queue_snapshot = []
    bong_tools.shuffle_enabled = False
    bong_tools.current_track = None
    bong_tools.song_queue.clear()
    vc = guild.voice_client if guild else None
    error = None
    if vc and (vc.is_playing() or vc.is_paused()):
        await _stop_playback(vc)
    else:
        error = "Nothing is playing to stop."
    await _set_voice_status(guild, None)
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
        return "Nothing is playing to skip."
    if not bong_tools.pending_skip_target:
        bong_tools.pending_skip = False
        return "No music files to skip to."
    bong_tools.current_track = str(bong_tools.pending_skip_target)
    bong_tools.pending_play_audio = str(bong_tools.pending_skip_target)
    await _stop_playback(vc)
    bong_tools.pending_skip = False
    bong_tools.pending_skip_target = None
    return None


async def _dispatch_start_listening(guild, bot):
    """Handle pending_start_listening flag. Starts the voice command listener."""
    if not bong_tools.pending_start_listening:
        return None
    bong_tools.pending_start_listening = None

    if not guild:
        return "Cannot start voice commands in DMs."

    channel_id = bong_tools.current_channel_id
    text_channel = None
    if channel_id:
        text_channel = guild.get_channel(channel_id)
        if not text_channel:
            text_channel = bot.get_channel(channel_id)

    result = await voice_commands.start_listening(bot, guild, text_channel)
    if result:
        debug.log("AI", f"Voice cmd listener: {result}")
        if result.startswith("Listening"):
            return None
        return result
    return None


async def _dispatch_stop_listening(guild):
    """Handle pending_stop_listening flag. Stops the voice command listener."""
    if not bong_tools.pending_stop_listening:
        return None
    bong_tools.pending_stop_listening = False

    if not guild:
        return "Cannot stop voice commands in DMs."

    result = await voice_commands.stop_listening(guild)
    if result:
        debug.log("AI", f"Voice cmd listener: {result}")
        if result.startswith("Stopped"):
            return None
        return result
    return None


async def dispatch_voice_actions(guild, message, bot=None):
    """Dispatch all pending voice/audio/file-send actions in order.
    
    Order matters: join before play, stop/skip before play, etc.
    Also sends any queued image or text files.
    Returns the first error encountered, or None on success.
    """
    try:
        # Dispatch in priority order — stop/skip before play, join before play, etc.
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

        # Dispatch voice command listener
        if bot:
            start_result = await _dispatch_start_listening(guild, bot)
            stop_result = await _dispatch_stop_listening(guild)
            if start_result:
                results.append(start_result)
            if stop_result:
                results.append(stop_result)

        # Update voice channel status after all dispatchers have run
        # (catches loop/shuffle toggles, play, stop, etc.)
        if guild and guild.voice_client and (guild.voice_client.is_playing() or guild.voice_client.is_paused()):
            await _set_voice_status(guild, _build_now_playing_status())

        # Send any queued image files
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

        # Send any queued text files
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

        # Return the first error from voice dispatchers
        for error in results:
            if error:
                return error
        return None

    except Exception as e:
        debug.log("AI", f"Error during voice/audio dispatch: {e}")
        debug.error("AI", f"Error during voice/audio dispatch: {e}")
        # Force-disconnect the voice client on error to reset the Opus state
        if guild and guild.voice_client:
            try:
                await guild.voice_client.disconnect(force=True)
                debug.log("AI", "Force-disconnected voice client to reset Opus state")
            except Exception:
                pass
        return f"An error occurred: {e}"


async def apply_reactions(message):
    """Apply any pending emoji reactions to the message and clear the queue."""
    for emoji in bong_tools.pending_reactions:
        try:
            await message.add_reaction(emoji)
        except Exception as e:
            debug.log("AI", f"Failed to react with {emoji}: {e}")
    bong_tools.pending_reactions.clear()


async def summarize_history_chunk(text: str) -> str:
    """Compress a chunk of conversation history into a short summary using the description model."""
    prompt = (
        "Summarize this conversation in 2-3 short sentences. "
        "Preserve key facts, names, decisions, and any important context. "
        "Be concise and factual.\n\n"
        f"{text}"
    )
    response = await invoke_with_retry(description_model, [HumanMessage(content=prompt)])
    return _extract_response_text(response).strip()


async def _summarize_and_trim(channel_id: int, to_summarize: list[str]):
    """Background task: summarize old messages and trim them from history."""
    try:
        summary_text = "\n".join(to_summarize)
        try:
            summary = await summarize_history_chunk(summary_text)
            if summary:
                channel_summaries.setdefault(channel_id, []).append(summary)
                if len(channel_summaries[channel_id]) > MAX_SUMMARIES_PER_CHANNEL:
                    channel_summaries[channel_id] = channel_summaries[channel_id][-MAX_SUMMARIES_PER_CHANNEL:]
                debug.log("AI", f"Summarized {len(to_summarize)} messages for channel {channel_id}")
        except Exception as e:
            debug.log("AI", f"Background summarization failed: {e}")
        history = chat_memories.get(channel_id)
        if history and len(history) >= SUMMARIZE_CHUNK_SIZE:
            history[-SUMMARIZE_CHUNK_SIZE:] = []
    finally:
        _summarization_in_progress.discard(channel_id)


def record_history(history, message, result, attachment_desc, tool_summaries):
    """Store the exchange in the channel's rolling history.
    
    The history is stored newest-first (index 0 = most recent).
    Trimming is handled by background summarization — see _summarize_and_trim.
    """
    attachment_suffix = f" {attachment_desc}" if attachment_desc != "None" else ""
    tool_summary_str = f" [Already completed: {'; '.join(tool_summaries)}]" if tool_summaries else ""
    history_entry = f"{message.author.display_name} at {datetime.now().strftime('%H:%M')}: {message.content.replace(chr(10), '. ')}{attachment_suffix}\nBong's response: {result.replace(chr(10), '. ')}{tool_summary_str}"
    history.insert(0, history_entry)
    debug.log("AI", f"response {len(history)}")
    debug.log_to_file("AI", f"RESPONSE: {result}")


def record_passive_message(history, message, attachment_desc):
    """Store a non-Bong-targeted message in channel history for context.
    
    Even messages not directed at Bong are recorded so the classifier and LLM
    have recent conversation context to work with.
    Trimming is handled by background summarization — see _summarize_and_trim.
    """
    attachment_suffix = f" {attachment_desc}" if attachment_desc != "None" else ""
    history_entry = f"{message.author.display_name} at {datetime.now().strftime('%H:%M')}: {message.content.replace(chr(10), '. ')}{attachment_suffix}"
    history.insert(0, history_entry)
    debug.log("AI", f"remembered {len(history)}")


async def process_voice_command(bot, guild, channel, user_id: int, username: str, text: str):
    """Process a transcribed voice command through Bong's LLM pipeline.

    This is called by voice_commands.py when a wake word is detected.
    It mimics the on_message pipeline but with a synthesized message.
    """
    if channel is None:
        debug.log("VoiceCmd", f"Cannot process voice command: no text channel (user={user_id})")
        return

    # Per-user cooldown to prevent rapid overlapping voice commands
    now = time.time()
    last_cmd = _voice_cooldowns.get(user_id, 0)
    if now - last_cmd < VOICE_COOLDOWN_SECONDS:
        debug.log("VoiceCmd", f"Cooldown: user {user_id} sent voice command too quickly ({now - last_cmd:.1f}s < {VOICE_COOLDOWN_SECONDS}s)")
        return
    _voice_cooldowns[user_id] = now
    if len(_voice_cooldowns) > 20:
        oldest = sorted(_voice_cooldowns, key=lambda k: _voice_cooldowns[k])[:len(_voice_cooldowns) - 20]
        for k in oldest:
            del _voice_cooldowns[k]

    try:
        history = chat_memories.setdefault(channel.id, [])
        voice_status_str = "Not in any voice channel."
        if guild and guild.voice_client and guild.voice_client.is_connected():
            vc = guild.voice_client
            voice_status_str = f"Connected to voice channel '{vc.channel.name}'."
            if bong_tools.current_track and (vc.is_playing() or vc.is_paused()):
                voice_status_str += f"\nCurrently playing '{Path(bong_tools.current_track).stem}'."
        if guild and voice_commands.is_listening(guild.id):
            voice_status_str += "\nVoice command listener is ACTIVE — listening for 'hey bong' wake word."

        # Build the system prompt using a synthetic approach
        memories_str = bong_memory_helpers.retrieve_memories(f"{username}: {text}", user_id=user_id)
        summaries = channel_summaries.get(channel.id, [])
        summaries_str = ""
        if summaries:
            summaries_str = "\n\nThe older conversation above has been summarized for context. These are compressed versions of what was discussed before the chat history you can see:\n" + "\n".join(f"- {s}" for s in summaries)

        system_msg_content = prompt_template.replace("{username}", username).replace("{userID}", str(user_id)).replace("{message}", text).replace("{voice_status}", voice_status_str).replace("{attachments}", "").replace("{history}", "\n".join(history)).replace("{reply_context}", "").replace("{memories}", memories_str).replace("{summaries}", summaries_str)

        messages = [HumanMessage(content=system_msg_content)]

        last_prompt_path = BONG_DATA / "logs" / "last_prompt.log" if debug.is_debug() else None
        if last_prompt_path:
            last_prompt_path.parent.mkdir(parents=True, exist_ok=True)
            last_prompt_path.write_text(system_msg_content + "\n\n========\n", encoding="utf-8")
        debug.log_to_file("AI", f"VOICE COMMAND from {username} ({user_id}): {text}")

        # Set up shared state for this invocation
        await update_voice_state(guild, user_id)
        bong_tools.current_user_id = user_id
        bong_tools.current_username = username
        bong_tools.current_channel_id = channel.id

        bound_model = base_model.bind_tools(bong_tools.tools)
        result, tool_summaries = await run_tool_loop(bound_model, messages, [], [], last_prompt_path)

        # Voice commands have no Discord message to react to, so discard any pending reactions
        bong_tools.pending_reactions.clear()

        # Record in history
        history_entry = f"{username} at {datetime.now().strftime('%H:%M')} (voice): {text}\nBong's response: {result.replace(chr(10), '. ')}"
        history.insert(0, history_entry)
        debug.log("AI", f"voice response recorded {len(history)}")

        # Dispatch voice actions — for voice commands we pass the text channel
        # directly since there's no Discord message object
        voice_error = None
        if guild:
            # Create a minimal namespace for dispatch_voice_actions
            # that has a .channel attribute pointing to our text channel
            class _FakeMessage:
                def __init__(self, ch):
                    self.channel = ch
            voice_error = await dispatch_voice_actions(guild, _FakeMessage(channel), bot=bot)

        # Send response to text channel
        if voice_error:
            messages.append(HumanMessage(content=f"System: The voice/audio action failed with this error: {voice_error}. Please let the user know and suggest what they can do."))
            error_response = await invoke_with_retry(bound_model, messages)
            error_text = _extract_response_text(error_response)
            await send_chunked(channel, error_text)
        else:
            await send_chunked(channel, result)
        debug.log("VoiceCmd", f"Voice command processed: {text[:50]}...")

        if bong_tools.pending_shutdown:
            if user_data.has_permission(user_id, "admin"):
                if guild and guild.voice_client and guild.voice_client.channel:
                    await _set_voice_status(guild, None)
                await channel.send("🫡")
                await bot.close()
            else:
                debug.log("VoiceCmd", "Unauthorized shutdown attempt")
            bong_tools.pending_shutdown = False

    except Exception as e:
        debug.log("VoiceCmd", f"Error processing voice command: {e}")
        debug.error("VoiceCmd", f"Error processing voice command: {e}")
        debug.log_to_file("VoiceCmd", f"Error processing voice command: {e}")
        bong_tools.reset_pending()
        try:
            await channel.send("Something went wrong processing that voice command. Try again?")
        except Exception:
            pass


class BongCog(commands.Cog):
    """Main Discord bot cog — handles all message events and the LLM tool loop."""
    
    def __init__(self, bot):
        self.bot = bot
        self._processed_ids: set[int] = set()
        self._user_cooldowns: dict[int, float] = {}
        # Preload channel history on startup so Bong has context immediately
        self.bot.loop.create_task(self._preload_channel())
        # Start the reminder checker background task
        self.reminder_task = self.bot.loop.create_task(self._check_reminders())
        # Start the periodic persist flush task
        self.persist_task = self.bot.loop.create_task(self._flush_persist_periodically())
        # Start the e621 subscription checker background task
        self.e621_task = self.bot.loop.create_task(self._check_e621_subscriptions())

    async def _flush_persist_periodically(self):
        """Flush all persist stores to disk every 60 seconds if dirty."""
        await self.bot.wait_until_ready()
        while not self.bot.is_closed():
            await asyncio.sleep(60)
            persist.flush_all()

    async def cog_unload(self):
        """Flush all persist stores to disk when the cog is unloaded (shutdown/reload)."""
        persist.flush_all()
        self.reminder_task.cancel()
        self.persist_task.cancel()
        self.e621_task.cancel()
    
    async def _preload_channel(self):
        """Load recent message history from debug channels on startup so Bong has context."""
        await self.bot.wait_until_ready()
        for d_channel in DEBUG_CHANNEL_IDS:
            channel = self.bot.get_channel(d_channel)
            # If the ID doesn't match a guild channel, try treating it as a DM
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

    async def _check_reminders(self):
        """Background task that checks for due reminders every 30 seconds and DMs the user."""
        await self.bot.wait_until_ready()
        reminders.load_reminders()
        while not self.bot.is_closed():
            try:
                now = datetime.now().timestamp()
                due = [r for r in reminders.reminders if r["due_at"] <= now]
                for r in due:
                    reminders.reminders.remove(r)
                    user = self.bot.get_user(r["user_id"])
                    if not user:
                        try:
                            user = await self.bot.fetch_user(r["user_id"])
                        except Exception:
                            continue
                    try:
                        await user.send(f"⏰ **Reminder**: {r['message']}")
                    except discord.Forbidden:
                        pass
                if due:
                    reminders.save_reminders()
            except Exception as e:
                debug.log("Reminders", f"Error checking reminders: {e}")
            await asyncio.sleep(30)

    async def _check_e621_subscriptions(self):
        """Background task that polls e621 for new posts matching subscribed tags and DMs users."""
        await self.bot.wait_until_ready()
        bong_e621.load_subscriptions()
        while not self.bot.is_closed():
            if bong_e621.tag_registry:
                debug.log("e621", f"Polling {len(bong_e621.tag_registry)} tag(s)")
            dirty = False
            for tags, last_id in list(bong_e621.tag_registry.items()):
                try:
                    new_posts, new_id = bong_e621.get_new_posts(tags, last_id)
                    if new_id != last_id:
                        bong_e621.tag_registry[tags] = new_id
                        dirty = True
                    if new_posts:
                        subscriber_ids = user_data.get_all_e621_subscribers(tags)
                        lines = [f"New e621 post matching '{tags}':"]
                        for p in new_posts[:5]:
                            pid = p.get("id", "?")
                            score = p.get("score", {}).get("total", 0)
                            rating = p.get("rating", "?")
                            lines.append(f"  #{pid} [score:{score} rating:{rating}] https://e621.net/posts/{pid}")
                        msg = "\n".join(lines)
                        for uid in subscriber_ids:
                            try:
                                user = self.bot.get_user(uid) or await self.bot.fetch_user(uid)
                                await user.send(msg)
                            except discord.Forbidden:
                                pass
                            except Exception as e:
                                debug.log("e621", f"Could not DM user {uid}: {e}")
                except Exception as e:
                    debug.log("e621", f"Error polling tag '{tags}': {e}")
                await asyncio.sleep(1)
            if dirty:
                bong_e621.save_subscriptions()
            await asyncio.sleep(bong_e621.E621_POLL_INTERVAL)

    @commands.Cog.listener()
    async def on_message(self, message):
        """Main message handler — runs for every message in every active channel."""
        # Ignore own messages
        if message.author == self.bot.user:
            return

        # Deduplication: Discord can deliver the same gateway event multiple times
        # (e.g. on reconnect/resume), which would cause duplicate LLM invocations.
        # Since this check-and-add is synchronous (no await), it's atomic in the
        # asyncio event loop — no other coroutine can interleave between the check
        # and the add, so there's no race condition.
        if message.id in self._processed_ids:
            return
        self._processed_ids.add(message.id)
        # Prevent unbounded memory growth — keep only recent IDs
        # (Discord snowflakes are monotonically increasing, so we keep the highest)
        if len(self._processed_ids) > 2000:
            self._processed_ids = set(sorted(self._processed_ids)[-1000:])

        # Cooldown: users without llm_fast must wait 60s between Bong responses.
        if not user_data.has_permission(message.author.id, "llm_fast"):
            now = datetime.now().timestamp()
            last = self._user_cooldowns.get(message.author.id, 0)
            if now - last < 60:
                remaining = int(60 - (now - last))
                await message.channel.send(f"Slow down! Try again in {remaining}s.", delete_after=5)
                return
            self._user_cooldowns[message.author.id] = now

        # Handle DMs — check llm permission before responding
        if isinstance(message.channel, discord.DMChannel):
            if not user_data.has_permission(message.author.id, "llm"):
                should_process = await dm_approval.process_dm(message, self.bot)
                if not should_process:
                    return
            # Activate their DM channel for Bong if not already active
            if message.channel.id not in active_channels:
                active_channels.add(message.channel.id)
                chat_memories.setdefault(message.channel.id, [])
                async for msg in message.channel.history(limit=MAX_MEMORY_SIZE):
                    chat_memories[message.channel.id].append(f"{msg.author.display_name} at {msg.created_at.strftime('%H:%M')}: {msg.content}")

        # For guild channels, only process messages in active channels and from users with llm permission
        if not isinstance(message.channel, discord.DMChannel) and message.channel.id not in active_channels:
            return
        # In guild channels, ignore users without llm permission
        if not isinstance(message.channel, discord.DMChannel) and not user_data.has_permission(message.author.id, "llm"):
            return
        # Ignore bot commands (messages starting with the command prefix that match a real command)
        if message.content.startswith(self.bot.command_prefix):
            after_prefix = message.content[len(self.bot.command_prefix):].strip()
            command_name = after_prefix.split()[0] if after_prefix else ""
            if command_name in self.bot.all_commands:
                return

        # Read and categorize any attachments on the message
        attachment_desc, image_attachments, text_attachments = await process_attachments(message)
        # Get or create the channel's rolling history
        history = chat_memories.setdefault(message.channel.id, [])

        # Kick off background summarization if history exceeds the threshold
        if len(history) >= MAX_MEMORY_SIZE and message.channel.id not in _summarization_in_progress:
            _summarization_in_progress.add(message.channel.id)
            to_summarize = list(history[-SUMMARIZE_CHUNK_SIZE:])
            asyncio.create_task(_summarize_and_trim(message.channel.id, to_summarize))

        # Grab the last 7 messages for the classifier's context window
        recent_messages = history[:7]

        # Check if the user is replying to another message
        replied_to = ""
        replied = False
        reply_context = ""
        if message.reference is not None:
            replied_to = await message.channel.fetch_message(message.reference.message_id)
            replied = True
            reply_context = f"User is replying to {replied_to.author.display_name}: {replied_to.content.replace(chr(10), '. ')}"

        # If the message isn't directed at Bong, just record it as context and move on
        if not await is_talking_to_bong(message.content.replace(chr(10), '. '), recent_messages, reply_context, self.bot.user.display_name):
            record_passive_message(history, message, attachment_desc)
            return

        # Show "Bong is typing..." while processing
        async with message.channel.typing():
            try:
                guild = message.guild
                _, voice_status = build_voice_status(guild)
                system_msg = build_system_prompt(message, history, voice_status, attachment_desc, image_attachments, text_attachments, replied, replied_to)

                # Start the message history with the system prompt
                messages = [HumanMessage(content=system_msg)]

                # Write the full prompt to the log file for debugging
                last_prompt_path = BONG_DATA / "logs" / "last_prompt.log" if debug.is_debug() else None
                if last_prompt_path:
                    last_prompt_path.parent.mkdir(parents=True, exist_ok=True)
                    last_prompt_path.write_text(system_msg + "\n\n========\n", encoding="utf-8")

                debug.log_to_file("AI", f"QUERY from {message.author.display_name} ({message.author.id}): {message.content}")

                # Set up shared state for this message's tool loop
                await update_voice_state(guild, message.author.id)
                bong_tools.current_user_id = message.author.id
                bong_tools.current_username = message.author.display_name
                bong_tools.current_channel_id = message.channel.id

                # Bind tools to a fresh model instance for this request
                bound_model = base_model.bind_tools(bong_tools.tools)
                # Run the LLM tool loop — this may involve multiple LLM calls and tool executions
                result, tool_summaries = await run_tool_loop(bound_model, messages, image_attachments, text_attachments, last_prompt_path)

                # Add emoji reactions if the LLM called the react tool
                await apply_reactions(message)
                # Record the exchange in channel history
                record_history(history, message, result, attachment_desc, tool_summaries)

                # Dispatch all pending voice/audio/file actions
                voice_error = await dispatch_voice_actions(guild, message, bot=self.bot)

                # If a voice action failed, ask the LLM to explain the error to the user
                if voice_error:
                    messages.append(HumanMessage(content=f"System: The voice/audio action failed with this error: {voice_error}. Please let the user know and suggest what they can do."))
                    error_response = await invoke_with_retry(bound_model, messages)
                    error_text = _extract_response_text(error_response)
                    await send_chunked(message.channel, error_text)
                else:
                    await send_chunked(message.channel, result)

                # Handle shutdown if the LLM called the shutdown tool
                if bong_tools.pending_shutdown:
                    if user_data.has_permission(message.author.id, "admin"):
                        # Clear voice channel status before shutting down
                        if message.guild and message.guild.voice_client and message.guild.voice_client.channel:
                            await _set_voice_status(message.guild, None)
                        await message.add_reaction("🫡")
                        await self.bot.close()
                    else:
                        debug.log("AI", "Unauthorized shutdown attempt")
                    bong_tools.pending_shutdown = False

            except Exception as e:
                debug.log("AI", f"Error generating response: {e}")
                debug.error("AI", f"Error generating response: {e}")
                debug.log_to_file("AI", f"Error generating response: {e}")
                # Clear all pending state so it doesn't leak into the next message
                bong_tools.reset_pending()
                await send_chunked(message.channel, "Something went wrong processing that message. Try again?")

    @commands.command(name="llm", help="Toggle Bong's activity in the current channel")
    async def llm(self, ctx):
        """Toggle Bong's activity in the current channel. Admin only."""
        if not user_data.has_permission(ctx.author.id, "admin"):
            await ctx.send("Only admins can use this command.")
            return
        if ctx.channel.id in active_channels:
            active_channels.discard(ctx.channel.id)
            await ctx.send(f"No more Bonging in this channel! (ID: {ctx.channel.id})")
        else:
            active_channels.add(ctx.channel.id)
            # Load recent history so Bong has immediate context
            history = chat_memories.setdefault(ctx.channel.id, [])
            async for msg in ctx.channel.history(limit=MAX_MEMORY_SIZE):
                history.append(f"{msg.author.display_name} at {msg.created_at.strftime('%H:%M')}: {msg.content}")
            await ctx.send(f"Bong is now active in this channel! (ID: {ctx.channel.id})")
            debug.log("AI", history)

    @commands.group(name="tags", help="Manage user permission tags")
    async def tags(self, ctx):
        """Manage user permission tags. Admin only. Subcommands: list, add, remove."""
        if ctx.invoked_subcommand is None:
            await ctx.send("Usage: @tags list <user_id> | @tags add <user_id> <tag> | @tags remove <user_id> <tag>\nTags: llm, llm_fast, music, vc_commands, e621, admin")

    @tags.command(name="list", help="List a user's permission tags")
    async def tags_list(self, ctx, user_id: int):
        """List the permission tags for a user. Admin only."""
        if not user_data.has_permission(ctx.author.id, "admin"):
            await ctx.send("Only admins can use this command.")
            return
        perms = user_data.get_permissions(user_id)
        if not perms:
            await ctx.send(f"User {user_id} has no tags (unknown user).")
        else:
            await ctx.send(f"User {user_id} tags: {', '.join(perms)}")

    @tags.command(name="add", help="Add a permission tag to a user")
    async def tags_add(self, ctx, user_id: int, tag: str):
        """Add a permission tag to a user. Admin only. Tags: llm, llm_fast, music, vc_commands, e621, admin"""
        if not user_data.has_permission(ctx.author.id, "admin"):
            await ctx.send("Only admins can use this command.")
            return
        if tag not in user_data.VALID_TAGS:
            await ctx.send(f"Invalid tag '{tag}'. Valid tags: {', '.join(sorted(user_data.VALID_TAGS))}")
            return
        user_data.add_permission(user_id, tag)
        await ctx.send(f"Added '{tag}' tag to user {user_id}. Current tags: {', '.join(user_data.get_permissions(user_id))}")

    @tags.command(name="remove", help="Remove a permission tag from a user")
    async def tags_remove(self, ctx, user_id: int, tag: str):
        """Remove a permission tag from a user. Admin only."""
        if not user_data.has_permission(ctx.author.id, "admin"):
            await ctx.send("Only admins can use this command.")
            return
        if user_data.remove_permission(user_id, tag):
            await ctx.send(f"Removed '{tag}' tag from user {user_id}. Current tags: {', '.join(user_data.get_permissions(user_id))}")
        else:
            await ctx.send(f"User {user_id} doesn't have the '{tag}' tag.")

async def setup(bot):
    """Entry point for discord.py to load this cog."""
    await bot.add_cog(BongCog(bot))