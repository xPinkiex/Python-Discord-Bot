# Bong

A Discord bot with a protogen personality, powered by a local LLM (Ollama) with LangChain tools. Bong can chat, remember things about people, play music, set reminders, summarize URLs, search e621, and respond to voice commands.

## Features

- **LLM Chat** — Responds in-character as Bong, a green protogen with rainbow accents, using a hosted model via Ollama. Automatically detects when someone is talking to it (keyword fast-path + classifier LLM slow-path)
- **Long-term Memory** — Saves facts about people, detects contradictions via LLM, recency-weighted retrieval, 180-day expiry, per-user and general memories via ChromaDB vector store
- **Permission Tags** — Six granular tags (`llm`, `llm_fast`, `music`, `vc_commands`, `e621`, `admin`) assigned per-user, with implication rules (`admin` → all, `llm_fast` → `llm`)
- **DM Approval Flow** — Unknown users who DM Bong trigger an approval request to the owner with preset Discord button options (Chat, Chat+Music, Full Access, Admin, Deny)
- **`@tags` Command** — Admin-only command to list, add, or remove permission tags from any user
- **Music Playback** — Download from YouTube, play in voice channels, pause/resume/skip, loop (track or queue), shuffle, song queue with auto-advance
- **Song Stats** — Tracks play counts, displays top songs and total plays in `bot_stats`
- **Reminders** — Set, list, and cancel reminders delivered via DM, persistent across restarts, supports relative and absolute times with timezone awareness
- **Timezone Storage** — Per-user UTC offset and named timezone support, `current_time` shows local time automatically
- **URL Summarization** — Fetches web pages, extracts text, summarizes via LLM
- **Web Search** — DuckDuckGo search and YouTube search
- **e621 Subscriptions** — Subscribe to e621 tags, get DM notifications when new posts match, search e621, full tag validation against the API
- **Voice Commands** — Wake word detection ("hey bong") via custom openWakeWord ONNX model, Whisper transcription, LLM processing in voice channel text chat
- **Image/Text Sharing** — Send saved images and text files in chat, describe attached images via vision model
- **Bot Stats** — Uptime, memory count, known users, reminders, top 3 most-played songs
- **Streaming Responses** — LLM responses stream to Discord in real-time, auto-split across messages when exceeding 2000 characters
- **Conversation Summarization** — Rolling per-channel history with automatic summarization of older messages

## Permission Tags

Users are granted access to features via tags stored in `bong_user_data/users.json`:

```json
{
  "273761843544064000": {
    "allowed": ["admin"],
    "timezone": 2
  },
  "123456789": {
    "allowed": ["llm", "music", "e621"],
    "timezone": -5
  }
}
```

| Tag | Grants | Implies |
|---|---|---|
| `llm` | Talk to Bong, chat-tier tools (memories, images, texts, web, reminders, timezone, stats, react) | — |
| `llm_fast` | No 60s cooldown + everything `llm` grants | `llm` |
| `music` | All music tools + join/leave voice | — |
| `vc_commands` | start_listening, stop_listening, wake word detection | — |
| `e621` | e621 search, subscribe, unsubscribe, DM notifications | — |
| `admin` | All admin commands + all other tags | everything |

**Owner guarantee**: `OWNER_ID` is always forced to `["admin"]` on load, even if missing from `users.json`.

### `@tags` Command

```
@tags list <user_id>          — Show a user's tags
@tags add <user_id> <tag>     — Add a tag to a user
@tags remove <user_id> <tag>  — Remove a tag from a user
```

Valid tags: `llm`, `llm_fast`, `music`, `vc_commands`, `e621`, `admin`

### DM Approval Flow

When a user without the `llm` tag DMs Bong, an approval request is sent to the owner with Discord UI buttons:

| Button | Tags Granted |
|---|---|
| Chat | `["llm"]` |
| Chat+Music | `["llm", "music"]` |
| Full Access | `["llm", "music", "vc_commands", "e621"]` |
| Admin | `["admin"]` |
| Deny | (no action, user removed from pending) |

After initial approval, `@tags` allows fine-grained adjustments.

### Tool Permission Map

| Tools | Required Tag |
|---|---|
| react, describe_image, read_text_file | `llm` |
| list_images, send_image, list_texts, send_text | `llm` |
| current_time, set_timezone, get_timezone | `llm` |
| set_reminder, cancel_reminder, list_reminders_tool | `llm` |
| bot_stats | `llm` |
| save_memory, recall_memories_by_userid, recall_memories_general, forget_memory | `llm` |
| web_search, summarize_url | `llm` |
| youtube_search | `llm` or `music` |
| join_voice, leave_voice | `music` |
| download_music, list_music, search_music, play_audio, pause_audio, resume_audio, stop_audio, skip_audio, loop_audio, music_shuffle_enabled, queue, clear_queue | `music` |
| start_listening, stop_listening | `vc_commands` |
| e621_subscribe, e621_unsubscribe, e621_list_subscriptions, e621_search | `e621` |
| shutdown | `admin` |

### Non-Tool Permission Checks

| Feature | Required Tag | Location |
|---|---|---|
| Talk to Bong (DMs and guilds) | `llm` | `bong.py:on_message`, `dm_approval.py` |
| Skip 60s cooldown | `llm_fast` or `admin` | `bong.py:on_message` |
| Wake word detection | `vc_commands` | `voice_commands.py` |
| Voice command transcription | `vc_commands` | `voice_commands.py` |
| `@llm` channel toggle | `admin` | `bong.py` |
| `@tags` (add/remove/list) | `admin` | `bong.py` |
| `@poweroff`, `@debug` | `admin` | `main.py` |

## Architecture

```
Message → BongCog.on_message()
  → is_talking_to_bong() (keyword fast-path / classifier LLM slow-path)
  → build_system_prompt() (history + voice status + memories + attachments)
  → run_tool_loop() (invoke LLM → execute tools → feed results → repeat, max 10 iterations)
  → dispatch_voice_actions() (async: join/leave/play/skip/pause/resume/stop/listen)
  → apply_reactions() (emoji reactions)
  → record_history() (rolling per-channel history with auto-summarization)
```

**Sync/async split**: LLM tools are synchronous — they write `pending_*` flags in `bong_tools`. The async cog dispatches actions after the tool loop.

### Module Map

| File | Lines | Role |
|---|---|---|
| `src/main.py` | 200 | Entry point, CLI args (`-d`, `--restore-backup`), `on_ready`, `@poweroff`/`@debug` commands, console thread |
| `src/bong.py` | 1460 | BongCog — message handling, LLM loop, voice dispatch, DM approval, e621 subscription polling, `@tags`/`@llm`/`@memories`/`@forget_user`/`@delete_memory` commands |
| `src/bong_tools.py` | 143 | Shared state hub: `pending_*` flags, library lists, `advance_queue()`, aggregated `tool_map` |
| `src/bong_state.py` | 348 | 19 LangChain `@tool` funcs: react, voice, images, texts, time, reminders, stats, shutdown |
| `src/bong_music.py` | 369 | 12 LangChain `@tool` funcs: download/search/play/pause/resume/stop/skip/loop/shuffle/queue/clear_queue |
| `src/bong_memory.py` | 121 | 4 LangChain `@tool` funcs: save/recall/recall_general/forget memory |
| `src/bong_memory_helpers.py` | 196 | ChromaDB vector store, retrieval, contradiction detection, recency boost, expiry |
| `src/bong_web.py` | 123 | 3 LangChain `@tool` funcs: web_search, youtube_search, summarize_url |
| `src/bong_e621.py` | 302 | 4 LangChain `@tool` funcs + poller: subscribe/unsubscribe/list_subscriptions/search, background DM notifications |
| `src/voice_commands.py` | 719 | BongVoiceSink, openWakeWord + Whisper pipeline, DAVE handling, per-user OWW state |
| `src/user_data.py` | 282 | Per-user settings (tags, timezone, tokens, e621 subs), `OWNER_ID`, permission checks |
| `src/dm_approval.py` | 173 | ApproveView (Discord UI buttons), DM approval flow for unknown users |
| `src/reminders.py` | 324 | Reminder persistence, absolute/relative time parsing, `PastDateError` |
| `src/persist.py` | 106 | PersistStore — dirty-flag JSON persistence, batched flush, `.bak` backup |
| `src/debug.py` | 93 | Tagged logging (console/file/error), toggle at runtime, survives hot reload |
| `src/llm_utils.py` | 5 | `_extract_response_text()` — handles string/list LLM content |
| `src/bong_song_stats.py` | 41 | Per-song play counts, top songs, total plays |
| `src/bong_cli.py` | 10 | CLI wrapper → `src.main.main()` |
| `src/train_wakeword.py` | 351 | Train custom "hey_bong" ONNX model (setup/generate/augment/train) |
| `src/setup_whisper.py` | 33 | Pre-download Whisper "small" model |
| `src/bong_utilities/manage_memory.py` | 484 | CLI: search/list/add/delete/edit/forget-user memories |
| `src/bong_utilities/dedup_memory.py` | 166 | CLI: find/merge duplicate memories via LLM |

### Dependency Graph

```
main.py → bong, bong_tools, user_data, persist, bong_song_stats, bong_memory_helpers,
          dm_approval, reminders, voice_commands, debug
bong.py → bong_tools, bong_song_stats, bong_memory_helpers, debug, dm_approval,
          persist, reminders, user_data, voice_commands, llm_utils, bong_e621
bong_tools.py → bong_music, bong_memory, bong_web, bong_state, bong_e621
bong_music.py → bong_tools, bong_song_stats
bong_memory.py → debug, bong_tools, bong_memory_helpers
bong_memory_helpers.py → bong_tools, debug, llm_utils
bong_web.py → bong_memory_helpers, llm_utils
bong_e621.py → bong_tools, debug, user_data
bong_state.py → reminders, user_data, bong_tools, bong_memory_helpers, bong_song_stats
voice_commands.py → debug, user_data, bong (lazy)
dm_approval.py → debug, persist, user_data
user_data.py → persist
reminders.py → persist
bong_song_stats.py → bong_tools, persist
persist.py, debug.py, llm_utils.py — standalone
```

### Voice Pipeline

```
Discord voice recv (48kHz stereo PCM)
  → BongVoiceSink.write()
  → DAVE filter (skip encrypted/fake/silence packets, reject peak > 12000)
  → Resample to 16kHz mono
  → Per-user OwwUserState (openWakeWord "hey_bong" at threshold 0.5)
  → VAD: frames with score < 0.5 across last 7 frames have predictions zeroed
  → Silence detection (0.8s gap triggers flush)
  → Utterance buffering (min 0.5s, max 30.0s)
  → faster-whisper "small" transcription
  → Strip wake word prefix
  → process_voice_command() → LLM pipeline → voice channel text chat
```

### e621 Subscriptions

Users with the `e621` tag can subscribe to e621 tags and receive DM notifications when new posts match. The background poller runs every 60 seconds, checks the e621 API for each subscribed tag, and DMs all subscribers with links to new posts. The first poll for a new tag is silent (recording the latest post ID without DMing), so users don't get flooded with old results.

- **Tag registry**: Global `subscriptions.json` maps tag strings to their last-seen post ID — multiple users subscribing to the same tag only polls once
- **Tag validation**: Tags are validated against the e621 API before subscribing; meta-tags (like `rating:`, `order:`, `id:`) are rejected, but artist/species/etc. prefixes are allowed
- **Cleanup**: Orphan tags (where no users remain subscribed) are cleaned up automatically
- **Authentication**: Uses `E621_USERNAME` and `E621_API_KEY` from `.env` for authenticated API access

## LLM Models

| Model | Purpose |
|---|---|
| `glm-5.1:cloud` | Main conversation + tool-calling |
| `gemma3:12b-cloud` | Classifier (talking to Bong?), image description, contradiction detection, URL summarization |
| `nomic-embed-text` | ChromaDB embeddings |
| `faster-whisper-small` | Speech-to-text |
| `hey_bong.onnx` (openWakeWord) | Wake word detection |

## LLM Tools Reference

| Tool | Description | Required Tag |
|---|---|---|
| `react` | React with an emoji | `llm` |
| `describe_image` | Describe an attached image | `llm` |
| `list_images` / `send_image` | List and send saved images | `llm` |
| `list_texts` / `send_text` | List and send saved text files | `llm` |
| `read_text_file` | Read a text file attachment | `llm` |
| `current_time` | Get current time (shows local time if timezone set) | `llm` |
| `set_timezone` / `get_timezone` | Store/retrieve user timezone | `llm` |
| `web_search` | Search the web (DuckDuckGo) | `llm` |
| `youtube_search` | Search YouTube | `llm` or `music` |
| `summarize_url` | Fetch and summarize a web page | `llm` |
| `save_memory` | Save a fact to long-term memory | `llm` |
| `recall_memories_by_userid` | Recall memories about a specific user | `llm` |
| `recall_memories_general` | Recall general memories | `llm` |
| `forget_memory` | Delete a memory | `llm` |
| `set_reminder` | Set a reminder (relative or absolute time) | `llm` |
| `cancel_reminder` | Cancel a pending reminder | `llm` |
| `list_reminders_tool` | List pending reminders | `llm` |
| `bot_stats` | Show uptime, memories, users, song stats | `llm` |
| `join_voice` / `leave_voice` | Join/leave a voice channel | `music` |
| `download_music` | Download audio from YouTube | `music` |
| `list_music` | List downloaded songs | `music` |
| `search_music` | Search YouTube and list results | `music` |
| `play_audio` | Play a song (or add to queue) | `music` |
| `pause_audio` / `resume_audio` / `stop_audio` | Playback controls | `music` |
| `skip_audio` | Skip to next song (queue > loop > shuffle) | `music` |
| `loop_audio` | Toggle loop (track or queue) | `music` |
| `music_shuffle_enabled` | Toggle random playback | `music` |
| `queue` / `clear_queue` | View or clear the song queue | `music` |
| `start_listening` / `stop_listening` | Enable/disable voice commands in current VC | `vc_commands` |
| `e621_subscribe` | Subscribe to an e621 tag for DM notifications | `e621` |
| `e621_unsubscribe` | Unsubscribe from an e621 tag | `e621` |
| `e621_list_subscriptions` | List your e621 tag subscriptions | `e621` |
| `e621_search` | Search e621 posts by tags | `e621` |
| `shutdown` | Shut down the bot | `admin` |

## Admin Commands

| Command | Description |
|---|---|
| `@poweroff` | Shut down the bot |
| `@debug` | Toggle debug logging |
| `@llm <channel>` | Toggle Bong's active presence in a channel |
| `@tags list <user_id>` | Show a user's permission tags |
| `@tags add <user_id> <tag>` | Add a permission tag to a user |
| `@tags remove <user_id> <tag>` | Remove a permission tag from a user |

## Data Files

| Path | Purpose |
|---|---|
| `bong_user_data/users.json` | Per-user settings: `allowed` tags, timezone, tokens, e621_subs, display_name |
| `bong_user_data/reminders.json` | Pending reminders |
| `bong_user_data/pending_approvals.json` | Pending DM approval user IDs |
| `bong_data/subscriptions.json` | Global e621 tag registry (tag → last seen post ID) |
| `bong_data/song_stats.json` | Per-song play counts |
| `bong_data/chroma_db/` | ChromaDB vector store for memories |
| `bong_data/Response Templates/Bong.txt` | System prompt template |
| `bong_data/Response Templates/Spoken_To_Classifier.txt` | Classifier prompt template |
| `bong_data/saved_sounds/` | Downloaded MP3 files |
| `bong_data/saved_images/` | Saved images |
| `bong_data/saved_texts/` | Saved text files |
| `bong_data/wakeword_models/hey_bong.onnx` | Custom wake word model |
| `bong_data/whisper_models/` | Cached faster-whisper models |

All data files are gitignored. `PersistStore` creates `.bak` backups on write and supports restoring from backup via `--restore-backup` or the console `restore-backup` command.

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) running with the following models:
  - `glm-5.1` — main LLM for chat and tool-calling
  - `gemma3:12b` (or cloud endpoint) — classifier, vision, contradiction detection, URL summarization
  - `nomic-embed-text` — embeddings for ChromaDB
- ffmpeg installed (for audio playback)
- A Discord bot token

### Install

```bash
git clone https://github.com/xPinkiex/Bong-Discord-Bot.git
cd Bong-Discord-Bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```
DISCORD_TOKEN=your_discord_bot_token
OLLAMA_HOST=127.0.0.1:11434
E621_USERNAME=your_e621_username
E621_API_KEY=your_e621_api_key
```

- `DISCORD_TOKEN` — Required. Bot won't start without it.
- `OLLAMA_HOST` — Optional. Defaults to `127.0.0.1:11434`.
- `E621_USERNAME` / `E621_API_KEY` — Optional. Required for e621 subscriptions and search. Uses HTTP Basic Auth against the e621 API.

The owner ID is hardcoded in `user_data.py` as `OWNER_ID` — change it to your Discord user ID.

### Train Custom Wake Word

```bash
python -m src.train_wakeword setup     # Clone repos, download models and data
python -m src.train_wakeword generate  # Generate samples with Piper TTS
python -m src.train_wakeword augment   # Augment with noise and room impulse responses
python -m src.train_wakeword train     # Train ONNX model → bong_data/wakeword_models/hey_bong.onnx
```

### Pre-download Whisper Model

```bash
python -m src.setup_whisper
```

Caches the faster-whisper "small" model to `bong_data/whisper_models/`.

### Run

```bash
source venv/bin/activate
python main.py              # Normal startup
python main.py -d           # With debug logging
python main.py --restore-backup  # Restore all .bak files on startup
```

The bot also accepts console commands while running: `reboot`, `shutdown`, `clear` (clears console), `restore-backup`.

## Reboot

`reboot` console command restarts the bot process via `os.execv`.

Changes to `main.py` require a full restart.

## Testing

```bash
pytest
```

| Test file | Tests | Covers |
|---|---|---|
| `tests/test_e621.py` | 66 | e621 tag validation, subscriptions, search, registry, polling |
| `tests/test_voice_commands.py` | 53 | Audio resampling, utterance flush, wake word detection, DAVE filtering |
| `tests/test_reminders.py` | 22 | Time delta parsing, absolute time parsing, PastDateError |
| `tests/test_cooldown.py` | 17 | Permission tag hierarchy, per-user cooldown, message dedup |

Note: `test_music.py` and `bong.py` require `discord.py`/`langchain` at runtime and are not run in the default test suite.

## Memory Management CLI

Two CLI utilities for managing memories outside of Discord:

```bash
# Search, list, add, delete, edit memories
python -m src.bong_utilities.manage_memory

# Find and merge duplicate memories
python -m src.bong_utilities.dedup_memory
```