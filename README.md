# Bong

A Discord bot with a furry protogen personality, powered by a local LLM (Ollama) with LangChain tools. Bong can chat, remember things about people, play music, set reminders, summarize URLs, and more.

## Features

- **LLM Chat** — Responds in-character as Bong, a green protogen with rainbow accents, using a hosted model via Ollama
- **Long-term Memory** — Saves facts about people, detects contradictions via LLM, recency-weighted retrieval, 180-day expiry
- **Permission Tiers** — Three tiers (admin, authorized, user) with DM approval flow via Discord buttons
- **DM Approval** — Unknown users who DM Bong trigger an approval request to the owner with User/Authorized/Admin/Deny buttons
- **Music Playback** — Download from YouTube, play in voice channels, pause/resume/skip, loop, shuffle
- **Song Queue** — Queue multiple songs, auto-advances when a track finishes, priority: queue > loop > shuffle
- **Reminders** — Set, list, and cancel reminders delivered via DM, persistent across restarts
- **Timezone Storage** — Per-user UTC offset, `current_time` shows local time automatically
- **URL Summarization** — Fetches web pages, extracts text, summarizes via LLM
- **Bot Stats** — Uptime, memory count, known users, reminders, top 3 most-played songs
- **Hot Reload** — `@reload` command refreshes code without restarting (owner only)
- **Image/Text Sharing** — Send saved images and text files in chat
- **Web Search** — DuckDuckGo search and YouTube search

## Architecture

```
main.py              Bot entry point, @reload command, on_ready init
bong.py              Discord cog — message handling, LLM loop, voice dispatch, DM approval
bong_tools.py        LangChain @tool definitions and shared state (pending_* flags)
user_data.py         Per-user settings (tier, timezone) persisted to users.json
dm_approval.py       DM approval flow with Discord button UI
reminders.py         Reminder persistence and time-delta parsing
debug.py             Logging utility
```

**How it works:**

1. A message comes in → `bong.py` sets context (user ID, voice state, etc.)
2. The LLM processes the message and may call tools
3. Tools in `bong_tools.py` are **synchronous** — they write to `pending_*` flags
4. After the LLM loop, the cog **dispatches** pending actions asynchronously (join voice, play audio, send files, deliver reminders)
5. The `after_play` callback auto-advances through the song queue

### Data Files (gitignored)

| File | Purpose |
|---|---|
| `users.json` | Per-user settings: tier, timezone |
| `reminders.json` | Pending reminders |
| `song_stats.json` | Song play counts |
| `chroma_db/` | ChromaDB vector store for memories |
| `saved_sounds/` | Downloaded mp3 files |
| `saved_images/` | Saved images |
| `saved_texts/` | Saved text files |

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) running with the `nomic-embed-text` model locally for embeddings
- A hosted `gemma3:12b-cloud` model endpoint (configured via Ollama)
- ffmpeg installed (for audio playback)
- A Discord bot token

### Install

```bash
git clone https://github.com/xPinkiex/Python-Discord-Bot.git
cd Python-Discord-Bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```
DISCORD_TOKEN=your_discord_bot_token
```

The owner ID is hardcoded in `user_data.py` as `OWNER_ID` — change it to your Discord user ID.

### Run

```bash
source venv/bin/activate
python main.py
```

## LLM Tools Reference

| Tool | Description |
|---|---|
| `react` | React with an emoji |
| `describe_image` | Describe an attached image |
| `current_time` | Get current time (shows local time if timezone set) |
| `set_timezone` / `get_timezone` | Store/retrieve user timezone |
| `web_search` | Search the web (DuckDuckGo) |
| `youtube_search` | Search YouTube |
| `summarize_url` | Fetch and summarize a web page |
| `download_music` | Download audio from YouTube |
| `play_audio` | Play a song (or add to queue if one is playing) |
| `queue` / `clear_queue` | View or clear the song queue |
| `loop_audio` | Toggle loop for current song |
| `music_shuffle_enabled` | Toggle random playback |
| `skip_audio` | Skip to next song (queue > shuffle) |
| `pause_audio` / `resume_audio` / `stop_audio` | Playback controls |
| `save_memory` | Save a fact to long-term memory |
| `recall_memories_by_userid` | Recall memories about a specific user |
| `recall_memories_general` | Recall general memories |
| `forget_memory` | Delete a memory |
| `set_reminder` / `cancel_reminder` / `list_reminders_tool` | Reminders via DM |
| `bot_stats` | Show uptime, memories, users, song stats |
| `shutdown` | Shut down the bot (authorized users only) |

## Hot Reload

Use `@reload` in Discord (owner only) to hot-reload `bong`, `bong_tools`, `dm_approval`, `reminders`, `user_data`, and `debug` without restarting. Changes to `main.py` require a full restart.

## Permissions

| Tier | Abilities |
|---|---|
| **Admin** | Shut down bot, full access to all tools |
| **Authorized** | LLM commands, music, reminders, memory |
| **User** | Basic chat access (approved via DM flow) |

Unknown users who DM Bong trigger an approval request sent to the owner with role-selection buttons.

---

*This README was generated with AI assistance.*