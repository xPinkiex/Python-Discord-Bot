# user_data.py — Per-User Permission Tags & Settings

Manages per-user permission tags, timezone, tokens, and e621 subscriptions. ~250 lines.

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

**Owner guarantee**: `OWNER_ID` (273761843544064000) always has `["admin"]` forced on load, even if `users.json` is missing.

**`has_permission(user_id, tag)`**: Returns True if the tag is in the user's `allowed` list OR if `"admin"` is in the list (admin implies all tags). `llm_fast` also implies `llm`.

## Tool Permission Map

| Tool | Required Tag |
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

## Non-Tool Permission Checks

| Feature | Required Tag | Location |
|---|---|---|
| Talk to Bong (DMs and guilds) | `llm` | `bong.py:on_message`, `dm_approval.py` |
| Skip 60s cooldown | `llm_fast` or `admin` | `bong.py:on_message` |
| Wake word detection | `vc_commands` | `voice_commands.py:434` |
| Voice transcription | `vc_commands` | `voice_commands.py:514` |
| @llm channel toggle | `admin` | `bong.py` |
| @tags (add/remove/list) | `admin` | `bong.py` |
| @poweroff, @debug | `admin` | `main.py` |
| e621 subscription polling DMs | (background, checks subscriber list) | `bong.py:_check_e621_subscriptions` |

## DM Approval Flow

When an unknown user DMs Bong:
1. If they have the `llm` tag → allowed through
2. If they're known but lack `llm` → told they don't have access
3. If they're unknown → approval request sent to owner with preset buttons:

| Button | Tags Granted |
|---|---|
| Chat | `["llm"]` |
| Chat+Music | `["llm", "music"]` |
| Full Access | `["llm", "music", "vc_commands", "e621"]` |
| Admin | `["admin"]` |
| Deny | (no action) |

After approval, `@tags` command allows fine-grained add/remove.

## Key Functions

| Function | Purpose |
|---|---|
| `has_permission(user_id, tag)` | Check if user has a tag (`admin` implies all, `llm_fast` implies `llm`) |
| `is_admin(user_id)` | Shorthand for `has_permission(user_id, "admin")` (also checks OWNER_ID) |
| `is_known(user_id)` | True if user exists in data (has any entry) |
| `get_permissions(user_id)` | Get user's tag list |
| `add_permission(user_id, tag)` | Add a tag to a user |
| `remove_permission(user_id, tag)` | Remove a tag from a user |
| `set_permissions(user_id, tags)` | Set user's tags to exactly the given list |

## Legacy Data

If `users.json` contains entries with the old `"tier"` field, `load_users()` converts string values (like `"admin"`) to `{"allowed": []}` (empty tags). You'll need to manually assign tags using `@tags add` or by editing `users.json` directly. The OWNER_ID is always forced to `["admin"]` on load regardless of file contents.

## @tags Command (admin-only)

```
@tags list <user_id>        — Show user's tags
@tags add <user_id> <tag>   — Add a tag to a user
@tags remove <user_id> <tag> — Remove a tag from a user
```

Valid tags: `llm`, `llm_fast`, `music`, `vc_commands`, `e621`, `admin`