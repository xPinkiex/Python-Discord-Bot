# bong_e621 — e621 Tag Subscriptions & Search

e621 tag subscription system with DM notifications and on-demand search. ~290 lines.

## Architecture

```
User subscribes via LLM tool
  → add_subscription() validates tags against e621 API (1s throttle per tag)
  → stores in user_data.e621_subs + global tag_registry
  → background poll loop (every 60s)
      → for each tag in tag_registry:
          → get_new_posts(tags, last_post_id)
          → DM all subscribers with new posts
          → 1s sleep between tags
      → save_subscriptions()
```

**Sync/async split**: LangChain `@tool` functions are synchronous — they call `user_data` helpers and `tag_registry` directly. The async cog (`bong.py`) runs `_check_e621_subscriptions()` as a background task that polls and DMs.

## Dependency Graph

```
bong_e621.py → persist, debug, user_data, requests (langchain_core.tools for @tool)
bong.py → bong_e621 (background task: _check_e621_subscriptions, cog_unload cancels e621_task)
bong_tools.py → bong_e621.tools (merged into tools list)
user_data.py → persist (stores e621_subs per user)
```

## Key Constants

| Constant | Value | Purpose |
|---|---|---|
| `E621_POSTS_URL` | `https://e621.net/posts.json` | Posts search endpoint |
| `E621_TAGS_URL` | `https://e621.net/tags.json` | Tag validation endpoint |
| `E621_POLL_INTERVAL` | `60` | Seconds between subscription polls |
| `E621_USER_AGENT` | `"BongBot/1.0 (by AkazuEve on e621)"` | Required by e621 API |
| `_STORE_PATH` | `bong_data/subscriptions.json` | Tag registry persist path |
| `_TAG_CATEGORIES` | `{"artist":1, "copyright":3, "character":4, "species":5, "lore":8, "meta":7}` | Prefix → e621 category ID mapping |

## Data Layout

**`bong_data/subscriptions.json`** — Global tag registry (one entry per unique tag string):
```json
{
  "protogen": 6451974,
  "species:wolf": 5100232
}
```
Values are `last_post_id` (int) or `null` for first poll. File is managed via `persist.PersistStore` with dirty-flag flushing.

**`bong_user_data/users.json`** — Per-user subscriptions in each user's `"e621_subs"` list:
```json
{
  "273761843544064000": {
    "allowed": ["admin"],
    "timezone": 2,
    "e621_subs": ["protogen", "species:wolf"]
  }
}
```
Managed by `user_data.py` helpers: `get_e621_subs`, `add_e621_sub`, `remove_e621_sub`, `get_all_e621_subscribers`.

**Old format migration**: `load_subscriptions()` handles both the old format (`{"tag_registry": {...}, "subscriptions": [...]}`) and the current flat dict. The old `subscriptions` list is discarded on load.

## 4 LangChain Tools

| Tool | Args | Purpose |
|---|---|---|
| `e621_subscribe` | `tags: str` | Subscribe to tag search, validate against e621 API, get DMs for new posts. Requires `e621` permission tag. |
| `e621_unsubscribe` | `tags: str` | Unsubscribe from tag search (must match exactly). Requires `e621` permission tag. |
| `e621_list_subscriptions` | *(none)* | List user's active e621 subscriptions |
| `e621_search` | `tags: str` | One-shot search, returns up to 5 posts with links |

Tools get `current_user_id` from `bong_tools` via lazy import (avoids circular import: `bong_e621 → bong_tools → bong_e621`).

## Tag Validation

When subscribing, each tag part is validated against the e621 `/tags.json` endpoint:

1. **Split**: `tags` string is split on spaces. Multi-word queries like `"protogen solo"` validate each part separately.
2. **Strip `~`**: The OR operator `~` is stripped before validation (`~protogen` → `protogen`).
3. **Meta-tags skipped**: Tags starting with `rating:`, `order:`, `id:`, `score:`, `favcount:`, `filesize:`, `duration:`, `date:`, `-` (negation), or containing `*` (wildcard) are not validated — they're e621 search operators, not content tags.
4. **Prefix tags**: `artist:`, `species:`, `character:`, `copyright:`, `lore:`, `meta:` are content tags with a category. The prefix is stripped and `search[category]` is set in the validation request.
5. **Graceful failure**: If `_validate_tag` gets an API error (429, timeout, etc.), it returns `(True, None)` — doesn't block the subscription.

Validation warns but doesn't block: unknown tags and zero-post tags produce warnings appended to the success message.

## Polling Flow (`bong.py:_check_e621_subscriptions`)

```
1. await bot.wait_until_ready()
2. load_subscriptions() from disk
3. Loop:
   a. dirty = False
   b. For each (tags, last_id) in tag_registry:
      - try:
        - get_new_posts(tags, last_id) → API call
        - If new_id != last_id: update tag_registry[tags] = new_id, dirty = True
        - If new_posts: find subscribers via user_data.get_all_e621_subscribers(tags)
          → DM each subscriber with post links
        - Handle DM errors per-user (Forbidden → skip, other → log)
      - except: log and continue to next tag
      - await asyncio.sleep(1)  (1s between tags)
   c. If dirty: save_subscriptions()
   d. await asyncio.sleep(60)  (1 minute between polls)
```

**Silent first poll**: When `last_post_id` is `None` (brand new tag), `get_new_posts` records the latest post ID without returning any posts. The user won't get spammed with old content — only truly new posts trigger DMs.

**New user inherits existing tag**: If user B subscribes to a tag user A already subscribes to, the `last_post_id` is already set. User B starts receiving DMs only for posts after their subscription time.

## Key Functions

| Function | File:Line | Purpose |
|---|---|---|
| `add_subscription` | `bong_e621.py:109` | Validate tags, add to user subs + global registry, return message |
| `remove_subscription` | `bong_e621.py:143` | Remove from user subs, prune registry if no subscribers left |
| `list_subscriptions` | `bong_e621.py:158` | Format user's subscription list |
| `get_new_posts` | `bong_e621.py:208` | Poll e621 for posts newer than `last_post_id` |
| `search_e621_posts` | `bong_e621.py:187` | One-shot search, format results with links |
| `_e621_request` | `bong_e621.py:168` | HTTP GET with BasicAuth, User-Agent, rate-limit handling |
| `_is_meta_tag` | `bong_e621.py:61` | Check if tag is a search operator (not a content tag) |
| `_split_tag_for_validation` | `bong_e621.py:82` | Parse tag into (name, category) tuples for validation |
| `_validate_tag` | `bong_e621.py:92` | Query `/tags.json` to check if tag exists |
| `load_subscriptions` | `bong_e621.py:44` | Load tag registry from disk, handle old format migration |
| `save_subscriptions` | `bong_e621.py:57` | Mark persist store dirty for flush |
| `cleanup_tag_registry` | `bong_e621.py:233` | Remove tags with zero subscribers |
| `_check_e621_subscriptions` | `bong.py:1163` | Async background task: poll tags, DM subscribers |

## API Details

| Detail | Value |
|---|---|
| Auth | `HTTPBasicAuth(E621_USERNAME, E621_API_KEY)` from `.env` |
| User-Agent | `BongBot/1.0 (by AkazuEve on e621)` (required by e621) |
| Rate limit handling | 1s `time.sleep()` between validation calls; 1s `asyncio.sleep()` between poll calls; 429 returns `None` (graceful skip) |
| Timeout | 15s per request |

## Design Decisions

- **Global tag registry**: One `last_post_id` per tag string, regardless of how many users subscribe. Eliminates duplicate API calls.
- **Per-user subs in `users.json`**: Uses existing `user_data` infrastructure. No separate per-user store.
- **Silent first poll**: New tags record the latest post ID without DMing. Prevents spam on fresh subscriptions.
- **Validate but don't block**: Unknown/zero-post tags produce warnings but the subscription still goes through. The tag might be valid but new, or the API might be flaky.
- **Permission gating**: `e621_subscribe` and `e621_unsubscribe` require `has_permission(user_id, "e621")`. `e621_list_subscriptions` and `e621_search` also require the `e621` tag. The `admin` tag implies all other tags.
- **Lazy `bong_tools` import**: `@tool` functions import `bong_tools` inside the function body to avoid circular import (`bong_tools → bong_e621.tools → bong_tools`).
- **Persist dirty-flag batching**: `save_subscriptions()` calls `_store.mark_dirty()` — actual disk write happens in the 60s persist flush cycle, not per-subscription.
- **Tag registry cleanup**: `cleanup_tag_registry()` removes orphan tags (no subscribers left). Also called implicitly by `remove_subscription()` when the last subscriber leaves.
- **`~` is not a meta-tag**: The tilde OR operator is stripped via `lstrip("~")` but not rejected — it's valid e621 search syntax.

## Tests

**`tests/test_e621.py`** — 66 tests:

| Class | Tests | Covers |
|---|---|---|
| `TestIsMetaTag` | 13 | `rating:`, `order:`, `id:`, `score:`, `-`, `~`, `*`, normal tags, `artist:`, `species:`, `favcount:`, `date:` |
| `TestSplitTagForValidation` | 12 | Prefix splitting (artist/species/character/copyright/lore/meta), normal tags, meta-tag skip, negation skip, wildcard skip, `~` stripping |
| `TestValidateTag` | 6 | Tag exists (with/without category), not found (dict/list), API failure, zero posts |
| `TestAddSubscription` | 9 | Basic add, normalization, duplicate, multi-user, empty tags, warnings (unknown/zero-post), meta-tag skip, artist prefix |
| `TestRemoveSubscription` | 4 | Remove existing, nonexistent, keeps tag if other subscribers, only affects user |
| `TestListSubscriptions` | 3 | Empty, with subs, only shows own |
| `TestGetAllSubscribers` | 3 | Finds subscribers, none, different tags |
| `TestGetNewPosts` | 7 | Filter by ID, silent first poll, no new posts, API failure, empty response, first poll empty, uses posts endpoint |
| `TestSearchFormatting` | 3 | Format results, no results, API failure |
| `TestCleanupTagRegistry` | 2 | Remove orphans, keep active |
| `TestTagRegistryGlobalState` | 2 | Shared registry, new user inherits existing |
| `TestLoadSubscriptionsMigration` | 2 | Flat dict load, old format migration |

Tests use `autouse` fixture with `tmp_path` stores — no real data files are touched.