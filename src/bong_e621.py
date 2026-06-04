# bong_e621.py — e621 tag subscriptions and search for Bong
#
# Users can subscribe to e621 tag searches and get DM notifications
# when new posts matching their tags appear. Tag subscriptions are global —
# multiple users subscribing to the same tag only triggers one API poll.
# Also provides an on-demand search tool.
#
# Data layout:
#   bong_data/subscriptions.json — global tag registry: {"protogen": 6451974, ...}
#   bong_user_data/users.json — per-user e621_subs: {"273761843544064000": {"e621_subs": ["protogen"], ...}, ...}

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BONG_DATA = PROJECT_ROOT / "bong_data"

import persist
import debug
import user_data

import requests
from requests.auth import HTTPBasicAuth

E621_POSTS_URL = "https://e621.net/posts.json"
E621_TAGS_URL = "https://e621.net/tags.json"
E621_USERNAME = os.getenv("E621_USERNAME", "")
E621_API_KEY = os.getenv("E621_API_KEY", "")
E621_USER_AGENT = "BongBot/1.0 (by AkazuEve on e621)"

E621_POLL_INTERVAL = 60  # 1 minute

_STORE_PATH = BONG_DATA / "subscriptions.json"
_store = persist.PersistStore(_STORE_PATH, default={})
persist.register(_store)

tag_registry: dict[str, int | None] = {}


def load_subscriptions():
    global tag_registry
    _store.load()
    data = _store.data
    if isinstance(data, dict) and "tag_registry" in data:
        tag_registry = data["tag_registry"]
    elif isinstance(data, dict):
        tag_registry = dict(data)
    else:
        tag_registry = {}
    _store.data = tag_registry


def save_subscriptions():
    _store.mark_dirty()


def _is_meta_tag(tag: str) -> bool:
    prefix_skip = ("rating:", "order:", "id:", "score:", "favcount:", "filesize:", "duration:", "date:")
    if tag.startswith(prefix_skip):
        return True
    if tag.startswith("-"):
        return True
    if "*" in tag:
        return True
    return False


_TAG_CATEGORIES = {
    "artist": 1,
    "copyright": 3,
    "character": 4,
    "species": 5,
    "lore": 8,
    "meta": 7,
}


def _split_tag_for_validation(tag: str) -> list[tuple[str, int | None]]:
    raw = tag.lstrip("~")
    if _is_meta_tag(raw):
        return []
    for prefix, category in _TAG_CATEGORIES.items():
        if raw.startswith(prefix + ":"):
            return [(raw[len(prefix) + 1:], category)]
    return [(raw, None)]


def _validate_tag(tag_name: str, category: int | None = None) -> tuple[bool, int | None]:
    params = {"search[name_matches]": tag_name, "limit": 1}
    if category is not None:
        params["search[category]"] = category
    result = _e621_request(E621_TAGS_URL, params)
    if result is None:
        return True, None
    tags = result.get("tags", [])
    if isinstance(tags, dict) and not tags:
        return False, None
    if isinstance(tags, list) and tags:
        found = tags[0]
        post_count = found.get("post_count", 0)
        return True, post_count
    return False, None


def add_subscription(user_id: int, tags: str) -> str:
    tags = tags.strip().lower()
    if not tags:
        return "Tags cannot be empty."

    existing = user_data.get_e621_subs(user_id)
    if tags in existing:
        return f"You're already subscribed to '{tags}'."

    warnings = []
    parts = tags.split()
    for part in parts:
        validations = _split_tag_for_validation(part)
        for tag_name, category in validations:
            time.sleep(1)
            exists, post_count = _validate_tag(tag_name, category)
            if not exists:
                warnings.append(f"Tag '{tag_name}' not found on e621")
            elif post_count == 0:
                warnings.append(f"Tag '{tag_name}' exists but has no posts yet")

    user_data.add_e621_sub(user_id, tags)
    if tags not in tag_registry:
        tag_registry[tags] = None
        debug.log("e621", f"New tag registered: '{tags}'")
    _store.mark_dirty()
    debug.log("e621", f"User {user_id} subscribed to '{tags}'")

    msg = f"Subscribed to e621 tag search: '{tags}'. You'll get DMs when new posts appear."
    if warnings:
        msg += "\n\nWarnings:\n" + "\n".join(f"  - {w}" for w in warnings)
    return msg


def remove_subscription(user_id: int, tags: str) -> str:
    tags = tags.strip().lower()
    found = user_data.remove_e621_sub(user_id, tags)
    if not found:
        return f"No subscription found for '{tags}'."

    still_subscribed = user_data.get_all_e621_subscribers(tags)
    if not still_subscribed and tags in tag_registry:
        del tag_registry[tags]
        debug.log("e621", f"Tag '{tags}' removed from registry (no subscribers left)")
    _store.mark_dirty()
    debug.log("e621", f"User {user_id} unsubscribed from '{tags}'")
    return f"Unsubscribed from e621 tag search: '{tags}'."


def list_subscriptions(user_id: int) -> str:
    subs = user_data.get_e621_subs(user_id)
    if not subs:
        return "You have no e621 subscriptions."
    lines = []
    for i, tags in enumerate(subs, 1):
        lines.append(f"  {i}. {tags}")
    return "Your e621 subscriptions:\n" + "\n".join(lines)


def _e621_request(url: str, params: dict) -> dict | None:
    headers = {"User-Agent": E621_USER_AGENT}
    auth = HTTPBasicAuth(E621_USERNAME, E621_API_KEY) if (E621_USERNAME and E621_API_KEY) else None

    try:
        debug.log("e621", f"Request: {url} params={params}")
        resp = requests.get(url, params=params, headers=headers, auth=auth, timeout=15)
        if resp.status_code == 429:
            debug.log("e621", "Rate limited, backing off")
            return None
        resp.raise_for_status()
        data = resp.json()
        debug.log("e621", f"Response: {resp.status_code} posts={len(data.get('posts', []))}")
        return data
    except Exception as e:
        debug.log("e621", f"API request failed: {e}")
        return None


def search_e621_posts(tags: str, limit: int = 5) -> str:
    debug.log("e621", f"Searching for '{tags}' (limit={limit})")
    result = _e621_request(E621_POSTS_URL, {"tags": tags, "limit": limit})
    if result is None:
        return "Could not reach e621. Try again later."

    posts = result.get("posts", [])
    if not posts:
        return f"No results found for '{tags}'."

    lines = []
    for post in posts:
        post_id = post.get("id", "?")
        score = post.get("score", {}).get("total", 0)
        rating = post.get("rating", "?")
        url = f"https://e621.net/posts/{post_id}"
        lines.append(f"  #{post_id} [score:{score} rating:{rating}] {url}")

    return "\n".join(lines)


def get_new_posts(tags: str, last_post_id: int | None) -> tuple[list[dict], int | None]:
    debug.log("e621", f"Polling '{tags}' since post #{last_post_id}")
    result = _e621_request(E621_POSTS_URL, {"tags": tags, "limit": 5})
    if result is None:
        return [], last_post_id

    posts = result.get("posts", [])
    if not posts:
        return [], last_post_id

    if last_post_id is None:
        newest_id = max(p["id"] for p in posts)
        debug.log("e621", f"Silent first poll for '{tags}': recorded post #{newest_id}")
        return [], newest_id

    new_posts = [p for p in posts if p["id"] > last_post_id]
    if new_posts:
        newest_id = max(p["id"] for p in new_posts)
        debug.log("e621", f"Found {len(new_posts)} new post(s) for '{tags}' (up to #{newest_id})")
        return new_posts, newest_id

    debug.log("e621", f"No new posts for '{tags}' (latest #{max(p['id'] for p in posts)})")
    return [], last_post_id


def cleanup_tag_registry():
    for tags in list(tag_registry):
        if not user_data.get_all_e621_subscribers(tags):
            del tag_registry[tags]
            debug.log("e621", f"Cleaned up orphan tag: '{tags}'")


# --- LangChain tools ---

from langchain_core.tools import tool


@tool
def e621_subscribe(tags: str) -> str:
    """Subscribe to an e621 tag search. You will receive DM notifications when new posts matching these tags appear. Tags are validated against e621's tag database. Only authorized users can use this.
    Args:
        tags: The e621 tag search query (e.g. 'protogen solo', 'species:wolf rating:s'). Multiple tags are AND-filtered.
    """
    import bong_tools as _bt
    user_id = _bt.current_user_id
    if not user_id:
        return "Cannot determine your user ID."
    if not user_data.has_permission(user_id, "e621"):
        return "You don't have access to e621 subscriptions. Ask an admin to grant you the e621 tag."
    return add_subscription(user_id, tags)


@tool
def e621_unsubscribe(tags: str) -> str:
    """Unsubscribe from an e621 tag search. Stops DM notifications for these tags. Requires the e621 permission tag.
    Args:
        tags: The tag search query to unsubscribe from (must match exactly).
    """
    import bong_tools as _bt
    user_id = _bt.current_user_id
    if not user_id:
        return "Cannot determine your user ID."
    if not user_data.has_permission(user_id, "e621"):
        return "You don't have access to e621 subscriptions. Ask an admin to grant you the e621 tag."
    return remove_subscription(user_id, tags)


@tool
def e621_list_subscriptions() -> str:
    """List all your active e621 tag subscriptions. Requires the e621 permission tag."""
    import bong_tools as _bt
    user_id = _bt.current_user_id
    if not user_id:
        return "Cannot determine your user ID."
    if not user_data.has_permission(user_id, "e621"):
        return "You don't have access to e621. Ask an admin to grant you the e621 tag."
    return list_subscriptions(user_id)


@tool
def e621_search(tags: str) -> str:
    """Search e621 for posts matching tags. Returns up to 5 results with links. Requires the e621 permission tag.
    Args:
        tags: The e621 tag search query (e.g. 'protogen cute', 'artist:name'). Multiple tags are AND-filtered.
    """
    import bong_tools as _bt
    user_id = _bt.current_user_id
    if not user_id:
        return "Cannot determine your user ID."
    if not user_data.has_permission(user_id, "e621"):
        return "You don't have access to e621. Ask an admin to grant you the e621 tag."
    return search_e621_posts(tags, limit=5)


tools = [e621_subscribe, e621_unsubscribe, e621_list_subscriptions, e621_search]