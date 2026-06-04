# user_data.py — Per-user settings persisted to users.json
#
# Stores permission tags, timezone, and other per-user data.
# The owner (Eve) is always guaranteed admin even if the file is missing.
#
# Permission tags: "llm", "llm_fast", "music", "vc_commands", "e621", "admin"
#   "admin" implies all other tags.
#   "llm_fast" implies "llm" for permission checks but also skips cooldown.
#
# users.json format:
#   {"273761843544064000": {"allowed": ["admin"], "timezone": 2}, ...}
#
# Migration: old "tier" field is converted to empty "allowed" on load.
# OWNER_ID is always guaranteed ["admin"].

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BONG_DATA = PROJECT_ROOT / "bong_data"
BONG_USER_DATA = PROJECT_ROOT / "bong_user_data"

import persist

_STORE_PATH = BONG_USER_DATA / "users.json"
_store = persist.PersistStore(_STORE_PATH, default={})
persist.register(_store)

# In-memory data: user_id -> dict of settings (alias to _store.data after load)
_user_data: dict[int, dict] = {}

# The owner who receives approval requests — always admin
OWNER_ID = 273761843544064000

# Valid permission tags
VALID_TAGS = {"llm", "llm_fast", "music", "vc_commands", "e621", "admin"}


def load_users():
    """Load user data from disk. Owner is always guaranteed admin."""
    global _user_data
    _store.load()
    raw = dict(_store.data)
    converted = {}
    for uid_str, value in raw.items():
        uid = int(uid_str)
        if isinstance(value, str):
            converted[uid] = {"allowed": []}
        else:
            converted[uid] = dict(value)
    converted.setdefault(OWNER_ID, {})["allowed"] = ["admin"]
    _store.data = converted
    _store.mark_dirty()
    _user_data = _store.data


def has_permission(user_id: int, tag: str) -> bool:
    """Check if a user has a specific permission tag.
    
    admin implies all other tags. llm_fast implies llm.
    Returns False if user is unknown or doesn't have the tag.
    """
    entry = _user_data.get(user_id)
    if entry is None:
        return False
    allowed = entry.get("allowed", [])
    if "admin" in allowed:
        return True
    if tag == "llm" and "llm_fast" in allowed:
        return True
    return tag in allowed


def is_admin(user_id: int) -> bool:
    """Check if a user has the admin tag. OWNER_ID is always admin."""
    if user_id == OWNER_ID:
        return True
    return has_permission(user_id, "admin")


def get_permissions(user_id: int) -> list[str]:
    """Get a user's permission tags. Returns empty list if unknown."""
    entry = _user_data.get(user_id)
    if entry is None:
        return []
    return entry.get("allowed", [])


def add_permission(user_id: int, tag: str):
    """Add a permission tag for a user. Creates entry if user doesn't exist."""
    if tag not in VALID_TAGS:
        return
    entry = _user_data.setdefault(user_id, {})
    allowed = entry.get("allowed", [])
    if tag not in allowed:
        allowed.append(tag)
        entry["allowed"] = allowed
        _store.mark_dirty()


def remove_permission(user_id: int, tag: str):
    """Remove a permission tag from a user. Returns True if tag was found and removed."""
    entry = _user_data.get(user_id)
    if entry is None:
        return False
    allowed = entry.get("allowed", [])
    if tag in allowed:
        allowed.remove(tag)
        entry["allowed"] = allowed
        _store.mark_dirty()
        return True
    return False


def set_permissions(user_id: int, tags: list[str]):
    """Set a user's permission tags to exactly the given list. Creates entry if needed."""
    entry = _user_data.setdefault(user_id, {})
    entry["allowed"] = [t for t in tags if t in VALID_TAGS]
    _store.mark_dirty()


def get_timezone(user_id: int) -> float | None:
    """Get the UTC offset for a user, or None if not set."""
    entry = _user_data.get(user_id)
    if entry is None:
        return None
    return entry.get("timezone")


def set_timezone(user_id: int, offset: float):
    """Set the UTC offset for a user and persist."""
    _user_data.setdefault(user_id, {})["timezone"] = offset
    _store.mark_dirty()


def add_tokens(user_id: int, count: int, display_name: str = ""):
    """Add token count to a user's lifetime total and persist."""
    entry = _user_data.setdefault(user_id, {})
    entry["tokens"] = entry.get("tokens", 0) + count
    if display_name:
        entry["display_name"] = display_name
    _store.mark_dirty()


def get_tokens(user_id: int) -> int:
    """Get the lifetime token total for a user, or 0."""
    entry = _user_data.get(user_id)
    if entry is None:
        return 0
    return entry.get("tokens", 0)


def get_top_users_by_tokens(n: int = 3) -> list[tuple[int, str, int]]:
    """Return top N users by token usage as (user_id, display_name, count)."""
    sorted_users = sorted(
        ((uid, d.get("display_name", ""), d.get("tokens", 0)) for uid, d in _user_data.items() if d.get("tokens", 0) > 0),
        key=lambda x: x[2], reverse=True,
    )
    return sorted_users[:n]


# ---- Timezone name / city lookup ----

_TZ_ALIASES: dict[str, float] = {
    # Common abbreviations
    "utc": 0, "gmt": 0, "est": -5, "edt": -4, "cst": -6, "cdt": -5,
    "mst": -7, "mdt": -6, "pst": -8, "pdt": -7,
    "cet": 1, "cest": 2, "eet": 2, "eest": 3,
    "aest": 10, "acst": 9.5, "awst": 8,
    "nzst": 12, "nzdt": 13,
    "ist": 5.5, "jst": 9, "kst": 9, "cst_china": 8, "hkt": 8, "sgt": 8,
    # Cities
    "new york": -5, "los angeles": -8, "chicago": -6, "denver": -7,
    "london": 0, "paris": 1, "berlin": 1, "amsterdam": 1, "madrid": 1,
    "rome": 1, "moscow": 3, "istanbul": 3, "dubai": 4, "mumbai": 5.5,
    "delhi": 5.5, "kolkata": 5.5, "bangkok": 7, "jakarta": 7,
    "shanghai": 8, "beijing": 8, "singapore": 8, "hong kong": 8,
    "tokyo": 9, "seoul": 9, "sydney": 11, "melbourne": 11,
    "auckland": 13, "honolulu": -10, "anchorage": -9,
    "sao paulo": -3, "buenos aires": -3, "mexico city": -6,
    "toronto": -5, "vancouver": -8, "calgary": -7,
}

_TZ_REGEX = __import__("re").compile(
    r"""
    ^\s*
    (?:UTC|GMT)?                              # optional UTC/GMT prefix
    \s*
    ([+-]?)                                    # optional sign
    \s*
    (?:
        (\d{1,2})                               # hours
        (?::(\d{2}))?                           # optional :minutes
        |
        (\d{1,2}(?:\.\d+)?)                    # decimal hours e.g. 5.5
    )
    \s*$
    """,
    __import__("re").VERBOSE,
)


def parse_timezone(text: str) -> float | None:
    """Parse a timezone string into a UTC offset in hours.

    Accepts:
      - Named zones: 'EST', 'PST', 'CET', 'New York', 'London', etc.
      - Offsets: 'UTC+2', 'GMT-5', '+2', '-7', '+5:30', '5.5'
    Returns the offset as a float (e.g. 2.0, -5.0, 5.5) or None if unparseable.
    """
    import re

    key = text.strip().lower()
    if key in _TZ_ALIASES:
        return _TZ_ALIASES[key]

    m = _TZ_REGEX.match(text)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        if m.group(4) is not None:
            hours = float(m.group(4))
        elif m.group(2) is not None:
            hours = int(m.group(2))
            minutes = int(m.group(3)) if m.group(3) else 0
            hours += minutes / 60.0
        else:
            return None
        result = sign * hours
        if -12 <= result <= 14:
            return result
        return None

    return None


def known_user_count() -> int:
    return len(_user_data)


# ---- e621 subscription helpers ----

def get_e621_subs(user_id: int) -> list[str]:
    """Get a user's e621 tag subscriptions as a list of tag strings."""
    entry = _user_data.get(user_id)
    if entry is None:
        return []
    return entry.get("e621_subs", [])


def add_e621_sub(user_id: int, tags: str):
    """Add an e621 tag subscription for a user. Tags should be pre-normalized (lowercase, stripped)."""
    entry = _user_data.setdefault(user_id, {})
    subs = entry.get("e621_subs", [])
    if tags not in subs:
        subs.append(tags)
        entry["e621_subs"] = subs
        _store.mark_dirty()


def remove_e621_sub(user_id: int, tags: str):
    """Remove an e621 tag subscription for a user. Returns True if found and removed."""
    entry = _user_data.get(user_id)
    if entry is None:
        return False
    subs = entry.get("e621_subs", [])
    if tags in subs:
        subs.remove(tags)
        entry["e621_subs"] = subs
        _store.mark_dirty()
        return True
    return False


def get_all_e621_subscribers(tags: str) -> list[int]:
    """Get all user IDs subscribed to a given tag string."""
    result = []
    for uid, entry in _user_data.items():
        if tags in entry.get("e621_subs", []):
            result.append(uid)
    return result


