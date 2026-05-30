# reminders.py — Persistent reminder system for Bong
#
# Users can set reminders via the `set_reminder` LLM tool. Reminders are stored
# in reminders.json and checked every 30 seconds by a background task in the cog.
# When a reminder is due, Bong DMs the user.

import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict

_REMINDERS_FILE = Path(__file__).parent / "reminders.json"

# In-memory list of active reminders
reminders: list[dict] = []

# Pending reminders that the cog should deliver (set by the tool, read by the cog)
pending_reminders: list[dict] = []


def load_reminders():
    """Load reminders from disk, removing any that are already past due."""
    global reminders
    reminders = []
    try:
        if _REMINDERS_FILE.exists():
            with open(_REMINDERS_FILE, "r") as f:
                all_reminders = json.load(f)
            now = datetime.now().timestamp()
            reminders = [r for r in all_reminders if r.get("due_at", 0) > now]
            # Save back to clean up expired ones
            save_reminders()
    except Exception:
        reminders = []


def save_reminders():
    """Persist reminders to disk."""
    try:
        with open(_REMINDERS_FILE, "w") as f:
            json.dump(reminders, f, indent=2)
    except Exception:
        pass


def add_reminder(user_id: int, username: str, message: str, due_at: float) -> dict:
    """Add a reminder and persist it. Returns the reminder dict."""
    reminder = {
        "user_id": user_id,
        "username": username,
        "message": message,
        "due_at": due_at,
    }
    reminders.append(reminder)
    reminders.sort(key=lambda r: r["due_at"])
    save_reminders()
    return reminder


def cancel_reminder(user_id: int, query: str = "") -> str:
    """Cancel the most recent reminder for a user, or one matching a query."""
    user_reminders = [r for r in reminders if r["user_id"] == user_id]
    if not user_reminders:
        return "No reminders found to cancel."

    if query:
        query_lower = query.lower()
        matching = [r for r in user_reminders if query_lower in r["message"].lower()]
        if not matching:
            return f"No reminders matching '{query}' found."
        reminder = matching[-1]
    else:
        reminder = user_reminders[-1]

    reminders.remove(reminder)
    save_reminders()
    due_str = datetime.fromtimestamp(reminder["due_at"]).strftime("%H:%M on %Y-%m-%d")
    return f"Cancelled reminder: '{reminder['message']}' (was due at {due_str})"


def list_reminders(user_id: int) -> str:
    """List all pending reminders for a user."""
    user_reminders = [r for r in reminders if r["user_id"] == user_id]
    if not user_reminders:
        return "No pending reminders."
    lines = []
    now = datetime.now().timestamp()
    for i, r in enumerate(user_reminders, 1):
        delta = r["due_at"] - now
        if delta > 0:
            when = _format_delta(delta)
            due_str = f"in {when}"
        else:
            due_str = "now"
        lines.append(f"  {i}. {r['message']} ({due_str})")
    return "\n".join(lines)


def _format_delta(seconds: float) -> str:
    """Format a time delta in seconds to a human-readable string."""
    delta = timedelta(seconds=seconds)
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if not parts:
        parts.append(f"{int(secs)} seconds")
    return ", ".join(parts)


def parse_time_delta(text: str) -> float | None:
    """Parse a human-readable time delta like '2 hours', '30 minutes', '1 day' into seconds.

    Supports combinations like '1 hour 30 minutes' and common abbreviations.
    Returns None if the text can't be parsed.
    """
    import re
    text = text.lower().strip()

    # Map of unit names and abbreviations to seconds
    units = {
        "second": 1, "seconds": 1, "sec": 1, "secs": 1, "s": 1,
        "minute": 60, "minutes": 60, "min": 60, "mins": 60, "m": 60,
        "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600, "h": 3600,
        "day": 86400, "days": 86400, "d": 86400,
        "week": 604800, "weeks": 604800, "w": 604800,
    }

    # Match patterns like "2 hours", "30m", "1 day 2 hours"
    pattern = r"(\d+(?:\.\d+)?)\s*(" + "|".join(units.keys()) + r")\b"
    matches = re.findall(pattern, text)

    if not matches:
        return None

    total = 0.0
    for value, unit in matches:
        if unit in units:
            total += float(value) * units[unit]

    return total if total > 0 else None