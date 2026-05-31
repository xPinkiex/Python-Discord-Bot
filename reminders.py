# reminders.py — Persistent reminder system for Bong
#
# Users can set reminders via the `set_reminder` LLM tool. Reminders are stored
# in reminders.json and checked every 30 seconds by a background task in the cog.
# When a reminder is due, Bong DMs the user.

import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional
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


_DAY_NAMES = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

_MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def parse_absolute_time(text: str, utc_offset: Optional[float] = None) -> Optional[float]:
    """Parse an absolute date/time expression into a UTC timestamp.

    Accepts expressions like:
      - "tomorrow at 3pm"
      - "friday at 12:00"
      - "next monday at 9am"
      - "june 5 at 3pm"
      - "6/5/2026 at 15:00"
      - "2026-06-05 15:00"
      - "tomorrow 8am"
      - "today at 5pm"

    If utc_offset is provided, the input time is interpreted in that timezone.
    If None, UTC is assumed.

    Returns the UTC timestamp, or None if the text can't be parsed.
    """
    import re

    text = text.lower().strip().rstrip(".")

    # Determine the user's "now" in their timezone
    if utc_offset is not None:
        now_local = datetime.utcnow() + timedelta(hours=utc_offset)
    else:
        now_local = datetime.utcnow()

    # Parse the time portion — supports "3pm", "3:00pm", "15:00", "3:00 pm", "12am", etc.
    time_patterns = [
        r'(?:(?:at\s+)?|^)(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b',
        r'(?:(?:at\s+)?|^)(\d{1,2}):(\d{2})\b',
    ]

    hour = None
    minute = 0

    for pattern in time_patterns:
        match = re.search(pattern, text)
        if match and hour is None:
            groups = match.groups()
            if len(groups) == 3 and groups[2] in ("am", "pm"):
                hour = int(groups[0])
                minute = int(groups[1]) if groups[1] else 0
                if groups[2] == "pm" and hour != 12:
                    hour += 12
                elif groups[2] == "am" and hour == 12:
                    hour = 0
            elif len(groups) == 2 and groups[1] is not None:
                hour = int(groups[0])
                minute = int(groups[1])
            break

    # Determine the date
    target_date = None

    # "today"
    if re.search(r'\btoday\b', text):
        target_date = now_local.date()

    # "tomorrow"
    elif re.search(r'\btomorrow\b', text):
        target_date = now_local.date() + timedelta(days=1)

    # "next <dayname>"
    next_match = re.search(r'\bnext\s+(\w+)\b', text)
    if next_match and target_date is None:
        day_name = next_match.group(1)
        if day_name in _DAY_NAMES:
            target_weekday = _DAY_NAMES[day_name]
            days_ahead = target_weekday - now_local.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            target_date = now_local.date() + timedelta(days=days_ahead)

    # "<dayname>" without "next" — means the next occurrence
    if target_date is None:
        for name, weekday in _DAY_NAMES.items():
            if re.search(rf'\b{name}\b', text) and not re.search(r'\bnext\s', text):
                days_ahead = weekday - now_local.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                target_date = now_local.date() + timedelta(days=days_ahead)
                break

    # "<month> <day>" or "<month> <day> <year>" — e.g. "june 5" or "june 5 2026"
    if target_date is None:
        month_match = re.search(
            r'\b(' + '|'.join(_MONTH_NAMES.keys()) + r')\s+(\d{1,2})(?:\s*,?\s*(\d{4}))?\b',
            text
        )
        if month_match:
            month = _MONTH_NAMES[month_match.group(1)]
            day = int(month_match.group(2))
            year = int(month_match.group(3)) if month_match.group(3) else now_local.year
            try:
                target_date = datetime(year, month, day).date()
            except ValueError:
                pass

    # "MM/DD/YYYY" or "MM-DD-YYYY" or "YYYY-MM-DD"
    if target_date is None:
        date_patterns = [
            (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b', lambda m: (int(m.group(3)), int(m.group(1)), int(m.group(2)))),
            (r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b', lambda m: (int(m.group(1)), int(m.group(2)), int(m.group(3)))),
        ]
        for pattern, extractor in date_patterns:
            match = re.search(pattern, text)
            if match:
                year, month, day = extractor(match)
                try:
                    target_date = datetime(year, month, day).date()
                except ValueError:
                    pass
                break

    # Just a time with no date = today (or tomorrow if the time has already passed)
    if target_date is None and hour is not None:
        target_date = now_local.date()
        tentative = datetime(target_date.year, target_date.month, target_date.day, hour, minute)
        if tentative <= now_local:
            target_date = target_date + timedelta(days=1)

    # Can't parse
    if target_date is None and hour is None:
        return None
    if target_date is None:
        return None

    # Default time to midnight if only a date was given
    if hour is None:
        hour = 9  # sensible default: 9am

    # Build the local datetime and convert to UTC
    try:
        local_dt = datetime(target_date.year, target_date.month, target_date.day, hour, minute)
    except ValueError:
        return None

    if utc_offset is not None:
        utc_dt = local_dt - timedelta(hours=utc_offset)
    else:
        utc_dt = local_dt

    ts = utc_dt.timestamp()

    # Must be in the future
    if ts <= datetime.utcnow().timestamp():
        return None

    return ts