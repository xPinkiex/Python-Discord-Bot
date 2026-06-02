import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bong_tools

SONG_STATS_FILE = bong_tools.BONG_DATA / "song_stats.json"
_song_stats: dict[str, int] = {}
_song_stats_dirty = False


def load_song_stats():
    global _song_stats
    try:
        if SONG_STATS_FILE.exists():
            with open(SONG_STATS_FILE, "r") as f:
                _song_stats = json.load(f)
    except Exception:
        _song_stats = {}


def _save_song_stats():
    try:
        with open(SONG_STATS_FILE, "w") as f:
            json.dump(bong_song_stats._song_stats, f, indent=2)
        bong_song_stats._song_stats_dirty = False
    except Exception:
        pass


def _increment_song(title: str):
    bong_song_stats._song_stats[title] = bong_song_stats._song_stats.get(title, 0) + 1
    bong_song_stats._song_stats_dirty = True


def _get_top_songs(n: int = 3) -> list[tuple[str, int]]:
    sorted_songs = sorted(bong_song_stats._song_stats.items(), key=lambda x: x[1], reverse=True)
    return sorted_songs[:n]


def _get_total_plays() -> int:
    return sum(bong_song_stats._song_stats.values())


import bong_song_stats