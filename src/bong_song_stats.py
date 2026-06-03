import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bong_tools
import persist

_STORE_PATH = bong_tools.BONG_DATA / "song_stats.json"
_store = persist.PersistStore(_STORE_PATH, default={})
persist.register(_store)

_song_stats: dict[str, int] = {}


def load_song_stats():
    global _song_stats
    _store.load()
    _song_stats = dict(_store.data)


def _save_song_stats():
    _store.flush()


def _increment_song(title: str):
    _song_stats[title] = _song_stats.get(title, 0) + 1
    _store.mark_dirty()


def _get_top_songs(n: int = 3) -> list[tuple[str, int]]:
    sorted_songs = sorted(_song_stats.items(), key=lambda x: x[1], reverse=True)
    return sorted_songs[:n]


def _get_total_plays() -> int:
    return sum(_song_stats.values())


import bong_song_stats