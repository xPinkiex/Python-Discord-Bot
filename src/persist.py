# persist.py — Centralized JSON persistence with dirty-flag batching
#
# Each module that needs persistence creates a PersistStore object.
# Mutations mark the store dirty; writes only happen on flush().
# A periodic task (60s) and shutdown hook flush all registered stores.
#
# Safety features:
#   - Backup before write: .bak file preserves last good state on every flush
#   - Dirty flag: mutations don't trigger immediate I/O
#   - Explicit backup restore via restore_all_from_backup() or --restore-backup

import json
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

_stores: list["PersistStore"] = []


class PersistStore:
    """Lazy JSON file store: load once, mutate in memory, flush only when dirty."""

    def __init__(self, path: Path, default=None):
        self.path = path
        self._default = default if default is not None else {}
        if isinstance(self._default, dict):
            self.data = dict(self._default)
        elif isinstance(self._default, list):
            self.data = list(self._default)
        else:
            self.data = self._default
        self._dirty = False

    def load(self):
        """Load from disk. Falls through to default if file is missing or corrupt."""
        self._dirty = False
        try:
            if self.path.exists():
                with open(self.path, "r") as f:
                    self.data = json.load(f)
                return
        except Exception:
            pass
        if isinstance(self._default, dict):
            self.data = dict(self._default)
        elif isinstance(self._default, list):
            self.data = list(self._default)
        else:
            self.data = self._default

    def mark_dirty(self):
        self._dirty = True

    def flush(self):
        """Write to disk only if dirty. Backs up the existing file first."""
        if not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                bak = self.path.with_suffix(self.path.suffix + ".bak")
                shutil.copy2(self.path, bak)
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2)
            self._dirty = False
        except Exception:
            pass

    def restore_from_backup(self) -> bool:
        """Load data from .bak file. Returns True if backup was loaded."""
        bak = self.path.with_suffix(self.path.suffix + ".bak")
        if not bak.exists():
            return False
        try:
            with open(bak, "r") as f:
                self.data = json.load(f)
            self._dirty = True
            return True
        except Exception:
            return False

    @property
    def dirty(self) -> bool:
        return self._dirty


def register(store: PersistStore):
    _stores.append(store)


def flush_all():
    for store in _stores:
        store.flush()


def restore_all_from_backup() -> list[str]:
    """Restore all stores from their .bak files. Returns list of restored filenames."""
    restored = []
    for store in _stores:
        if store.restore_from_backup():
            restored.append(store.path.name)
    if restored:
        flush_all()
    return restored