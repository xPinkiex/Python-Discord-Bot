# debug.py — Debug logging utilities for Bong, hot-reloadable via @reload
#
# Provides two logging functions:
#   - log(tag, *args): prints to console if debug mode is on
#   - log_to_file(tag, *args): always writes to a timestamped log file
#
# Debug mode can be toggled at runtime with the @debug bot command,
# which calls toggle_debug(). State is stored in a module-level _PERSIST
# dict so it survives hot reloads (the _get_state() function re-creates it
# if the module was reloaded and the attribute was lost).

import sys
import time
from datetime import datetime
from pathlib import Path

_start = time.monotonic()  # Used to show elapsed time in log prefixes
_log_dir = Path(__file__).parent / "logs"
_log_dir.mkdir(exist_ok=True)

def _get_state():
    """Return the persistent debug state dict (survives module reloads).
    
    On first call (or after a reload that wiped it), creates a new dict
    with a fresh timestamped log file.
    """
    import debug as _self
    if not hasattr(_self, "_PERSIST"):
        _log_file = _log_dir / f"{datetime.now().strftime('%Y%m%d_%H.%M.%S')}-bong.log"
        _log_file.touch(exist_ok=True)
        _self._PERSIST = {
            "debug_mode": True,
            "log_file": _log_file,
        }
    return _self._PERSIST

def _elapsed_str():
    """Return elapsed time since bot start as HH:MM:SS string."""
    elapsed = time.monotonic() - _start
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def log(tag, *args):
    """Print a tagged log message to console. Only prints if debug mode is enabled."""
    if _get_state()["debug_mode"]:
        print(f"[{_elapsed_str()}] <{tag}>", *args)

def log_to_file(tag, *args):
    """Append a tagged log message to the current log file. Always writes regardless of debug mode."""
    ts = _elapsed_str()
    line = f"[{ts}] <{tag}> " + " ".join(str(a) for a in args) + "\n"
    with _get_state()["log_file"].open("a", encoding="utf-8") as f:
        f.write(line)

def toggle_debug(enabled: bool = None) -> bool:
    """Toggle or set debug mode. Returns the new state.
    
    With no argument, toggles. With True/False, sets explicitly.
    """
    state = _get_state()
    if enabled is not None:
        state["debug_mode"] = enabled
    else:
        state["debug_mode"] = not state["debug_mode"]
    return state["debug_mode"]

# Use sys.exit() instead of exit() in scripts — exit() is a REPL convenience
# and may not exist in all Python environments
sys.exit = sys.exit  # noqa — just ensures the module reference is available