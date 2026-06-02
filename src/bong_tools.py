import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BONG_DATA = PROJECT_ROOT / "bong_data"
BONG_USER_DATA = PROJECT_ROOT / "bong_user_data"

DOWNLOAD_DIR = BONG_DATA / "saved_sounds"
DOWNLOAD_DIR.mkdir(exist_ok=True)
IMAGE_DIR = BONG_DATA / "saved_images"
IMAGE_DIR.mkdir(exist_ok=True)
TEXT_DIR = BONG_DATA / "saved_texts"
TEXT_DIR.mkdir(exist_ok=True)

BOT_USER_ID = "698627881760456724"

pending_reactions = []
pending_join_voice = None
pending_leave_voice = None
pending_shutdown = False
pending_play_audio = None
pending_pause = False
pending_resume = False
pending_stop = False
pending_skip = False
pending_skip_target = None
pending_skip_info = ""
pending_send_image = None
pending_send_text = None
pending_start_listening = None
pending_stop_listening = False

voice_connected = False
caller_in_voice = False
current_user_id = None
current_channel_id = None
authorized = False
current_username = ""
start_time = None
shuffle_enabled = False
loop_enabled = False
loop_track = None
current_track = None
song_queue: list[str] = []

image_library = []
text_library = []
music_library = []

import bong_tools

from bong_music import tools as _music_tools
from bong_memory import tools as _memory_tools
from bong_web import tools as _web_tools
from bong_state import tools as _state_tools
tools = _music_tools + _memory_tools + _web_tools + _state_tools
tool_map = {t.name: t for t in tools}


def reset_pending():
    bong_tools.pending_reactions.clear()
    bong_tools.pending_join_voice = None
    bong_tools.pending_leave_voice = None
    bong_tools.pending_shutdown = False
    bong_tools.pending_play_audio = None
    bong_tools.pending_pause = False
    bong_tools.pending_resume = False
    bong_tools.pending_stop = False
    bong_tools.pending_skip = False
    bong_tools.pending_skip_target = None
    bong_tools.pending_skip_info = ""
    bong_tools.pending_send_image = None
    bong_tools.pending_send_text = None
    bong_tools.pending_start_listening = None
    bong_tools.pending_stop_listening = False


def refresh_image_library():
    bong_tools.image_library = sorted(
        p for p in bong_tools.IMAGE_DIR.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
    )


def refresh_text_library():
    bong_tools.text_library = sorted(
        p for p in bong_tools.TEXT_DIR.iterdir()
        if p.suffix.lower() in (".txt", ".md", ".py", ".json", ".csv", ".xml", ".yaml", ".yml", ".cfg", ".ini", ".log", ".toml", ".rs", ".js", ".ts", ".html", ".css", ".sh", ".bat")
    )


def refresh_music_library():
    bong_tools.music_library = sorted(bong_tools.DOWNLOAD_DIR.glob("*.mp3"))


refresh_music_library()
refresh_image_library()
refresh_text_library()