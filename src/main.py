# main.py — Bot entry point
#
# This file sets up the discord.py bot, loads the Bong cog, and provides
# admin-only commands for toggling debug mode and shutting down the bot.
#
# Usage: bong [-d|--debug] [--restore-backup]
#   -d, --debug                Enable debug logging (console + file)
#   --restore-backup           Restore all data files from their .bak backups on startup

import argparse
import discord
import asyncio
import logging as _logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from discord.ext import commands

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BONG_DATA = PROJECT_ROOT / "bong_data"
BONG_USER_DATA = PROJECT_ROOT / "bong_user_data"

import debug
import user_data


def _check_ollama_available(ollama_host=None, timeout=30):
    import urllib.request
    import urllib.error

    ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
    _, _, port = ollama_host.partition(":")
    port = port or "11434"
    url = f"http://127.0.0.1:{port}/api/tags"

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2):
                debug.log("Ollama", f"Ollama is available at {ollama_host}")
                return
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            debug.log("Ollama", f"Ollama not reachable at {url}, retrying...")
            time.sleep(0.5)
    raise RuntimeError(
        f"Ollama is not running at {url}. "
        f"Start it with `ollama serve` or `systemctl start ollama.service`"
    )
_reboot_requested = False
_restore_backup = False


def _do_restore_backup():
    import persist
    import bong_song_stats
    import reminders
    restored = persist.restore_all_from_backup()
    if restored:
        user_data.load_users()
        bong_song_stats.load_song_stats()
        reminders.load_reminders()
        files = ", ".join(restored)
        print(f"Restored from backup: {files}")
        debug.log("Console", f"Restored from backup: {files}")
    else:
        print("No backup files found.")


def _console_reader(bot):
    global _reboot_requested
    _MUTE_LEVEL = _logging.CRITICAL + 1
    while True:
        try:
            print("Bong_OS: ", end="", flush=True)
            line = input("")
        except (EOFError, KeyboardInterrupt):
            return
        cmd = line.strip().lower()
        if cmd == "reboot":
            debug.log("Console", "Reboot requested via console")
            _reboot_requested = True
            asyncio.run_coroutine_threadsafe(bot.close(), bot.loop)
        elif cmd == "clear":
            for name in ("discord", "asyncio"):
                _logging.getLogger(name).setLevel(_MUTE_LEVEL)
            sys.stdout.write("\033[2J\033[3J\033[H")
            sys.stdout.flush()
            sys.stderr.flush()
            time.sleep(0.3)
            for name in ("discord", "asyncio"):
                _logging.getLogger(name).setLevel(_logging.WARNING)
            print("═══ Console Cleared ═══")
        elif cmd == "shutdown":
            print("Shutting down...")
            asyncio.run_coroutine_threadsafe(bot.close(), bot.loop)
        elif cmd == "restore-backup":
            _do_restore_backup()


def main():
    global _restore_backup
    parser = argparse.ArgumentParser(description="Bong Discord Bot")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--restore-backup", action="store_true", help="Restore all data files from their .bak backups on startup")
    args = parser.parse_args()

    if args.debug:
        debug.toggle_debug(True)
    _restore_backup = args.restore_backup

    load_dotenv(PROJECT_ROOT / ".env")

    _check_ollama_available()

    TOKEN = os.getenv("DISCORD_TOKEN")

    if not TOKEN:
        raise ValueError("No DISCORD_TOKEN found in environment variables!")

    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True

    bot = commands.Bot(command_prefix="@", case_insensitive=True, intents=intents)

    @bot.event
    async def on_ready():
        """Called when the bot connects to Discord and is ready to receive events."""
        debug.log("Bot", 'Bot booted, loading extensions...')
        await bot.load_extension('src.bong')
        import bong_tools
        import bong_song_stats
        import bong_memory_helpers
        import dm_approval
        import reminders

        await bot.user.edit(username="Bong")

        bong_memory_helpers._expire_old_memories()
        bong_song_stats.load_song_stats()
        dm_approval.load_pending_approvals()
        bong_tools.start_time = datetime.now()
        user_data.load_users()
        reminders.load_reminders()
        debug.log("Bot", f'Bot logged in as {bot.user}')
        if _restore_backup:
            _do_restore_backup()
        bot.loop.create_task(_setup_debugging())

    async def _setup_debugging():
        """Debug: auto-join a voice channel, enable VC commands, etc. on startup.
        Edit the function body to configure what runs at startup.
        Set _DEBUG_SETUP_ENABLED = False to disable.
        """
        _DEBUG_SETUP_ENABLED = False
        if not _DEBUG_SETUP_ENABLED:
            return

    @bot.command(name='poweroff', help="Power off the bot")
    async def poweroff(ctx):
        """Gracefully shut down the bot. Admin only."""
        if not user_data.has_permission(ctx.author.id, "admin") and not await bot.is_owner(ctx.author):
            await ctx.send("Only admins can use this command.")
            return
        await ctx.send("Onoffing...")
        await bot.close()

    @bot.command(name='debug', help="Toggle debug mode")
    async def toggle_debug_cmd(ctx, enabled: bool | None = None):
        """Toggle or set debug mode. Admin only.

        With no argument, toggles debug mode on/off.
        With True or False, explicitly sets it.
        """
        if not user_data.has_permission(ctx.author.id, "admin") and not await bot.is_owner(ctx.author):
            await ctx.send("Only admins can use this command.")
            return
        new_state = debug.toggle_debug(enabled)
        await ctx.send(f"Debug mode {'enabled' if new_state else 'disabled'}")

    t = threading.Thread(target=_console_reader, args=(bot,), daemon=True)
    t.start()

    bot.run(TOKEN)

    if _reboot_requested:
        debug.log("Console", "Rebooting...")
        os.execv(sys.executable, [sys.executable] + sys.argv)


if __name__ == "__main__":
    main()