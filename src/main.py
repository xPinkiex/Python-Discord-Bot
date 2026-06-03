# main.py — Bot entry point
#
# This file sets up the discord.py bot, loads the Bong cog, and provides
# admin-only commands for hot-reloading extensions, toggling debug mode,
# and shutting down the bot.
#
# Usage: bong [-d|--debug] [--reload-backup-data]
#   -d, --debug                Enable debug logging (console + file)
#   --reload-backup-data       Restore all data files from their .bak backups on startup

import argparse
import discord
import asyncio
import logging as _logging
import os
import sys
import threading
import time
import types
import importlib
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
            time.sleep(0.5)
    raise RuntimeError(
        f"Ollama is not running at {url}. "
        f"Start it with `ollama serve` or `systemctl start ollama.service`"
    )
_reboot_requested = False
_reload_backup = False


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
        elif cmd == "reload-backup-data":
            _do_restore_backup()


def main():
    global _reload_backup
    parser = argparse.ArgumentParser(description="Bong Discord Bot")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--reload-backup-data", action="store_true", help="Restore all data files from .bak backups on startup")
    args = parser.parse_args()

    if args.debug:
        debug.toggle_debug(True)
    _reload_backup = args.reload_backup_data

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
        bong_memory_helpers._expire_old_memories()
        bong_song_stats.load_song_stats()
        dm_approval.load_pending_approvals()
        bong_tools.start_time = datetime.now()
        user_data.load_users()
        reminders.load_reminders()
        debug.log("Bot", f'Bot logged in as {bot.user}')
        if _reload_backup:
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

    @bot.command(name='reload')
    async def reload_ext(ctx, util: str = "bong"):
        """Hot-reload a cog and its related modules without restarting the bot.

        This is the primary development workflow command. It:
          1. Snapshots all mutable state from the module (so e.g. music state isn't lost)
          2. Reloads the Python modules (bong, bong_tools, debug, and any submodules)
          3. Restores the saved state into the freshly reloaded modules
          4. Unloads and re-loads the discord.py cog extension

        Only admins can use this command.
        """
        if not user_data.is_admin(ctx.author.id) and not await bot.is_owner(ctx.author):
            await ctx.send("Only admins can use this command.")
            return
        try:
            # Stop all active voice listeners before reload — their sink objects
            # hold references to the old bot/guild/loop that won't survive the reload
            import voice_commands as _vc_pre
            for gid in list(_vc_pre._active_listeners):
                try:
                    sink = _vc_pre._active_listeners[gid]
                    sink._stopped = True
                    if sink._silence_task and not sink._silence_task.done():
                        sink._silence_task.cancel()
                except Exception:
                    pass
            _vc_pre._active_listeners.clear()
            _vc_pre._is_listening.clear()

            # Snapshot and restore mutable state across all related modules
            # so runtime state (shuffle, current track, pending flags, etc.) survives the reload
            # Skip _active_listeners and _is_listening from voice_commands — they hold stale references
            _skip_attrs = {"voice_commands": {"_active_listeners", "_is_listening"}}
            snapshots = {}
            for mod in [util, util + "_tools", util + "_music", util + "_memory", util + "_web", util + "_state", "debug", "dm_approval", "reminders", "user_data", "voice_commands"]:
                if mod in sys.modules:
                    skip = _skip_attrs.get(mod, set())
                    # Save all non-function, non-module, non-class attributes (i.e. runtime state)
                    snapshots[mod] = {k: getattr(sys.modules[mod], k) for k in dir(sys.modules[mod])
                        if not k.startswith("__") and k not in skip and not isinstance(getattr(sys.modules[mod], k), (types.FunctionType, types.ModuleType, type))}
                    importlib.reload(sys.modules[mod])
                    # Restore the saved state into the reloaded module
                    for k, v in snapshots[mod].items():
                        setattr(sys.modules[mod], k, v)
                debug.log("Bot", f"Reloaded module {mod}")
            # Also reload any submodules (e.g. bong_utilities.something)
            for mod_name in list(sys.modules):
                if mod_name.startswith(util + ".") or (mod_name != util and util in mod_name.split(".")):
                    snapshots[mod_name] = {k: getattr(sys.modules[mod_name], k) for k in dir(sys.modules[mod_name])
                        if not k.startswith("__") and not isinstance(getattr(sys.modules[mod_name], k), (types.FunctionType, types.ModuleType, type))}
                    importlib.reload(sys.modules[mod_name])
                    for k, v in snapshots[mod_name].items():
                        setattr(sys.modules[mod_name], k, v)
                    debug.log("Bot", f"Reloaded submodule {mod_name}")

            await bot.unload_extension(util)
            debug.log("Bot", f"Unloaded extension {util}")

            await bot.load_extension(util)
            debug.log("Bot", f"Reloaded extension {util}")

            # Inject a reload notification into all active channels so Bong has context
            import bong
            from datetime import datetime
            timestamp = datetime.now().strftime('%H:%M')
            for channel_id, history in list(bong.chat_memories.items()):
                history.insert(0, f"System at {timestamp}: Bong was hot-reloaded with new code. Previous conversation context is preserved.")

            if not isinstance(ctx.channel, discord.DMChannel):
                await ctx.message.delete()
        except commands.ExtensionNotLoaded:
            pass
        except discord.HTTPException as e:
            # 50003: Cannot execute action on DM channel — reload likely succeeded
            # but a post-reload action (like voice status) failed because we're in a DM.
            # Only surface the error in guild channels, not DMs.
            if not isinstance(ctx.channel, discord.DMChannel):
                await ctx.send(f"Error reloading extension: {e}")
        except Exception as e:
            await ctx.send(f"Error reloading extension: {e}")

    @bot.command(name='load')
    async def load_ext(ctx, util: str):
        """Load a cog extension by name. Admin only."""
        if not user_data.is_admin(ctx.author.id) and not await bot.is_owner(ctx.author):
            await ctx.send("Only admins can use this command.")
            return
        try:
            await bot.load_extension(util)
            debug.log("Bot", f"Loaded extension {util}")
            await ctx.send(f"Extension {util} loaded successfully!")

        except Exception as e:
            await ctx.send(f"Error loading extension: {e}")

    @bot.command(name='unload')
    async def unload_ext(ctx, util: str):
        """Unload a cog extension by name. Admin only."""
        if not user_data.is_admin(ctx.author.id) and not await bot.is_owner(ctx.author):
            await ctx.send("Only admins can use this command.")
            return
        try:
            await bot.unload_extension(util)
            debug.log("Bot", f"Unloaded extension {util}")
            await ctx.send(f"Extension {util} unloaded successfully!")

        except Exception as e:
            await ctx.send(f"Error unloading extension: {e}")

    @bot.command(name='poweroff', help="Power off the bot")
    async def poweroff(ctx):
        """Gracefully shut down the bot. Admin only."""
        if not user_data.is_admin(ctx.author.id) and not await bot.is_owner(ctx.author):
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
        if not user_data.is_admin(ctx.author.id) and not await bot.is_owner(ctx.author):
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