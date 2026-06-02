# main.py — Bot entry point
#
# This file sets up the discord.py bot, loads the Bong cog, and provides
# owner-only commands for hot-reloading extensions, toggling debug mode,
# and shutting down the bot.
#
# Usage: bong [-d|--debug]
#   -d, --debug    Enable debug logging (console + file)

import argparse
import discord
import asyncio
import os
import signal
import subprocess
import sys
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

_ollama_process = None


def _start_ollama():
    global _ollama_process

    ollama_host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
    env = {**os.environ, "OLLAMA_HOST": ollama_host}

    _kill_ollama()

    debug.log("Ollama", f"Starting ollama serve (OLLAMA_HOST={ollama_host})...")
    _ollama_process = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    _wait_for_ollama(ollama_host)

    model = "nomic-embed-text"
    debug.log("Ollama", f"Ensuring embedding model '{model}' is pulled...")
    subprocess.run(
        ["ollama", "pull", model],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    debug.log("Ollama", "Ready.")


def _kill_ollama():
    try:
        result = subprocess.run(
            ["pgrep", "-f", "ollama serve"],
            capture_output=True, text=True,
        )
        for pid_str in result.stdout.strip().splitlines():
            pid = int(pid_str.strip())
            if pid != os.getpid():
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        time.sleep(0.5)
        for pid_str in result.stdout.strip().splitlines():
            pid = int(pid_str.strip())
            if pid != os.getpid():
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    except Exception:
        pass
    time.sleep(0.5)


def _wait_for_ollama(ollama_host, timeout=30):
    import urllib.request
    import urllib.error

    host, _, port = ollama_host.partition(":")
    port = port or "11434"
    url = f"http://127.0.0.1:{port}/api/tags"

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2):
                debug.log("Ollama", "Server is responding.")
                return
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            time.sleep(0.5)
    raise RuntimeError(f"Ollama server did not respond within {timeout}s at {url}")


def _stop_ollama():
    global _ollama_process
    if _ollama_process is not None:
        debug.log("Ollama", "Stopping ollama serve...")
        try:
            _ollama_process.terminate()
            _ollama_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _ollama_process.kill()
        except Exception:
            pass
        _ollama_process = None


def main():
    parser = argparse.ArgumentParser(description="Bong Discord Bot")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        debug.toggle_debug(True)

    load_dotenv(PROJECT_ROOT / ".env")

    _start_ollama()

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
        import user_data
        bong_memory_helpers._expire_old_memories()
        bong_song_stats.load_song_stats()
        bong_tools.start_time = datetime.now()
        user_data.load_users()
        reminders.load_reminders()
        debug.log("Bot", f'Bot logged in as {bot.user}')
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
    @commands.is_owner()
    async def reload_ext(ctx, util: str = "bong"):
        """Hot-reload a cog and its related modules without restarting the bot.

        This is the primary development workflow command. It:
          1. Snapshots all mutable state from the module (so e.g. music state isn't lost)
          2. Reloads the Python modules (bong, bong_tools, debug, and any submodules)
          3. Restores the saved state into the freshly reloaded modules
          4. Unloads and re-loads the discord.py cog extension

        Only the bot owner can use this command.
        """
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
    @commands.is_owner()
    async def load_ext(ctx, util: str):
        """Load a cog extension by name. Only the bot owner can use this."""
        try:
            await bot.load_extension(util)
            debug.log("Bot", f"Loaded extension {util}")
            await ctx.send(f"Extension {util} loaded successfully!")

        except Exception as e:
            await ctx.send(f"Error loading extension: {e}")

    @bot.command(name='unload')
    @commands.is_owner()
    async def unload_ext(ctx, util: str):
        """Unload a cog extension by name. Only the bot owner can use this."""
        try:
            await bot.unload_extension(util)
            debug.log("Bot", f"Unloaded extension {util}")
            await ctx.send(f"Extension {util} unloaded successfully!")

        except Exception as e:
            await ctx.send(f"Error unloading extension: {e}")

    @bot.command(name='poweroff', help="Power off the bot")
    @commands.is_owner()
    async def poweroff(ctx):
        """Gracefully shut down the bot. Only the bot owner can use this."""
        await ctx.send("Onoffing...")
        await bot.close()

    @bot.command(name='debug', help="Toggle debug mode")
    @commands.is_owner()
    async def toggle_debug_cmd(ctx, enabled: bool | None = None):
        """Toggle or set debug mode. Only the bot owner can use this.

        With no argument, toggles debug mode on/off.
        With True or False, explicitly sets it.
        """
        new_state = debug.toggle_debug(enabled)
        await ctx.send(f"Debug mode {'enabled' if new_state else 'disabled'}")

    @bot.event
    async def on_close():
        _stop_ollama()

    try:
        bot.run(TOKEN)
    finally:
        _stop_ollama()


if __name__ == "__main__":
    main()