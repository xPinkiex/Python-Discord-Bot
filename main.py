# main.py — Bot entry point
#
# This file sets up the discord.py bot, loads the Bong cog, and provides
# owner-only commands for hot-reloading extensions, toggling debug mode,
# and shutting down the bot.

import discord
import os
import sys
import types
import importlib
from dotenv import load_dotenv
from discord.ext import commands
import debug

# Load environment variables from .env file (DISCORD_TOKEN, etc.)
load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")

if not TOKEN:
    raise ValueError("No DISCORD_TOKEN found in environment variables!")

# discord.py requires these intents to read message content and member info
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix="@", case_insensitive=True, intents=intents)

@bot.event
async def on_ready():
    """Called when the bot connects to Discord and is ready to receive events."""
    debug.log("Bot", 'Bot booted, loading extensions...')
    await bot.load_extension('bong')
    import bong_tools
    import dm_approval
    bong_tools._expire_old_memories()
    dm_approval.load_users()
    debug.log("Bot", f'Bot logged in as {bot.user}')

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
        # Snapshot and restore mutable state across all related modules
        # so runtime state (shuffle, current track, pending flags, etc.) survives the reload
        snapshots = {}
        for mod in [util, util + "_tools", "debug"]:
            if mod in sys.modules:
                # Save all non-function, non-module, non-class attributes (i.e. runtime state)
                snapshots[mod] = {k: getattr(sys.modules[mod], k) for k in dir(sys.modules[mod])
                    if not k.startswith("__") and not isinstance(getattr(sys.modules[mod], k), (types.FunctionType, types.ModuleType, type))}
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
        
        await ctx.message.delete()
    except commands.ExtensionNotLoaded:
        pass
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
async def toggle_debug(ctx, enabled: bool | None = None):
    """Toggle or set debug mode. Only the bot owner can use this.
    
    With no argument, toggles debug mode on/off.
    With True or False, explicitly sets it.
    """
    if enabled is None:
        enabled = not debug.toggle_debug()
    debug.toggle_debug(enabled)
    await ctx.send(f"Debug mode {'enabled' if enabled else 'disabled'}")

# Start the bot — this blocks until bot.close() is called
bot.run(TOKEN)