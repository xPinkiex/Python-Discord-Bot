# dm_approval.py — Permission tiers and DM approval system for Bong
#
# Three permission tiers:
#   admin     — Full access: all commands, shutdown, channel toggle, DM access
#   authorized — Full access for now (future-proofing for restricted commands)
#   user      — Chatting + all tools (music, voice, memory, etc.) but no system commands
#
# When a non-allowlisted user DMs Bong, this module sends a request to Eve
# with approve/deny buttons. Approved users are persisted to users.json.

import json
import discord
from pathlib import Path

# Path to the JSON file that persists user tiers across restarts
_USERS_FILE = Path(__file__).parent / "users.json"

# In-memory tier lookup: user_id -> "admin" | "authorized" | "user"
user_tiers: dict[int, str] = {}

# Track users currently waiting for approval so we don't send duplicate requests
pending_approval: set[int] = set()

# The owner who receives approval requests — always has admin access even if the file is missing
OWNER_ID = 273761843544064000


def load_users():
    """Load user tiers from disk. The owner is always guaranteed admin access."""
    global user_tiers
    user_tiers = {}
    try:
        if _USERS_FILE.exists():
            with open(_USERS_FILE, "r") as f:
                file_data = json.load(f)
            for uid_str, tier in file_data.items():
                user_tiers[int(uid_str)] = tier
    except Exception:
        pass
    # Owner is always admin — even if the file is missing or corrupted
    user_tiers[OWNER_ID] = "admin"


def save_users():
    """Persist all user tiers to disk."""
    try:
        data = {str(uid): tier for uid, tier in user_tiers.items()}
        with open(_USERS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def get_tier(user_id: int) -> str | None:
    """Get the permission tier for a user, or None if not in any tier."""
    return user_tiers.get(user_id)


def is_admin(user_id: int) -> bool:
    """Check if a user has admin tier (full access including system commands)."""
    return user_tiers.get(user_id) == "admin"


def is_authorized(user_id: int) -> bool:
    """Check if a user has admin or authorized tier (full access)."""
    return user_tiers.get(user_id) in ("admin", "authorized")


def is_known(user_id: int) -> bool:
    """Check if a user is in any tier (admin, authorized, or user)."""
    return user_id in user_tiers


class ApproveView(discord.ui.View):
    """Discord UI view with tier selection buttons for DM access requests."""

    def __init__(self, requesting_user: discord.User | discord.Member, dm_channel: discord.DMChannel):
        super().__init__(timeout=300)
        self.requesting_user = requesting_user
        self.dm_channel = dm_channel

    async def _approve(self, interaction: discord.Interaction, tier: str):
        if interaction.user.id != OWNER_ID:
            await interaction.response.send_message("Only Eve can approve DM access.", ephemeral=True)
            return
        user_tiers[self.requesting_user.id] = tier
        save_users()
        pending_approval.discard(self.requesting_user.id)
        self.stop()
        tier_label = {"admin": "Admin", "authorized": "Authorized", "user": "User"}[tier]
        await interaction.response.edit_message(
            content=f"✅ Approved **{self.requesting_user.display_name}** ({self.requesting_user.id}) as **{tier_label}**.",
            view=None,
        )
        try:
            await self.requesting_user.send("Eve has approved you to talk with me! You can now send me messages here. 🎉")
        except discord.Forbidden:
            pass

    @discord.ui.button(label="User", style=discord.ButtonStyle.secondary, emoji="👥")
    async def approve_user(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._approve(interaction, "user")

    @discord.ui.button(label="Authorized", style=discord.ButtonStyle.primary, emoji="🔐")
    async def approve_authorized(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._approve(interaction, "authorized")

    @discord.ui.button(label="Admin", style=discord.ButtonStyle.success, emoji="👑")
    async def approve_admin(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._approve(interaction, "admin")

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.danger, emoji="❌")
    async def deny(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != OWNER_ID:
            await interaction.response.send_message("Only Eve can deny DM access.", ephemeral=True)
            return
        pending_approval.discard(self.requesting_user.id)
        self.stop()
        await interaction.response.edit_message(
            content=f"❌ Denied **{self.requesting_user.display_name}** ({self.requesting_user.id}) DM access.",
            view=None,
        )
        try:
            await self.requesting_user.send("Eve has denied your request to talk with me. Sorry!")
        except discord.Forbidden:
            pass

    async def on_timeout(self):
        pending_approval.discard(self.requesting_user.id)


async def process_dm(message: discord.Message, bot: discord.Client) -> bool:
    """Process a DM message. Returns True if the message should be handled by Bong.

    If the user is not in any tier, sends an approval request to Eve and returns False.
    If the user is in any tier, returns True.
    If the user has a pending request, tells them to wait and returns False.
    """
    user = message.author

    if is_known(user.id):
        return True

    if user.id in pending_approval:
        try:
            await message.channel.send("Your request is still pending — Eve hasn't responded yet!")
        except discord.Forbidden:
            pass
        return False

    pending_approval.add(user.id)

    owner = bot.get_user(OWNER_ID)
    if not owner:
        try:
            owner = await bot.fetch_user(OWNER_ID)
        except Exception:
            pending_approval.discard(user.id)
            return False

    preview = (message.content[:100] + "...") if len(message.content) > 100 else (message.content or "(attachment)")
    dm_channel = await user.create_dm()
    view = ApproveView(user, dm_channel)
    try:
        await owner.send(
            f"🔒 **DM Access Request**\n"
            f"**{user.display_name}** (`{user.id}`) wants to talk to Bong in DMs.\n"
            f"First message: \"{preview}\"",
            view=view,
        )
    except discord.Forbidden:
        pending_approval.discard(user.id)
        return False

    try:
        await user.send("I've sent your request to Eve. I'll let you know once she decides! 📬")
    except discord.Forbidden:
        pass

    return False