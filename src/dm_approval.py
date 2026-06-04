# dm_approval.py — DM approval system for unknown users
#
# When a user DMs Bong without the llm tag, this module sends a request to Eve
# with tag selection buttons. Approved users get tags persisted via user_data.py
# into users.json.

import discord
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import debug
import persist
import user_data

_APPROVAL_STORE_PATH = user_data.BONG_USER_DATA / "pending_approvals.json"
_approval_store = persist.PersistStore(_APPROVAL_STORE_PATH, default=[])
persist.register(_approval_store)

pending_approval: set[int] = set()


def load_pending_approvals():
    _approval_store.load()
    pending_approval.clear()
    pending_approval.update(_approval_store.data)


def _sync_store():
    _approval_store.data = list(pending_approval)
    _approval_store.mark_dirty()

# The owner who receives approval requests — always admin
OWNER_ID = user_data.OWNER_ID

# Tag preset definitions for approval buttons
TAG_PRESETS = {
    "chat": {"label": "Chat", "tags": ["llm"], "emoji": "\U0001f4ac"},
    "chat+music": {"label": "Chat+Music", "tags": ["llm", "music"], "emoji": "\U0001f3b5"},
    "full": {"label": "Full Access", "tags": ["llm", "music", "vc_commands", "e621"], "emoji": "\u2705"},
    "admin": {"label": "Admin", "tags": ["admin"], "emoji": "\U0001f451"},
}


class ApproveView(discord.ui.View):
    """Discord UI view with tag selection buttons for DM access requests."""

    def __init__(self, requesting_user: discord.User | discord.Member):
        super().__init__(timeout=300)
        self.requesting_user = requesting_user
        self._expired = False

    async def _approve(self, interaction: discord.Interaction, tags: list[str]):
        if interaction.user.id != OWNER_ID:
            await interaction.response.send_message("Only Eve can approve DM access.", ephemeral=True)
            return
        if self._expired:
            await interaction.response.edit_message(content="This request has timed out.", view=None)
            return
        tag_labels = ", ".join(tags)
        await interaction.response.edit_message(
            content=f"Approved **{self.requesting_user.display_name}** ({self.requesting_user.id}) with tags: {tag_labels}.",
            view=None,
        )
        user_data.set_permissions(self.requesting_user.id, tags)
        pending_approval.discard(self.requesting_user.id)
        _sync_store()
        self.stop()
        try:
            await self.requesting_user.send(f"Eve has approved you to talk with me! You now have access to: {tag_labels}. \U0001f389")
        except discord.Forbidden:
            pass

    @discord.ui.button(label="Chat", style=discord.ButtonStyle.secondary, emoji="\U0001f4ac")
    async def approve_chat(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._approve(interaction, ["llm"])

    @discord.ui.button(label="Chat+Music", style=discord.ButtonStyle.primary, emoji="\U0001f3b5")
    async def approve_chat_music(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._approve(interaction, ["llm", "music"])

    @discord.ui.button(label="Full Access", style=discord.ButtonStyle.success, emoji="\u2705")
    async def approve_full(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._approve(interaction, ["llm", "music", "vc_commands", "e621"])

    @discord.ui.button(label="Admin", style=discord.ButtonStyle.success, emoji="\U0001f451")
    async def approve_admin(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._approve(interaction, ["admin"])

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.danger, emoji="\u274c")
    async def deny(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != OWNER_ID:
            await interaction.response.send_message("Only Eve can deny DM access.", ephemeral=True)
            return
        if self._expired:
            await interaction.response.edit_message(content="This request has timed out.", view=None)
            return
        await interaction.response.edit_message(
            content=f"Denied **{self.requesting_user.display_name}** ({self.requesting_user.id}) DM access.",
            view=None,
        )
        pending_approval.discard(self.requesting_user.id)
        _sync_store()
        self.stop()
        try:
            await self.requesting_user.send("Eve has denied your request to talk with me. Sorry!")
        except discord.Forbidden:
            pass

    async def on_timeout(self):
        self._expired = True
        pending_approval.discard(self.requesting_user.id)
        _sync_store()
        try:
            await self.requesting_user.send("Your approval request has timed out.")
        except Exception:
            pass


async def process_dm(message: discord.Message, bot: discord.Client) -> bool:
    """Process a DM message. Returns True if the message should be handled by Bong.

    If the user doesn't have the llm tag, sends an approval request to Eve and returns False.
    If the user has the llm tag, returns True.
    If the user has a pending request, tells them to wait and returns False.
    """
    user = message.author

    if user_data.has_permission(user.id, "llm"):
        return True

    if user.id in pending_approval:
        try:
            await message.channel.send("Your request is still pending — Eve hasn't responded yet!")
        except discord.Forbidden:
            pass
        return False

    # User doesn't have llm — trigger approval flow
    pending_approval.add(user.id)
    _sync_store()

    owner = bot.get_user(OWNER_ID)
    if not owner:
        try:
            owner = await bot.fetch_user(OWNER_ID)
        except Exception as e:
            debug.error("Approval", f"Failed to fetch owner: {e}")
            pending_approval.discard(user.id)
            _sync_store()
            return False

    preview = (message.content[:100] + "...") if len(message.content) > 100 else (message.content or ("(attachment)" if message.attachments else "(empty message)"))
    view = ApproveView(user)
    try:
        await owner.send(
            f"\U0001f512 **DM Access Request**\n"
            f"**{user.display_name}** (`{user.id}`) wants to talk to Bong in DMs.\n"
            f"First message: \"{preview}\"",
            view=view,
        )
    except discord.Forbidden:
        pending_approval.discard(user.id)
        _sync_store()
        return False

    try:
        await user.send("I've sent your request to Eve. I'll let you know once she decides! \U0001f4ec")
    except discord.Forbidden:
        pass

    return False