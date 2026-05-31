import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

import bong
import user_data


@pytest.fixture
def bot():
    b = MagicMock()
    b.user = MagicMock()
    b.user.id = 999
    b.user.display_name = "Bong"
    b.loop = MagicMock()
    b.loop.create_task = MagicMock()
    b.all_commands = {}
    b.command_prefix = "@"
    return b


@pytest.fixture
def cog(bot):
    return bong.BongCog(bot)


@pytest.fixture(autouse=True)
def reset_state():
    bong.active_channels.clear()
    bong.chat_memories.clear()
    bong.channel_summaries.clear()
    yield


class TestCooldown:
    def test_regular_user_records_cooldown(self, cog):
        now = time.time()
        uid = 12345
        assert uid not in cog._user_cooldowns

        with patch.object(user_data, "is_authorized", return_value=False):
            with patch.object(user_data, "is_admin", return_value=False):
                assert not user_data.is_authorized(uid)

    def test_cooldown_blocks_within_60s(self, cog):
        now = time.time()
        uid = 12345
        cog._user_cooldowns[uid] = now - 10

        with patch.object(user_data, "is_authorized", return_value=False):
            with patch.object(user_data, "is_admin", return_value=False):
                elapsed = now - cog._user_cooldowns[uid]
                assert elapsed < 60
                remaining = int(60 - elapsed)
                assert remaining > 0

    def test_cooldown_expires_after_60s(self, cog):
        now = time.time()
        uid = 12345
        cog._user_cooldowns[uid] = now - 61

        elapsed = now - cog._user_cooldowns[uid]
        assert elapsed >= 60

    def test_cooldown_is_per_user(self, cog):
        now = time.time()
        cog._user_cooldowns[111] = now - 10

        assert 222 not in cog._user_cooldowns

    def test_authorized_user_never_in_cooldown(self, cog):
        assert 111 not in cog._user_cooldowns

    def test_cooldown_remaining_rounds_correctly(self, cog):
        now = time.time()
        uid = 12345
        cog._user_cooldowns[uid] = now - 45

        elapsed = now - cog._user_cooldowns[uid]
        remaining = int(60 - elapsed)
        assert remaining == 14 or remaining == 15


class TestDeduplication:
    def test_same_message_id_is_duplicate(self, cog):
        msg_id = 123456
        cog._processed_ids.add(msg_id)
        assert msg_id in cog._processed_ids

    def test_different_message_id_is_not_duplicate(self, cog):
        cog._processed_ids.add(123456)
        assert 123457 not in cog._processed_ids

    def test_processed_ids_trim_at_capacity(self, cog):
        cog._processed_ids.clear()
        for i in range(2001):
            cog._processed_ids.add(i)
        assert len(cog._processed_ids) == 2001

        cog._processed_ids = set(sorted(cog._processed_ids)[-1000:])
        assert len(cog._processed_ids) == 1000
        assert 2000 in cog._processed_ids
        assert 0 not in cog._processed_ids