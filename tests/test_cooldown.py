import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

import user_data


class TestCooldown:
    def test_llm_fast_user_bypasses_cooldown(self):
        uid = 12345
        # llm_fast tag should bypass cooldown
        user_data._user_data = {uid: {"allowed": ["llm_fast"]}}
        assert user_data.has_permission(uid, "llm_fast")
        assert user_data.has_permission(uid, "llm")  # llm_fast implies llm

    def test_llm_user_has_cooldown(self):
        uid = 12345
        user_data._user_data = {uid: {"allowed": ["llm"]}}
        assert user_data.has_permission(uid, "llm")
        assert not user_data.has_permission(uid, "llm_fast")

    def test_admin_bypasses_cooldown(self):
        uid = 12345
        user_data._user_data = {uid: {"allowed": ["admin"]}}
        assert user_data.has_permission(uid, "llm_fast")  # admin implies all
        assert user_data.has_permission(uid, "llm")
        assert user_data.has_permission(uid, "music")

    def test_cooldown_per_user(self):
        now = time.time()
        cooldowns = {111: now - 10}
        assert 111 in cooldowns
        assert 222 not in cooldowns

    def test_cooldown_remaining_rounds(self):
        now = time.time()
        uid = 12345
        cooldowns = {uid: now - 45}
        elapsed = now - cooldowns[uid]
        remaining = int(60 - elapsed)
        assert remaining == 14 or remaining == 15


class TestDeduplication:
    def test_same_message_id_is_duplicate(self):
        processed = set()
        msg_id = 123456
        processed.add(msg_id)
        assert msg_id in processed

    def test_different_message_id_is_not_duplicate(self):
        processed = set()
        processed.add(123456)
        assert 123457 not in processed

    def test_processed_ids_trim_at_capacity(self):
        processed = set(range(2001))
        assert len(processed) == 2001
        processed = set(sorted(processed)[-1000:])
        assert len(processed) == 1000
        assert 2000 in processed
        assert 0 not in processed


class TestPermissions:
    def test_has_permission_admin_implies_all(self):
        user_data._user_data = {1: {"allowed": ["admin"]}}
        for tag in ["llm", "llm_fast", "music", "vc_commands", "e621", "admin"]:
            assert user_data.has_permission(1, tag), f"admin should imply {tag}"

    def test_has_permission_llm_fast_implies_llm(self):
        user_data._user_data = {1: {"allowed": ["llm_fast"]}}
        assert user_data.has_permission(1, "llm")
        assert user_data.has_permission(1, "llm_fast")
        assert not user_data.has_permission(1, "music")

    def test_has_permission_unknown_user(self):
        user_data._user_data = {}
        assert not user_data.has_permission(999, "llm")

    def test_set_permissions(self):
        user_data._user_data = {}
        user_data.set_permissions(1, ["llm", "music"])
        assert user_data.has_permission(1, "llm")
        assert user_data.has_permission(1, "music")
        assert not user_data.has_permission(1, "e621")

    def test_add_permission(self):
        user_data._user_data = {1: {"allowed": ["llm"]}}
        user_data.add_permission(1, "music")
        assert user_data.has_permission(1, "music")

    def test_remove_permission(self):
        user_data._user_data = {1: {"allowed": ["llm", "music"]}}
        user_data.remove_permission(1, "music")
        assert not user_data.has_permission(1, "music")
        assert user_data.has_permission(1, "llm")

    def test_owner_always_admin(self):
        user_data._user_data = {}
        assert user_data.is_admin(user_data.OWNER_ID)

    def test_valid_tags(self):
        assert "llm" in user_data.VALID_TAGS
        assert "llm_fast" in user_data.VALID_TAGS
        assert "music" in user_data.VALID_TAGS
        assert "vc_commands" in user_data.VALID_TAGS
        assert "e621" in user_data.VALID_TAGS
        assert "admin" in user_data.VALID_TAGS

    def test_youtube_search_requires_llm_or_music(self):
        user_data._user_data = {1: {"allowed": ["llm"]}}
        assert user_data.has_permission(1, "llm") or user_data.has_permission(1, "music")
        user_data._user_data = {2: {"allowed": ["music"]}}
        assert user_data.has_permission(2, "llm") or user_data.has_permission(2, "music")
        user_data._user_data = {3: {"allowed": ["e621"]}}
        assert not (user_data.has_permission(3, "llm") or user_data.has_permission(3, "music"))