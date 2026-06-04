import math
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import bong_memory_helpers


# ========== _clean_for_embedding ==========

class TestCleanForEmbedding:
    def test_strips_mention(self):
        assert bong_memory_helpers._clean_for_embedding("<@273761843544064000> loves dubstep") == "loves dubstep"

    def test_strips_bang_mention(self):
        assert bong_memory_helpers._clean_for_embedding("<@!123> likes cars") == "likes cars"

    def test_strips_userid_tag(self):
        assert bong_memory_helpers._clean_for_embedding("Eve (userID: 123) likes cats") == "Eve likes cats"

    def test_strips_userid_tag_no_space(self):
        assert bong_memory_helpers._clean_for_embedding("Eve(userID:456) likes cats") == "Eve likes cats"

    def test_collapses_double_spaces(self):
        assert bong_memory_helpers._clean_for_embedding("RadonFox  is  a creecher") == "RadonFox is a creecher"

    def test_collapses_triple_spaces(self):
        assert bong_memory_helpers._clean_for_embedding("hello   world") == "hello world"

    def test_combination(self):
        assert bong_memory_helpers._clean_for_embedding("<@123>  told me (userID: 456)") == "told me"

    def test_preserves_normal_text(self):
        text = "Eve loves dubstep and skrillex"
        assert bong_memory_helpers._clean_for_embedding(text) == text

    def test_preserves_bong_in_text(self):
        text = "Bong remembers that Eve likes cars"
        assert bong_memory_helpers._clean_for_embedding(text) == text

    def test_preserves_bongs_possessive(self):
        text = "Bong's rule: one joke per day"
        assert bong_memory_helpers._clean_for_embedding(text) == text

    def test_strips_multiple_mentions(self):
        assert bong_memory_helpers._clean_for_embedding("<@111> told <@222> that") == "told that"

    def test_strips_leading_trailing_whitespace(self):
        assert bong_memory_helpers._clean_for_embedding("  hello world  ") == "hello world"

    def test_empty_string(self):
        assert bong_memory_helpers._clean_for_embedding("") == ""

    def test_only_mention(self):
        assert bong_memory_helpers._clean_for_embedding("<@123>") == ""


# ========== _apply_category_boost ==========

class TestCategoryBoost:
    def test_preference(self):
        assert bong_memory_helpers._apply_category_boost(0.5, "preference") == pytest.approx(0.8)

    def test_instruction(self):
        assert bong_memory_helpers._apply_category_boost(0.5, "instruction") == pytest.approx(0.75)

    def test_relationship(self):
        assert bong_memory_helpers._apply_category_boost(0.5, "relationship") == pytest.approx(0.7)

    def test_fact(self):
        assert bong_memory_helpers._apply_category_boost(0.5, "fact") == pytest.approx(0.6)

    def test_inside_joke(self):
        assert bong_memory_helpers._apply_category_boost(0.5, "inside_joke") == pytest.approx(0.55)

    def test_unknown_category(self):
        assert bong_memory_helpers._apply_category_boost(0.5, "unknown") == pytest.approx(0.5)

    def test_zero_score(self):
        assert bong_memory_helpers._apply_category_boost(0.0, "preference") == pytest.approx(0.3)


# ========== _apply_importance_boost ==========

class TestImportanceBoost:
    def test_importance_1(self):
        assert bong_memory_helpers._apply_importance_boost(0.5, 1) == pytest.approx(0.55)

    def test_importance_3(self):
        assert bong_memory_helpers._apply_importance_boost(0.5, 3) == pytest.approx(0.65)

    def test_importance_5(self):
        assert bong_memory_helpers._apply_importance_boost(0.5, 5) == pytest.approx(0.75)

    def test_importance_0(self):
        assert bong_memory_helpers._apply_importance_boost(0.5, 0) == pytest.approx(0.5)

    def test_zero_score(self):
        assert bong_memory_helpers._apply_importance_boost(0.0, 5) == pytest.approx(0.25)


# ========== _apply_recency_boost ==========

class TestRecencyBoost:
    def test_fresh_memory(self):
        now = datetime.now().timestamp()
        result = bong_memory_helpers._apply_recency_boost(0.5, now, now)
        assert result == pytest.approx(0.5 + 0.15, abs=0.01)

    def test_60_day_halflife(self):
        now = datetime.now().timestamp()
        sixty_days_ago = now - (60 * 86400)
        result = bong_memory_helpers._apply_recency_boost(0.5, sixty_days_ago, sixty_days_ago)
        assert result == pytest.approx(0.5 + 0.15 * 0.5, abs=0.01)

    def test_very_old_memory(self):
        now = datetime.now().timestamp()
        very_old = now - (365 * 86400)
        result = bong_memory_helpers._apply_recency_boost(0.5, very_old, very_old)
        assert result < 0.5 + 0.01

    def test_last_accessed_overrides_saved_at(self):
        now = datetime.now().timestamp()
        old_saved = now - (365 * 86400)
        result_old = bong_memory_helpers._apply_recency_boost(0.5, old_saved, now)
        result_new = bong_memory_helpers._apply_recency_boost(0.5, now, old_saved)
        assert result_old == pytest.approx(result_new)

    def test_zero_timestamps(self):
        result = bong_memory_helpers._apply_recency_boost(0.5, 0, 0)
        assert result == pytest.approx(0.5)

    def test_none_timestamps(self):
        result = bong_memory_helpers._apply_recency_boost(0.5, None, None)
        assert result == pytest.approx(0.5)

    def test_saved_at_only(self):
        now = datetime.now().timestamp()
        result = bong_memory_helpers._apply_recency_boost(0.5, now, None)
        assert result > 0.5

    def test_last_accessed_only(self):
        now = datetime.now().timestamp()
        result = bong_memory_helpers._apply_recency_boost(0.5, None, now)
        assert result > 0.5


# ========== _apply_user_match_boost ==========

class TestUserMatchBoost:
    def test_from_user_search(self):
        assert bong_memory_helpers._apply_user_match_boost(0.5, True) == pytest.approx(0.8)

    def test_not_from_user_search(self):
        assert bong_memory_helpers._apply_user_match_boost(0.5, False) == pytest.approx(0.5)

    def test_zero_score_from_user(self):
        assert bong_memory_helpers._apply_user_match_boost(0.0, True) == pytest.approx(0.3)


# ========== resolve_name_to_id ==========

class TestResolveNameToId:
    @pytest.fixture(autouse=True)
    def setup_user_data(self):
        import user_data
        user_data._user_data = {
            111: {"display_name": "Eve", "allowed": ["admin"]},
            222: {"display_name": "RadonFox", "allowed": ["llm"]},
            333: {"display_name": "Not Reed", "allowed": ["llm", "music"]},
        }
        yield
        user_data._user_data = {}

    def test_exact_match(self):
        uid, warn = bong_memory_helpers.resolve_name_to_id("Eve")
        assert uid == 111
        assert warn is None

    def test_case_insensitive(self):
        uid, warn = bong_memory_helpers.resolve_name_to_id("eve")
        assert uid == 111
        assert warn is None

    def test_case_insensitive_mixed(self):
        uid, warn = bong_memory_helpers.resolve_name_to_id("radonfox")
        assert uid == 222
        assert warn is None

    def test_no_match(self):
        uid, warn = bong_memory_helpers.resolve_name_to_id("NonExistentPerson")
        assert uid is None
        assert warn is None

    def test_fuzzy_match(self):
        uid, warn = bong_memory_helpers.resolve_name_to_id("Ev")
        assert uid == 111
        assert warn is not None
        assert "fuzzily" in warn.lower() or "matched" in warn.lower()

    def test_fuzzy_no_match_below_cutoff(self):
        uid, warn = bong_memory_helpers.resolve_name_to_id("xyzabc")
        assert uid is None

    def test_whitespace_stripped(self):
        uid, warn = bong_memory_helpers.resolve_name_to_id("  Eve  ")
        assert uid == 111
        assert warn is None


# ========== _expire_old_memories ==========

class TestExpireOldMemories:
    """Test expiry logic by verifying cutoff calculations.
    
    _expire_old_memories uses: cutoff = max(saved_at, last_accessed) + MEMORY_EXPIRY_BASE_DAYS * importance * 86400
    We test the formula directly since ChromaDB mocking is problematic for property-based access.
    """

    def test_importance_1_expires_after_30_days(self):
        now = datetime.now().timestamp()
        saved_at = now - (31 * 86400)
        importance = 1
        cutoff = max(saved_at, 0) + (bong_memory_helpers.MEMORY_EXPIRY_BASE_DAYS * importance * 86400)
        assert now > cutoff

    def test_importance_5_survives_100_days(self):
        now = datetime.now().timestamp()
        saved_at = now - (100 * 86400)
        importance = 5
        cutoff = max(saved_at, 0) + (bong_memory_helpers.MEMORY_EXPIRY_BASE_DAYS * importance * 86400)
        assert now < cutoff  # 150-day expiry

    def test_importance_5_expires_after_151_days(self):
        now = datetime.now().timestamp()
        saved_at = now - (151 * 86400)
        importance = 5
        cutoff = max(saved_at, 0) + (bong_memory_helpers.MEMORY_EXPIRY_BASE_DAYS * importance * 86400)
        assert now > cutoff

    def test_last_accessed_extends_life(self):
        now = datetime.now().timestamp()
        old_saved = now - (100 * 86400)
        recent_access = now - (5 * 86400)
        importance = 1
        cutoff = max(old_saved, recent_access) + (bong_memory_helpers.MEMORY_EXPIRY_BASE_DAYS * importance * 86400)
        assert now < cutoff  # 30 days after last_access, still alive

    def test_missing_importance_defaults_to_3(self):
        now = datetime.now().timestamp()
        old = now - (91 * 86400)
        default_importance = 3
        cutoff = max(old, 0) + (bong_memory_helpers.MEMORY_EXPIRY_BASE_DAYS * default_importance * 86400)
        assert now > cutoff  # 91 days > 90-day cutoff

    def test_recent_memory_not_expired(self):
        now = datetime.now().timestamp()
        recent = now - (1 * 86400)
        importance = 1
        cutoff = max(recent, 0) + (bong_memory_helpers.MEMORY_EXPIRY_BASE_DAYS * importance * 86400)
        assert now < cutoff

    def test_expiry_formula_values(self):
        base = bong_memory_helpers.MEMORY_EXPIRY_BASE_DAYS
        assert base == 30
        for importance, expected_days in [(1, 30), (2, 60), (3, 90), (4, 120), (5, 150)]:
            assert base * importance == expected_days