import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch

from reminders import parse_absolute_time, parse_time_delta


class TestParseTimeDelta:
    def test_minutes(self):
        assert parse_time_delta("30 minutes") == 1800

    def test_hours(self):
        assert parse_time_delta("2 hours") == 7200

    def test_combined(self):
        assert parse_time_delta("1 hour 30 minutes") == 5400

    def test_days(self):
        assert parse_time_delta("1 day") == 86400

    def test_weeks(self):
        assert parse_time_delta("1 week") == 604800

    def test_abbreviations(self):
        assert parse_time_delta("30m") == 1800
        assert parse_time_delta("2h") == 7200

    def test_invalid(self):
        assert parse_time_delta("sometime") is None

    def test_zero(self):
        assert parse_time_delta("0 seconds") is None


class TestParseAbsoluteTime:
    """Test parse_absolute_time with a fixed "now" to make tests deterministic."""

    FIXED_NOW = datetime(2026, 6, 1, 10, 0, 0)  # Monday, June 1 2026 at 10:00 UTC
    UTC_OFFSET = 2.0  # UTC+2 (so local time is 12:00)

    def _parse(self, text, utc_offset=None):
        """Parse with a fixed "now" for deterministic tests."""
        with patch("reminders.datetime") as mock_dt:
            mock_dt.utcnow.return_value = self.FIXED_NOW
            # Also patch datetime.now() used in _check_reminders etc.
            mock_dt.now.return_value = self.FIXED_NOW
            # Allow real datetime construction inside the function
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            # timedelta should still work
            mock_dt.timedelta = timedelta
            return parse_absolute_time(text, utc_offset)

    def test_tomorrow_at_3pm(self):
        result = self._parse("tomorrow at 3pm", utc_offset=2.0)
        assert result is not None
        expected_local = datetime(2026, 6, 2, 15, 0, 0)
        expected_utc = expected_local - timedelta(hours=2.0)
        assert abs(result - expected_utc.timestamp()) < 1

    def test_tomorrow_8am(self):
        result = self._parse("tomorrow 8am", utc_offset=2.0)
        assert result is not None
        expected_local = datetime(2026, 6, 2, 8, 0, 0)
        expected_utc = expected_local - timedelta(hours=2.0)
        assert abs(result - expected_utc.timestamp()) < 1

    def test_friday_at_12(self):
        result = self._parse("friday at 12:00", utc_offset=2.0)
        assert result is not None
        expected_local = datetime(2026, 6, 5, 12, 0, 0)
        expected_utc = expected_local - timedelta(hours=2.0)
        assert abs(result - expected_utc.timestamp()) < 1

    def test_next_monday_at_9am(self):
        result = self._parse("next monday at 9am", utc_offset=2.0)
        assert result is not None
        expected_local = datetime(2026, 6, 8, 9, 0, 0)
        expected_utc = expected_local - timedelta(hours=2.0)
        assert abs(result - expected_utc.timestamp()) < 1

    def test_today_at_5pm(self):
        result = self._parse("today at 5pm", utc_offset=2.0)
        assert result is not None
        expected_local = datetime(2026, 6, 1, 17, 0, 0)
        expected_utc = expected_local - timedelta(hours=2.0)
        assert abs(result - expected_utc.timestamp()) < 1

    def test_june_5_at_3pm(self):
        result = self._parse("june 5 at 3pm", utc_offset=2.0)
        assert result is not None
        expected_local = datetime(2026, 6, 5, 15, 0, 0)
        expected_utc = expected_local - timedelta(hours=2.0)
        assert abs(result - expected_utc.timestamp()) < 1

    def test_24h_time(self):
        result = self._parse("tomorrow at 15:00", utc_offset=2.0)
        assert result is not None
        expected_local = datetime(2026, 6, 2, 15, 0, 0)
        expected_utc = expected_local - timedelta(hours=2.0)
        assert abs(result - expected_utc.timestamp()) < 1

    def test_no_timezone_uses_utc(self):
        result = self._parse("tomorrow at 3pm", utc_offset=None)
        assert result is not None
        expected_utc = datetime(2026, 6, 2, 15, 0, 0)
        assert abs(result - expected_utc.timestamp()) < 1

    def test_time_already_passed_today_moves_to_tomorrow(self):
        result = self._parse("8am", utc_offset=2.0)
        assert result is not None
        expected_local = datetime(2026, 6, 2, 8, 0, 0)
        expected_utc = expected_local - timedelta(hours=2.0)
        assert abs(result - expected_utc.timestamp()) < 1

    def test_invalid_text_returns_none(self):
        result = self._parse("sometime maybe", utc_offset=2.0)
        assert result is None

    def test_past_date_returns_none(self):
        result = self._parse("january 1 at 3pm", utc_offset=2.0)
        assert result is None

    def test_negative_timezone_offset(self):
        result = self._parse("tomorrow at 3pm", utc_offset=-5.0)
        assert result is not None
        expected_local = datetime(2026, 6, 2, 15, 0, 0)
        expected_utc = expected_local + timedelta(hours=5.0)
        assert abs(result - expected_utc.timestamp()) < 1