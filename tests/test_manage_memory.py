import argparse
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import bong_memory_helpers
import bong_utilities.manage_memory as mm

NOW = datetime.now().timestamp()


def make_doc(doc_id, text, user_id=None, user_name="", category="fact", importance=3, saved_at=None, last_accessed=None, access_count=0):
    meta = {
        "category": category,
        "importance": importance,
        "saved_at": saved_at or NOW,
        "last_accessed": last_accessed or NOW,
        "access_count": access_count,
    }
    if user_id is not None:
        meta["user_id"] = user_id
        meta["user_name"] = user_name
    doc = MagicMock()
    doc.page_content = text
    doc.metadata = meta
    doc.id = doc_id
    return doc


class MockCollection:
    def __init__(self, docs=None):
        if docs is None:
            docs = []
        self.ids = [d.id for d in docs]
        self.texts = [d.page_content for d in docs]
        self.metas = [d.metadata for d in docs]
        self._deleted_ids = []

    def get(self, **kwargs):
        return {"ids": self.ids, "documents": self.texts, "metadatas": self.metas}

    def count(self):
        return len(self.ids)

    def delete(self, ids=None):
        if ids:
            self._deleted_ids.extend(ids)

    def update(self, ids=None, metadatas=None):
        pass


def fake_args(**overrides):
    defaults = {
        "list": False,
        "search": None,
        "k": 10,
        "add": None,
        "edit": None,
        "delete": None,
        "forget_user": None,
        "expire": False,
        "user": None,
        "dry_run": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ========== _format_timestamp ==========

class TestFormatTimestamp:
    def test_none_returns_never(self):
        result = mm._format_timestamp(None)
        assert "never" in result

    def test_zero_returns_never(self):
        result = mm._format_timestamp(0)
        assert "never" in result

    def test_valid_timestamp(self):
        result = mm._format_timestamp(NOW)
        assert datetime.now().strftime("%Y-%m-%d") in result

    def test_invalid_string(self):
        result = mm._format_timestamp("not_a_number")
        assert "unknown" in result

    def test_negative_timestamp(self):
        result = mm._format_timestamp(-1)
        assert "unknown" in result or "never" in result or result.strip()


# ========== _format_importance ==========

class TestFormatImportance:
    def test_importance_1(self):
        result = mm._format_importance(1)
        assert "(1)" in result

    def test_importance_5(self):
        result = mm._format_importance(5)
        assert "(5)" in result

    def test_importance_none(self):
        result = mm._format_importance(None)
        assert "(3)" in result

    def test_importance_string(self):
        result = mm._format_importance("4")
        assert "(4)" in result

    def test_importance_invalid_string(self):
        result = mm._format_importance("abc")
        assert "(3)" in result


# ========== _format_category ==========

class TestFormatCategory:
    def test_preference(self):
        result = mm._format_category("preference")
        assert "preference" in result

    def test_fact(self):
        result = mm._format_category("fact")
        assert "fact" in result

    def test_none_defaults_to_fact(self):
        result = mm._format_category(None)
        assert "fact" in result

    def test_empty_string_defaults_to_fact(self):
        result = mm._format_category("")
        assert "fact" in result

    def test_unknown_category(self):
        result = mm._format_category("custom_thing")
        assert "custom_thing" in result


# ========== cmd_list ==========

class TestCmdList:
    def test_empty_db(self):
        buf = StringIO()
        with patch.object(mm, '_get_all_memories', return_value=[]), \
             patch('sys.stdout', buf):
            mm.cmd_list(fake_args(list=True))
        assert "No memories stored" in buf.getvalue()

    def test_with_memories(self):
        items = [
            ("m1", "Eve likes dubstep", {"user_id": 111, "user_name": "Eve", "category": "preference", "importance": 4, "saved_at": NOW, "last_accessed": NOW, "access_count": 2}),
            ("m2", "Radon likes cars", {"user_id": 222, "user_name": "Radon", "category": "fact", "importance": 2, "saved_at": NOW, "last_accessed": NOW, "access_count": 0}),
        ]
        buf = StringIO()
        with patch.object(mm, '_get_all_memories', return_value=[(d, t, m) for d, t, m in items]), \
             patch('sys.stdout', buf):
            mm.cmd_list(fake_args(list=True))
        output = buf.getvalue()
        assert "Total: 2 memories" in output
        assert "Eve likes dubstep" in output
        assert "Radon likes cars" in output

    def test_user_filter(self):
        items = [
            ("m1", "Eve likes dubstep", {"user_id": 111, "user_name": "Eve", "category": "preference", "importance": 4, "saved_at": NOW, "last_accessed": NOW, "access_count": 0}),
        ]
        buf = StringIO()
        with patch.object(mm, '_get_all_memories', return_value=items) as mock_get, \
             patch.object(mm, '_resolve_user', return_value=(111, "Eve")), \
             patch('sys.stdout', buf):
            mm.cmd_list(fake_args(list=True, user="Eve"))
        mock_get.assert_called_once_with(111)

    def test_user_not_found(self):
        buf = StringIO()
        with patch.object(mm, '_resolve_user', return_value=(None, None)), \
             patch('sys.stdout', buf):
            mm.cmd_list(fake_args(list=True, user="Nobody"))
        assert "not found" in buf.getvalue().lower()


# ========== cmd_search ==========

class TestCmdSearch:
    def test_basic_search(self):
        doc = make_doc("m1", "Eve likes dubstep", user_id=111, user_name="Eve")
        mock_db = MagicMock()
        mock_db.similarity_search_with_relevance_scores.return_value = [(doc, 0.85)]
        buf = StringIO()
        with patch.object(bong_memory_helpers, '_vector_db', mock_db), \
             patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x), \
             patch('sys.stdout', buf):
            mm.cmd_search(fake_args(search="dubstep"))
        output = buf.getvalue()
        assert "dubstep" in output
        assert "0.85" in output

    def test_no_results(self):
        mock_db = MagicMock()
        mock_db.similarity_search_with_relevance_scores.return_value = []
        buf = StringIO()
        with patch.object(bong_memory_helpers, '_vector_db', mock_db), \
             patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x), \
             patch('sys.stdout', buf):
            mm.cmd_search(fake_args(search="nonexistent"))
        assert "No results found" in buf.getvalue()

    def test_search_with_user_filter(self):
        doc = make_doc("m1", "Eve likes dubstep", user_id=111, user_name="Eve")
        mock_db = MagicMock()
        mock_db.similarity_search_with_relevance_scores.return_value = [(doc, 0.85)]
        buf = StringIO()
        with patch.object(bong_memory_helpers, '_vector_db', mock_db), \
             patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x), \
             patch.object(mm, '_resolve_user', return_value=(111, "Eve")), \
             patch('sys.stdout', buf):
            mm.cmd_search(fake_args(search="dubstep", user="Eve"))
        mock_db.similarity_search_with_relevance_scores.assert_called_once_with(
            "dubstep", k=10, filter={"user_id": 111}
        )


# ========== cmd_delete — the index bug ==========

class TestCmdDeleteIndexBug:
    def test_index_matches_after_relevance_filter(self):
        docs = [
            make_doc("m1", "high relevance", user_id=111, user_name="Eve", category="fact", importance=3),
            make_doc("m2", "low relevance", user_id=222, user_name="Radon", category="fact", importance=1),
            make_doc("m3", "medium relevance", user_id=333, user_name="Not Reed", category="fact", importance=2),
        ]
        docs[1].metadata["user_name"] = "Radon"
        docs[2].metadata["user_name"] = "Not Reed"

        results = [
            (docs[0], 0.85),
            (docs[1], 0.3),
            (docs[2], 0.72),
        ]

        mock_db = MagicMock()
        mock_db.similarity_search_with_relevance_scores.return_value = results

        buf = StringIO()
        with patch.object(bong_memory_helpers, '_vector_db', mock_db), \
             patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x), \
             patch('bong_utilities.manage_memory._get_all_memories', return_value=[]), \
             patch('builtins.input', return_value='yes'), \
             patch('sys.stdout', buf):
            mm.cmd_delete(fake_args(delete="test query"))

        output = buf.getvalue()
        shown_lines = [l for l in output.split('\n') if '[0.' in l]
        assert len(shown_lines) == 2
        assert "high relevance" in output
        assert "medium relevance" in output

    def test_delete_by_index(self):
        items = [
            ("m1", "memory one", {"user_id": 111, "user_name": "Eve", "category": "fact", "importance": 3}),
        ]
        col = MockCollection()
        buf = StringIO()
        with patch.object(mm, '_get_all_memories', return_value=items), \
             patch.object(mm, '_get_collection', return_value=col), \
             patch('builtins.input', return_value='yes'), \
             patch('sys.stdout', buf):
            mm.cmd_delete(fake_args(delete="1"))
        assert "1 memory" in buf.getvalue()
        assert "m1" in col._deleted_ids

    def test_delete_index_out_of_range(self):
        items = [
            ("m1", "only one memory", {"user_id": 111, "user_name": "Eve"}),
        ]

        buf = StringIO()
        with patch.object(mm, '_get_all_memories', return_value=items), \
             patch('sys.stdout', buf):
            mm.cmd_delete(fake_args(delete="5"))

        assert "out of range" in buf.getvalue().lower() or "range" in buf.getvalue().lower()

    def test_delete_empty_db(self):
        buf = StringIO()
        with patch.object(mm, '_get_all_memories', return_value=[]), \
             patch('sys.stdout', buf):
            mm.cmd_delete(fake_args(delete="1"))

        assert "No memories stored" in buf.getvalue()

    def test_delete_confirmed_with_yes(self):
        items = [
            ("m1", "remember this", {"user_id": 111, "user_name": "Eve", "category": "fact", "importance": 3}),
        ]
        mock_col = MagicMock()

        with patch.object(mm, '_get_all_memories', return_value=items), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch('builtins.input', return_value='yes'), \
             patch('builtins.print'):
            mm.cmd_delete(fake_args(delete="1"))
        mock_col.delete.assert_called_once_with(ids=["m1"])

    def test_delete_cancelled(self):
        items = [
            ("m1", "remember this", {"user_id": 111, "user_name": "Eve"}),
        ]
        mock_col = MagicMock()

        buf = StringIO()
        with patch.object(mm, '_get_all_memories', return_value=items), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch('builtins.input', return_value='no'), \
             patch('sys.stdout', buf):
            mm.cmd_delete(fake_args(delete="1"))

        mock_col.delete.assert_not_called()
        assert "Cancelled" in buf.getvalue()

    def test_delete_dry_run(self):
        items = [
            ("m1", "remember this", {"user_id": 111, "user_name": "Eve", "category": "fact", "importance": 3, "saved_at": NOW, "last_accessed": NOW, "access_count": 0}),
        ]
        mock_col = MagicMock()

        buf = StringIO()
        with patch.object(mm, '_get_all_memories', return_value=items), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch('builtins.input', return_value='yes'), \
             patch('sys.stdout', buf):
            mm.cmd_delete(fake_args(delete="1", dry_run=True))

        mock_col.delete.assert_not_called()
        assert "Dry run" in buf.getvalue()


# ========== cmd_forget_user ==========

class TestCmdForgetUser:
    def test_forget_user_confirmed(self):
        items = [
            ("m1", "Eve fact 1", {"user_id": 111, "user_name": "Eve", "category": "fact", "importance": 3}),
            ("m2", "Eve fact 2", {"user_id": 111, "user_name": "Eve", "category": "preference", "importance": 4}),
        ]
        mock_col = MagicMock()

        with patch.object(mm, '_resolve_user', return_value=(111, "Eve")), \
             patch.object(mm, '_get_all_memories', return_value=items), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch('builtins.input', return_value='Eve'), \
             patch('builtins.print'):
            mm.cmd_forget_user(fake_args(forget_user="Eve"))
        mock_col.delete.assert_called_once_with(ids=["m1", "m2"])

    def test_forget_user_cancelled(self):
        items = [
            ("m1", "Eve fact", {"user_id": 111, "user_name": "Eve", "category": "fact", "importance": 3}),
        ]
        mock_col = MagicMock()

        with patch.object(mm, '_resolve_user', return_value=(111, "Eve")), \
             patch.object(mm, '_get_all_memories', return_value=items), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch('builtins.input', return_value='wrong_name'), \
             patch('builtins.print'):
            mm.cmd_forget_user(fake_args(forget_user="Eve"))
        mock_col.delete.assert_not_called()

    def test_forget_user_not_found(self):
        buf = StringIO()
        with patch.object(mm, '_resolve_user', return_value=(None, None)), \
             patch('sys.stdout', buf):
            mm.cmd_forget_user(fake_args(forget_user="Nobody"))
        assert "not found" in buf.getvalue().lower()

    def test_forget_user_no_memories(self):
        mock_col = MagicMock()

        buf = StringIO()
        with patch.object(mm, '_resolve_user', return_value=(111, "Eve")), \
             patch.object(mm, '_get_all_memories', return_value=[]), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch('sys.stdout', buf):
            mm.cmd_forget_user(fake_args(forget_user="Eve"))

        assert "No memories found" in buf.getvalue()

    def test_forget_user_dry_run(self):
        items = [
            ("m1", "Eve fact", {"user_id": 111, "user_name": "Eve", "category": "fact", "importance": 3}),
        ]
        mock_col = MagicMock()

        buf = StringIO()
        with patch.object(mm, '_resolve_user', return_value=(111, "Eve")), \
             patch.object(mm, '_get_all_memories', return_value=items), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch('sys.stdout', buf):
            mm.cmd_forget_user(fake_args(forget_user="Eve", dry_run=True))

        mock_col.delete.assert_not_called()
        assert "Dry run" in buf.getvalue()


# ========== cmd_expire ==========

class TestCmdExpire:
    def _make_col(self, docs):
        col = MagicMock()
        col.get.return_value = {
            "ids": [d.id for d in docs],
            "metadatas": [d.metadata for d in docs],
        }
        return col

    def test_no_expired_memories(self):
        doc = make_doc("m1", "fresh", importance=5, saved_at=NOW, last_accessed=NOW)
        col = self._make_col([doc])
        buf = StringIO()
        with patch.object(mm, '_get_collection', return_value=col), \
             patch('sys.stdout', buf):
            mm.cmd_expire(fake_args(expire=True))
        assert "No expired" in buf.getvalue()

    def test_expired_memories_found(self):
        old_ts = NOW - (200 * 86400)
        expired_meta = {"saved_at": old_ts, "last_accessed": old_ts, "importance": 1, "user_name": "Eve"}
        fresh_meta = {"saved_at": NOW, "last_accessed": NOW, "importance": 5, "user_name": "Radon"}
        docs = [
            make_doc("m1", "old memory", **expired_meta),
            make_doc("m2", "fresh memory", **fresh_meta),
        ]
        col = self._make_col(docs)
        buf = StringIO()
        with patch.object(mm, '_get_collection', return_value=col), \
             patch('builtins.input', return_value='yes'), \
             patch('sys.stdout', buf):
            mm.cmd_expire(fake_args(expire=True))
        output = buf.getvalue()
        assert "1 expired" in output
        col.delete.assert_called_once()

    def test_expire_dry_run(self):
        old_ts = NOW - (200 * 86400)
        expired_meta = {"saved_at": old_ts, "last_accessed": old_ts, "importance": 1, "user_name": "Eve"}
        docs = [make_doc("m1", "old memory", **expired_meta)]
        col = self._make_col(docs)
        buf = StringIO()
        with patch.object(mm, '_get_collection', return_value=col), \
             patch('sys.stdout', buf):
            mm.cmd_expire(fake_args(expire=True, dry_run=True))
        output = buf.getvalue()
        assert "Dry run" in output
        col.delete.assert_not_called()

    def test_expire_cancelled(self):
        old_ts = NOW - (200 * 86400)
        expired_meta = {"saved_at": old_ts, "last_accessed": old_ts, "importance": 1, "user_name": "Eve"}
        docs = [make_doc("m1", "old memory", **expired_meta)]
        col = self._make_col(docs)
        buf = StringIO()
        with patch.object(mm, '_get_collection', return_value=col), \
             patch('builtins.input', return_value='no'), \
             patch('sys.stdout', buf):
            mm.cmd_expire(fake_args(expire=True))
        col.delete.assert_not_called()
        assert "Cancelled" in buf.getvalue()

    def test_importance_5_survives_100_days(self):
        old_ts = NOW - (100 * 86400)
        meta = {"saved_at": old_ts, "last_accessed": old_ts, "importance": 5, "user_name": "Radon"}
        docs = [make_doc("m1", "important memory", **meta)]
        col = self._make_col(docs)
        buf = StringIO()
        with patch.object(mm, '_get_collection', return_value=col), \
             patch('sys.stdout', buf):
            mm.cmd_expire(fake_args(expire=True))
        assert "No expired" in buf.getvalue()


# ========== _resolve_user ==========

class TestResolveUser:
    @patch('user_data.load_users')
    @patch('user_data._user_data', {111: {"display_name": "Eve", "allowed": ["admin"]}})
    def test_numeric_id_found(self, mock_load):
        uid, name = mm._resolve_user("111")
        assert uid == 111
        assert name == "Eve"

    @patch('user_data.load_users')
    @patch('user_data._user_data', {111: {"display_name": "Eve", "allowed": ["admin"]}})
    def test_numeric_id_not_found(self, mock_load):
        uid, name = mm._resolve_user("999")
        assert uid == 999
        assert name == "999"

    @patch('bong_memory_helpers.resolve_name_to_id', return_value=(222, None))
    @patch('user_data.load_users')
    @patch('user_data._user_data', {222: {"display_name": "Radon", "allowed": ["admin"]}})
    def test_name_resolved(self, mock_load, mock_resolve):
        uid, name = mm._resolve_user("Radon")
        assert uid == 222
        assert name == "Radon"

    @patch('bong_memory_helpers.resolve_name_to_id', return_value=(None, None))
    def test_name_not_found(self, mock_resolve):
        uid, name = mm._resolve_user("Nobody")
        assert uid is None
        assert name is None


# ========== cmd_add ==========

class TestCmdAdd:
    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    def test_add_with_defaults(self, mock_clean, mock_add):
        inputs = ["fact", "3", "", "y"]
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print'):
            mm.cmd_add(fake_args(add="test memory"))
        mock_add.assert_called_once()
        call_args = mock_add.call_args
        texts = call_args[0][0]
        metas = call_args[1]["metadatas"]
        assert texts == ["test memory"]
        assert metas[0]["category"] == "fact"
        assert metas[0]["importance"] == 3

    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    def test_add_with_category_and_importance(self, mock_clean, mock_add):
        inputs = ["preference", "5", "", "y"]
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print'):
            mm.cmd_add(fake_args(add="I love cats"))
        mock_add.assert_called_once()
        metas = mock_add.call_args[1]["metadatas"]
        assert metas[0]["category"] == "preference"
        assert metas[0]["importance"] == 5

    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    def test_add_cancelled(self, mock_clean, mock_add):
        inputs = ["fact", "3", "", "n"]
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print'):
            mm.cmd_add(fake_args(add="test memory"))
        mock_add.assert_not_called()

    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: "")
    def test_add_empty_after_clean(self, mock_clean):
        buf = StringIO()
        with patch('builtins.input'), \
             patch('sys.stdout', buf):
            mm.cmd_add(fake_args(add="   "))
        assert "empty" in buf.getvalue().lower()

    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    @patch.object(mm, '_resolve_user', return_value=(111, "Eve"))
    def test_add_with_user(self, mock_resolve, mock_clean, mock_add):
        inputs = ["preference", "4", "y"]
        args = fake_args(add="I like dubstep", user="Eve")
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print'):
            mm.cmd_add(args)
        mock_add.assert_called_once()
        metas = mock_add.call_args[1]["metadatas"]
        assert metas[0]["user_id"] == 111
        assert metas[0]["user_name"] == "Eve"

    def test_add_user_not_found(self):
        buf = StringIO()
        with patch.object(mm, '_resolve_user', return_value=(None, None)), \
             patch('sys.stdout', buf):
            mm.cmd_add(fake_args(add="test", user="Nobody"))
        assert "not found" in buf.getvalue().lower()

    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    def test_add_invalid_category_defaults_to_fact(self, mock_clean, mock_add):
        inputs = ["invalid_category", "3", "", "y"]
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print'):
            mm.cmd_add(fake_args(add="test memory"))
        mock_add.assert_called_once()
        metas = mock_add.call_args[1]["metadatas"]
        assert metas[0]["category"] == "fact"

    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    def test_add_importance_clamped(self, mock_clean, mock_add):
        inputs = ["fact", "10", "", "y"]
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print'):
            mm.cmd_add(fake_args(add="test memory"))
        mock_add.assert_called_once()
        metas = mock_add.call_args[1]["metadatas"]
        assert metas[0]["importance"] == 5


# ========== cmd_edit ==========

class TestCmdEdit:
    def test_edit_by_index_metadata_only(self):
        meta = {"user_id": 111, "user_name": "Eve", "category": "fact", "importance": 3, "saved_at": NOW, "last_accessed": NOW, "access_count": 5}
        items = [("m1", "test memory", meta)]
        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": ["m1"], "documents": ["test memory"], "metadatas": [meta]}

        inputs = ["3", "5", "5", "y"]
        buf = StringIO()
        with patch.object(mm, '_get_all_memories', return_value=items), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch('builtins.input', side_effect=inputs), \
             patch('sys.stdout', buf):
            mm.cmd_edit(fake_args(edit="1"))
        output = buf.getvalue()
        assert "Updated" in output

    def test_edit_index_out_of_range(self):
        items = [("m1", "only one", {"category": "fact"})]
        buf = StringIO()
        with patch.object(mm, '_get_all_memories', return_value=items), \
             patch('sys.stdout', buf):
            mm.cmd_edit(fake_args(edit="5"))
        assert "out of range" in buf.getvalue().lower() or "range" in buf.getvalue().lower()

    def test_edit_cancelled(self):
        meta = {"user_id": 111, "user_name": "Eve", "category": "fact", "importance": 3}
        items = [("m1", "test memory", meta)]

        inputs = ["6"]
        buf = StringIO()
        with patch.object(mm, '_get_all_memories', return_value=items), \
             patch.object(mm, '_get_collection', return_value=MagicMock()), \
             patch('builtins.input', side_effect=inputs), \
             patch('sys.stdout', buf):
            mm.cmd_edit(fake_args(edit="1"))
        assert "Cancelled" in buf.getvalue()

    def test_edit_empty_db(self):
        buf = StringIO()
        with patch.object(mm, '_get_all_memories', return_value=[]), \
             patch('sys.stdout', buf):
            mm.cmd_edit(fake_args(edit="1"))
        assert "No memories" in buf.getvalue() or "out of range" in buf.getvalue().lower()