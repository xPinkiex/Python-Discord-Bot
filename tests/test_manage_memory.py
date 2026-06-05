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
        "user_name": user_name,
    }
    if user_id is not None:
        meta["user_id"] = user_id
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
        self._added = []

    def get(self, include=None, **kwargs):
        result = {"ids": self.ids, "documents": self.texts, "metadatas": self.metas}
        if include and "embeddings" in include:
            result["embeddings"] = [[0.1] * 768 for _ in self.ids]
        return result

    def count(self):
        return len(self.ids)

    def delete(self, ids=None):
        if ids:
            self._deleted_ids.extend(ids)

    def update(self, ids=None, metadatas=None):
        pass

    def add(self, ids=None, documents=None, metadatas=None, embeddings=None):
        self._added.append({"ids": ids, "documents": documents, "metadatas": metadatas, "embeddings": embeddings})


def fake_args(**overrides):
    defaults = {
        "list": False,
        "search": None,
        "k": 10,
        "add": None,
        "category": None,
        "importance": None,
        "about": None,
        "edit": None,
        "delete": None,
        "forget_user": None,
        "expire": False,
        "user": None,
        "dry_run": False,
        "backup": False,
        "restore": None,
        "drop": False,
        "fast": False,
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
             patch.object(mm, '_auto_backup', return_value=None), \
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
             patch.object(mm, '_auto_backup', return_value=None), \
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
             patch.object(mm, '_auto_backup', return_value=None), \
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
             patch.object(mm, '_auto_backup', return_value=None), \
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


# ========== _resolve_category ==========

class TestResolveCategory:
    def test_index_1(self):
        cat, valid = mm._resolve_category("1")
        assert cat == "preference"
        assert valid is True

    def test_index_2(self):
        cat, valid = mm._resolve_category("2")
        assert cat == "fact"
        assert valid is True

    def test_index_3(self):
        cat, valid = mm._resolve_category("3")
        assert cat == "relationship"
        assert valid is True

    def test_index_4(self):
        cat, valid = mm._resolve_category("4")
        assert cat == "inside_joke"
        assert valid is True

    def test_index_5(self):
        cat, valid = mm._resolve_category("5")
        assert cat == "instruction"
        assert valid is True

    def test_name_preference(self):
        cat, valid = mm._resolve_category("preference")
        assert cat == "preference"
        assert valid is True

    def test_name_fact(self):
        cat, valid = mm._resolve_category("fact")
        assert cat == "fact"
        assert valid is True

    def test_name_relationship(self):
        cat, valid = mm._resolve_category("relationship")
        assert cat == "relationship"
        assert valid is True

    def test_name_inside_joke(self):
        cat, valid = mm._resolve_category("inside_joke")
        assert cat == "inside_joke"
        assert valid is True

    def test_name_instruction(self):
        cat, valid = mm._resolve_category("instruction")
        assert cat == "instruction"
        assert valid is True

    def test_invalid_returns_fact(self):
        cat, valid = mm._resolve_category("xyz")
        assert cat == "fact"
        assert valid is False

    def test_empty_returns_fact(self):
        cat, valid = mm._resolve_category("")
        assert cat == "fact"
        assert valid is False

    def test_case_insensitive(self):
        cat, valid = mm._resolve_category("PREFERENCE")
        assert cat == "preference"
        assert valid is True

    def test_whitespace(self):
        cat, valid = mm._resolve_category("  1  ")
        assert cat == "preference"
        assert valid is True



class TestCmdAdd:
    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    def test_add_with_defaults(self, mock_clean, mock_add):
        inputs = ["", "", "", "y"]
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
        assert metas[0]["user_name"] == ""

    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    def test_add_with_category_and_importance(self, mock_clean, mock_add):
        inputs = ["1", "5", "", "y"]
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
        inputs = ["", "", "", "n"]
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print'):
            mm.cmd_add(fake_args(add="test memory"))
        mock_add.assert_not_called()

    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: "")
    def test_add_empty_after_clean(self, mock_clean):
        buf = StringIO()
        with patch('builtins.input', side_effect=["", "", ""]), \
             patch('sys.stdout', buf):
            mm.cmd_add(fake_args(add="   "))
        assert "empty" in buf.getvalue().lower()

    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    @patch.object(mm, '_resolve_user', return_value=(111, "Eve"))
    def test_add_with_user(self, mock_resolve, mock_clean, mock_add):
        inputs = ["1", "4", "y"]
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
             patch('builtins.input', side_effect=["", "", ""]), \
             patch('sys.stdout', buf):
            mm.cmd_add(fake_args(add="test", user="Nobody"))
        assert "not found" in buf.getvalue().lower()

    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    def test_add_invalid_category_defaults_to_fact(self, mock_clean, mock_add):
        inputs = ["invalid_category", "", "", "y"]
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print'):
            mm.cmd_add(fake_args(add="test memory"))
        mock_add.assert_called_once()
        metas = mock_add.call_args[1]["metadatas"]
        assert metas[0]["category"] == "fact"

    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    def test_add_importance_clamped(self, mock_clean, mock_add):
        inputs = ["", "10", "", "y"]
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print'):
            mm.cmd_add(fake_args(add="test memory"))
        mock_add.assert_called_once()
        metas = mock_add.call_args[1]["metadatas"]
        assert metas[0]["importance"] == 5

    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    def test_add_category_flag(self, mock_clean, mock_add):
        inputs = ["", "", "y"]
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print'):
            mm.cmd_add(fake_args(add="test memory", category="1"))
        mock_add.assert_called_once()
        metas = mock_add.call_args[1]["metadatas"]
        assert metas[0]["category"] == "preference"

    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    def test_add_category_flag_by_name(self, mock_clean, mock_add):
        inputs = ["", "", "y"]
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print'):
            mm.cmd_add(fake_args(add="test memory", category="instruction"))
        mock_add.assert_called_once()
        metas = mock_add.call_args[1]["metadatas"]
        assert metas[0]["category"] == "instruction"

    def test_add_category_flag_invalid_aborts(self):
        buf = StringIO()
        with patch('builtins.input'), \
             patch('sys.stdout', buf):
            mm.cmd_add(fake_args(add="test memory", category="xyz"))
        assert "invalid category" in buf.getvalue().lower()

    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    def test_add_importance_flag(self, mock_clean, mock_add):
        inputs = ["", "", "y"]
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print'):
            mm.cmd_add(fake_args(add="test memory", importance=5))
        mock_add.assert_called_once()
        metas = mock_add.call_args[1]["metadatas"]
        assert metas[0]["importance"] == 5

    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    @patch.object(mm, '_resolve_user', return_value=(111, "Eve"))
    def test_add_about_flag(self, mock_resolve, mock_clean, mock_add):
        inputs = ["y"]
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print'):
            mm.cmd_add(fake_args(add="test", category="2", importance=3, about="Eve"))
        mock_add.assert_called_once()
        metas = mock_add.call_args[1]["metadatas"]
        assert metas[0]["user_id"] == 111
        assert metas[0]["user_name"] == "Eve"

    def test_add_about_flag_not_found_aborts(self):
        buf = StringIO()
        with patch.object(mm, '_resolve_user', return_value=(None, None)), \
             patch('builtins.input'), \
             patch('sys.stdout', buf):
            mm.cmd_add(fake_args(add="test", category="2", importance=3, about="Nobody"))
        assert "not found" in buf.getvalue().lower()

    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    def test_add_about_interactive_general(self, mock_clean, mock_add):
        inputs = ["", "", "", "y"]
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print'):
            mm.cmd_add(fake_args(add="test memory"))
        mock_add.assert_called_once()
        metas = mock_add.call_args[1]["metadatas"]
        assert "user_id" not in metas[0]
        assert metas[0]["user_name"] == ""

    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    @patch.object(mm, '_resolve_user', return_value=(111, "Eve"))
    def test_add_about_interactive_resolved(self, mock_resolve, mock_clean, mock_add):
        inputs = ["2", "3", "Eve", "y"]
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print'):
            mm.cmd_add(fake_args(add="test memory"))
        mock_add.assert_called_once()
        metas = mock_add.call_args[1]["metadatas"]
        assert metas[0]["user_id"] == 111
        assert metas[0]["user_name"] == "Eve"

    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    @patch.object(mm, '_resolve_user')
    def test_add_about_interactive_not_found_then_id(self, mock_resolve, mock_clean, mock_add):
        def resolve_side_effect(arg):
            if arg == "Unknown":
                return (None, None)
            return (999, "CustomUser")
        mock_resolve.side_effect = resolve_side_effect
        inputs = ["2", "3", "Unknown", "999", "CustomUser", "y"]
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print'):
            mm.cmd_add(fake_args(add="test memory"))
        mock_add.assert_called_once()
        metas = mock_add.call_args[1]["metadatas"]
        assert metas[0]["user_id"] == 999
        assert metas[0]["user_name"] == "CustomUser"

    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    @patch.object(mm, '_resolve_user')
    def test_add_about_interactive_not_found_then_general(self, mock_resolve, mock_clean, mock_add):
        mock_resolve.return_value = (None, None)
        inputs = ["2", "3", "Unknown", "", "y"]
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print'):
            mm.cmd_add(fake_args(add="test memory"))
        mock_add.assert_called_once()
        metas = mock_add.call_args[1]["metadatas"]
        assert "user_id" not in metas[0]
        assert metas[0]["user_name"] == ""

    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    def test_add_general_fact_always_has_user_name(self, mock_clean, mock_add):
        inputs = ["", "", "", "y"]
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print'):
            mm.cmd_add(fake_args(add="general fact"))
        mock_add.assert_called_once()
        metas = mock_add.call_args[1]["metadatas"]
        assert "user_name" in metas[0]
        assert metas[0]["user_name"] == ""

    @patch.object(bong_memory_helpers._vector_db, 'add_texts')
    @patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x)
    @patch.object(mm, '_resolve_user', return_value=(111, "Eve"))
    def test_add_user_flag_used_as_about(self, mock_resolve, mock_clean, mock_add):
        inputs = ["2", "3", "y"]
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print'):
            mm.cmd_add(fake_args(add="test memory", user="Eve"))
        mock_add.assert_called_once()
        metas = mock_add.call_args[1]["metadatas"]
        assert metas[0]["user_id"] == 111
        assert metas[0]["user_name"] == "Eve"


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


# ========== cmd_backup ==========

class TestCmdBackup:
    def test_backup_creates_json(self, tmp_path):
        col = MockCollection()
        col.ids = ["m1", "m2"]
        col.texts = ["memory one", "memory two"]
        col.metas = [{"category": "fact"}, {"category": "preference"}]
        with patch.object(mm, '_get_collection', return_value=col), \
             patch.object(mm, 'BACKUP_DIR', tmp_path):
            result = mm.cmd_backup(fake_args(backup=True))
        assert result is not None
        assert Path(result).exists()
        import json
        with open(result) as f:
            data = json.load(f)
        assert data["memory_count"] == 2
        assert data["model"] == "nomic-embed-text"
        assert len(data["ids"]) == 2
        assert len(data["embeddings"]) == 2

    def test_backup_empty_db(self):
        col = MockCollection()
        buf = StringIO()
        with patch.object(mm, '_get_collection', return_value=col), \
             patch('sys.stdout', buf):
            result = mm.cmd_backup(fake_args(backup=True))
        assert result is None
        assert "No memories" in buf.getvalue()

    def test_backup_quiet_mode(self, tmp_path):
        col = MockCollection()
        col.ids = ["m1"]
        col.texts = ["test"]
        col.metas = [{"category": "fact"}]
        buf = StringIO()
        with patch.object(mm, '_get_collection', return_value=col), \
             patch.object(mm, 'BACKUP_DIR', tmp_path), \
             patch('sys.stdout', buf):
            result = mm.cmd_backup(fake_args(backup=True), _quiet=True)
        assert result is not None
        assert buf.getvalue() == ""

    def test_backup_file_naming(self, tmp_path):
        col = MockCollection()
        col.ids = ["m1"]
        col.texts = ["test"]
        col.metas = [{"category": "fact"}]
        with patch.object(mm, '_get_collection', return_value=col), \
             patch.object(mm, 'BACKUP_DIR', tmp_path):
            result = mm.cmd_backup(fake_args(backup=True))
        filename = Path(result).name
        assert filename.startswith("memory_backup_")
        assert filename.endswith(".json")

    def test_backup_atomic_write(self, tmp_path):
        col = MockCollection()
        col.ids = ["m1"]
        col.texts = ["test"]
        col.metas = [{"category": "fact"}]
        with patch.object(mm, '_get_collection', return_value=col), \
             patch.object(mm, 'BACKUP_DIR', tmp_path):
            result = mm.cmd_backup(fake_args(backup=True))
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1

    def test_backup_creates_directory(self, tmp_path):
        nested = tmp_path / "new_dir"
        col = MockCollection()
        col.ids = ["m1"]
        col.texts = ["test"]
        col.metas = [{"category": "fact"}]
        with patch.object(mm, '_get_collection', return_value=col), \
             patch.object(mm, 'BACKUP_DIR', nested):
            result = mm.cmd_backup(fake_args(backup=True))
        assert nested.exists()
        assert result is not None

    def test_backup_disk_error(self, tmp_path):
        col = MockCollection()
        col.ids = ["m1"]
        col.texts = ["test"]
        col.metas = [{"category": "fact"}]
        buf = StringIO()
        with patch.object(mm, '_get_collection', return_value=col), \
             patch.object(mm, 'BACKUP_DIR', tmp_path), \
             patch('builtins.open', side_effect=OSError("disk full")), \
             patch('sys.stdout', buf):
            result = mm.cmd_backup(fake_args(backup=True))
        assert result is None
        assert "failed" in buf.getvalue().lower()

    def test_backup_preserves_metadata(self, tmp_path):
        col = MockCollection()
        col.ids = ["m1"]
        col.texts = ["test memory"]
        col.metas = [{"user_id": 111, "user_name": "Eve", "category": "preference", "importance": 5}]
        with patch.object(mm, '_get_collection', return_value=col), \
             patch.object(mm, 'BACKUP_DIR', tmp_path):
            result = mm.cmd_backup(fake_args(backup=True))
        import json
        with open(result) as f:
            data = json.load(f)
        assert data["metadatas"][0]["user_name"] == "Eve"
        assert data["metadatas"][0]["importance"] == 5

    def test_backup_no_tmp_on_success(self, tmp_path):
        col = MockCollection()
        col.ids = ["m1"]
        col.texts = ["test"]
        col.metas = [{"category": "fact"}]
        with patch.object(mm, '_get_collection', return_value=col), \
             patch.object(mm, 'BACKUP_DIR', tmp_path):
            mm.cmd_backup(fake_args(backup=True))
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0


# ========== cmd_restore ==========

class TestCmdRestore:
    def _make_backup_json(self, tmp_path, ids=None, documents=None, metadatas=None, embeddings=None, extra_keys=None):
        import json
        if ids is None:
            ids = ["m1"]
        if documents is None:
            documents = ["test memory"]
        if metadatas is None:
            metadatas = [{"category": "fact", "importance": 3}]
        if embeddings is None:
            embeddings = [[0.1] * 768 for _ in ids]
        data = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "embeddings": embeddings,
            "backup_time": NOW,
            "memory_count": len(ids),
            "model": "nomic-embed-text",
        }
        if extra_keys:
            data.update(extra_keys)
        path = tmp_path / "backup.json"
        with open(path, "w") as f:
            json.dump(data, f)
        return str(path)

    def test_restore_default_reembed(self, tmp_path):
        path = self._make_backup_json(tmp_path)
        mock_db = MagicMock()
        buf = StringIO()
        with patch.object(mm, '_check_ollama', return_value=True), \
             patch.object(bong_memory_helpers, '_vector_db', mock_db), \
             patch.object(mm, '_get_collection', return_value=MagicMock()), \
             patch('builtins.input', return_value='yes'), \
             patch('sys.stdout', buf):
            mm.cmd_restore(fake_args(restore=path))
        mock_db.add_texts.assert_called()
        call_args = mock_db.add_texts.call_args
        assert "embeddings" not in call_args[1]

    def test_restore_fast_mode(self, tmp_path):
        path = self._make_backup_json(tmp_path)
        mock_col = MagicMock()
        buf = StringIO()
        with patch.object(mm, '_check_ollama', return_value=False), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch('builtins.input', return_value='yes'), \
             patch('sys.stdout', buf):
            mm.cmd_restore(fake_args(restore=path, fast=True))
        mock_col.add.assert_called_once()
        call_kwargs = mock_col.add.call_args[1]
        assert "embeddings" in call_kwargs

    def test_restore_file_not_found(self):
        buf = StringIO()
        with patch('sys.stdout', buf):
            mm.cmd_restore(fake_args(restore="/nonexistent/path.json"))
        assert "not found" in buf.getvalue().lower()

    def test_restore_malformed_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json{{{")
        buf = StringIO()
        with patch('sys.stdout', buf):
            mm.cmd_restore(fake_args(restore=str(bad_file)))
        assert "failed" in buf.getvalue().lower() or "error" in buf.getvalue().lower()

    def test_restore_missing_keys(self, tmp_path):
        import json
        path = tmp_path / "incomplete.json"
        with open(path, "w") as f:
            json.dump({"ids": ["m1"], "documents": ["test"]}, f)
        buf = StringIO()
        with patch('sys.stdout', buf):
            mm.cmd_restore(fake_args(restore=str(path)))
        assert "missing" in buf.getvalue().lower()

    def test_restore_mismatched_lengths(self, tmp_path):
        path = self._make_backup_json(tmp_path, ids=["m1", "m2"], documents=["only one"], metadatas=[{"category": "fact"}], embeddings=[[0.1] * 768])
        buf = StringIO()
        with patch('sys.stdout', buf):
            mm.cmd_restore(fake_args(restore=path))
        assert "mismatch" in buf.getvalue().lower() or "corrupted" in buf.getvalue().lower()

    def test_restore_cancelled(self, tmp_path):
        path = self._make_backup_json(tmp_path)
        mock_col = MagicMock()
        buf = StringIO()
        with patch.object(mm, '_check_ollama', return_value=True), \
             patch.object(bong_memory_helpers, '_vector_db', MagicMock()), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch('builtins.input', return_value='no'), \
             patch('sys.stdout', buf):
            mm.cmd_restore(fake_args(restore=path))
        mock_col.add.assert_not_called()
        assert "Cancelled" in buf.getvalue()

    def test_restore_empty_backup(self, tmp_path):
        import json
        path = tmp_path / "empty.json"
        with open(path, "w") as f:
            json.dump({"ids": [], "documents": [], "metadatas": [], "embeddings": []}, f)
        buf = StringIO()
        with patch('sys.stdout', buf):
            mm.cmd_restore(fake_args(restore=str(path)))
        assert "no memories" in buf.getvalue().lower()

    def test_restore_ollama_unreachable_reembed(self, tmp_path):
        path = self._make_backup_json(tmp_path)
        buf = StringIO()
        with patch.object(mm, '_check_ollama', return_value=False), \
             patch('sys.stdout', buf):
            mm.cmd_restore(fake_args(restore=path))
        assert "unreachable" in buf.getvalue().lower() or "cannot re-embed" in buf.getvalue().lower()

    def test_restore_ollama_unreachable_fast(self, tmp_path):
        path = self._make_backup_json(tmp_path)
        mock_col = MagicMock()
        buf = StringIO()
        with patch.object(mm, '_check_ollama', return_value=False), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch('builtins.input', return_value='yes'), \
             patch('sys.stdout', buf):
            mm.cmd_restore(fake_args(restore=path, fast=True))
        assert "Warning" in buf.getvalue() or "warning" in buf.getvalue().lower()

    def test_restore_fast_with_missing_embeddings(self, tmp_path):
        path = self._make_backup_json(tmp_path, embeddings=[None])
        buf = StringIO()
        with patch.object(mm, '_check_ollama', return_value=False), \
             patch.object(mm, '_get_collection', return_value=MagicMock()), \
             patch('sys.stdout', buf):
            mm.cmd_restore(fake_args(restore=path, fast=True))
        assert "missing embeddings" in buf.getvalue().lower() or "Cannot use --fast" in buf.getvalue()

    def test_restore_drop_happy_path(self, tmp_path):
        path = self._make_backup_json(tmp_path, ids=["m1"], documents=["backup memory"], metadatas=[{"category": "fact"}], embeddings=[[0.1] * 768])
        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": ["m1", "m_extra"], "metadatas": [{}, {}]}
        mock_db = MagicMock()
        buf = StringIO()
        with patch.object(mm, '_check_ollama', return_value=True), \
             patch.object(bong_memory_helpers, '_vector_db', mock_db), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch('builtins.input', return_value='yes'), \
             patch('sys.stdout', buf):
            mm.cmd_restore(fake_args(restore=path, drop=True))
        all_deleted = []
        for call in mock_col.delete.call_args_list:
            all_deleted.extend(call[1]["ids"])
        assert "m1" in all_deleted
        assert "m_extra" in all_deleted

    def test_restore_reembed_overwrites_existing(self, tmp_path):
        path = self._make_backup_json(tmp_path, ids=["m1"], documents=["updated memory"], metadatas=[{"category": "fact"}], embeddings=[[0.1] * 768])
        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": ["m1", "m_other"], "metadatas": [{}, {}]}
        mock_db = MagicMock()
        buf = StringIO()
        with patch.object(mm, '_check_ollama', return_value=True), \
             patch.object(bong_memory_helpers, '_vector_db', mock_db), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch('builtins.input', return_value='yes'), \
             patch('sys.stdout', buf):
            mm.cmd_restore(fake_args(restore=path))
        all_deleted = []
        for call in mock_col.delete.call_args_list:
            all_deleted.extend(call[1]["ids"])
        assert "m1" in all_deleted
        assert "m_other" not in all_deleted
        mock_db.add_texts.assert_called()
        assert "overwrite" in buf.getvalue().lower() or len(mock_col.delete.call_args_list) >= 1

    def test_restore_drop_reembed_with_overlapping_ids(self, tmp_path):
        path = self._make_backup_json(tmp_path, ids=["m1", "m2"], documents=["backup1", "backup2"], metadatas=[{"category": "fact"}, {"category": "fact"}], embeddings=[[0.1] * 768, [0.2] * 768])
        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": ["m1", "m_old"], "metadatas": [{}, {}]}
        mock_db = MagicMock()
        buf = StringIO()
        with patch.object(mm, '_check_ollama', return_value=True), \
             patch.object(bong_memory_helpers, '_vector_db', mock_db), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch('builtins.input', return_value='yes'), \
             patch('sys.stdout', buf):
            mm.cmd_restore(fake_args(restore=path, drop=True))
        all_deleted = []
        for call in mock_col.delete.call_args_list:
            all_deleted.extend(call[1]["ids"])
        assert "m1" in all_deleted
        assert "m_old" in all_deleted
        assert mock_db.add_texts.call_count == 2

    def test_restore_drop_insert_fails_no_delete(self, tmp_path):
        path = self._make_backup_json(tmp_path, ids=["m1"], documents=["backup memory"], metadatas=[{"category": "fact"}], embeddings=[[0.1] * 768])
        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": ["m_old"], "metadatas": [{}]}
        mock_db = MagicMock()
        mock_db.add_texts.side_effect = Exception("embedding failed")
        buf = StringIO()
        with patch.object(mm, '_check_ollama', return_value=True), \
             patch.object(bong_memory_helpers, '_vector_db', mock_db), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch('builtins.input', return_value='yes'), \
             patch('sys.stdout', buf):
            mm.cmd_restore(fake_args(restore=path, drop=True))
        mock_col.delete.assert_not_called()
        assert "incomplete" in buf.getvalue().lower() or "failed" in buf.getvalue().lower()

    def test_restore_reembed_partial_failure(self, tmp_path):
        path = self._make_backup_json(tmp_path, ids=["m1", "m2"], documents=["ok", "bad"], metadatas=[{"category": "fact"}, {"category": "fact"}], embeddings=[[0.1] * 768, [0.2] * 768])
        mock_db = MagicMock()
        call_count = [0]

        def add_texts_side_effect(texts, metadatas=None, ids=None):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("embedding failed for m2")
            return ["m1"]

        mock_db.add_texts.side_effect = add_texts_side_effect
        buf = StringIO()
        with patch.object(mm, '_check_ollama', return_value=True), \
             patch.object(bong_memory_helpers, '_vector_db', mock_db), \
             patch.object(mm, '_get_collection', return_value=MagicMock()), \
             patch('builtins.input', return_value='yes'), \
             patch('sys.stdout', buf):
            mm.cmd_restore(fake_args(restore=path))
        output = buf.getvalue()
        assert "1/2" in output or "Restored 1" in output

    def test_restore_old_schema_still_works(self, tmp_path):
        import json
        path = tmp_path / "old_schema.json"
        data = {
            "ids": ["m1"],
            "documents": ["old memory"],
            "metadatas": [{"username": "Eve", "category": "fact", "importance": 3}],
            "embeddings": [[0.1] * 768],
            "backup_time": NOW,
            "memory_count": 1,
            "model": "nomic-embed-text",
        }
        with open(path, "w") as f:
            json.dump(data, f)
        mock_col = MagicMock()
        buf = StringIO()
        with patch.object(mm, '_check_ollama', return_value=False), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch('builtins.input', return_value='yes'), \
             patch('sys.stdout', buf):
            mm.cmd_restore(fake_args(restore=str(path), fast=True))
        mock_col.add.assert_called_once()

    def test_restore_non_json_file(self, tmp_path):
        bad_file = tmp_path / "binary.bin"
        bad_file.write_bytes(b"\x00\x01\x02\x03")
        buf = StringIO()
        with patch('sys.stdout', buf):
            mm.cmd_restore(fake_args(restore=str(bad_file)))
        assert "failed" in buf.getvalue().lower() or "error" in buf.getvalue().lower()


# ========== _auto_backup ==========

class TestAutoBackup:
    def test_auto_backup_called_before_delete(self):
        items = [("m1", "memory", {"user_id": 111, "user_name": "Eve", "category": "fact", "importance": 3})]
        mock_col = MagicMock()
        with patch.object(mm, '_get_all_memories', return_value=items), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch.object(mm, '_auto_backup', return_value="/tmp/backup.json") as mock_ab, \
             patch('builtins.input', return_value='yes'), \
             patch('builtins.print'):
            mm.cmd_delete(fake_args(delete="1"))
        mock_ab.assert_called_once()

    def test_auto_backup_failure_doesnt_block(self):
        items = [("m1", "memory", {"user_id": 111, "user_name": "Eve", "category": "fact", "importance": 3})]
        mock_col = MagicMock()
        buf = StringIO()
        with patch.object(mm, '_get_all_memories', return_value=items), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch.object(mm, 'cmd_backup', side_effect=OSError("disk full")), \
             patch('builtins.input', return_value='yes'), \
             patch('sys.stdout', buf):
            mm.cmd_delete(fake_args(delete="1"))
        mock_col.delete.assert_called_once()
        assert "Warning" in buf.getvalue() or "warning" in buf.getvalue().lower()

    def test_auto_backup_not_called_on_dry_run(self):
        items = [("m1", "memory", {"user_id": 111, "user_name": "Eve", "category": "fact", "importance": 3})]
        mock_col = MagicMock()
        with patch.object(mm, '_get_all_memories', return_value=items), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch.object(mm, '_auto_backup', return_value=None) as mock_ab, \
             patch('builtins.input', return_value='yes'), \
             patch('builtins.print'), \
             patch('sys.stdout', StringIO()):
            mm.cmd_delete(fake_args(delete="1", dry_run=True))
        mock_ab.assert_not_called()

    def test_auto_backup_on_edit_text_change(self):
        meta = {"user_id": 111, "user_name": "Eve", "category": "fact", "importance": 3, "saved_at": NOW, "last_accessed": NOW, "access_count": 0}
        items = [("m1", "old text", meta)]
        mock_col = MagicMock()
        inputs = ["1", "new text", "5", "y"]
        buf = StringIO()
        with patch.object(mm, '_get_all_memories', return_value=items), \
             patch.object(mm, '_get_collection', return_value=mock_col), \
             patch.object(mm, '_auto_backup', return_value=None) as mock_ab, \
             patch.object(bong_memory_helpers, '_clean_for_embedding', side_effect=lambda x: x), \
             patch.object(bong_memory_helpers, '_vector_db', MagicMock()), \
             patch('builtins.input', side_effect=inputs), \
             patch('sys.stdout', buf):
            mm.cmd_edit(fake_args(edit="1"))
        mock_ab.assert_called_once()


# ========== _check_ollama ==========

class TestCheckOllama:
    def test_ollama_reachable(self):
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            assert mm._check_ollama() is True

    def test_ollama_unreachable(self):
        import urllib.error
        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError("no connection")):
            assert mm._check_ollama() is False

    def test_ollama_uses_env_host(self):
        with patch.dict('os.environ', {"OLLAMA_HOST": "192.168.1.100:11434"}), \
             patch('urllib.request.urlopen') as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            mm._check_ollama()
            call_url = mock_urlopen.call_args[0][0].full_url
            assert "192.168.1.100" in call_url