from __future__ import annotations

import json
import logging
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from bot import (
    ApiJournal,
    MemoryStore,
    ReplyService,
    RequestRateLimiter,
    configure_logging,
    safe_url_for_log,
)
from runtime_storage import atomic_write_text, load_json_object, migrate_legacy_runtime_files


class RuntimeStorageTests(unittest.TestCase):
    def test_invalid_primary_recovers_from_last_good_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            atomic_write_text(path, json.dumps({"generation": 1}))
            atomic_write_text(path, json.dumps({"generation": 2}))
            path.write_text("{broken", encoding="utf-8")

            self.assertEqual(load_json_object(path), {"generation": 1})

    def test_legacy_runtime_file_is_migrated_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            source = project / "memory.json"
            target = project / "runtime" / "memory.json"
            source.write_text('{"legacy": true}', encoding="utf-8")

            migrated = migrate_legacy_runtime_files(project, (target,))
            source.write_text('{"legacy": false}', encoding="utf-8")
            migrated_again = migrate_legacy_runtime_files(project, (target,))

            self.assertEqual(migrated, [(source, target)])
            self.assertEqual(migrated_again, [])
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"legacy": True})


class ApiJournalPrivacyTests(unittest.TestCase):
    def test_default_journal_records_metadata_without_message_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api.jsonl"
            journal = ApiJournal(str(path), max_entries=10, enabled=True)

            journal.record(
                provider="test",
                model="model",
                messages=[{"role": "user", "content": "very private text"}],
                response="private answer",
                kind="reply",
                chat_id=42,
            )

            raw = path.read_text(encoding="utf-8")
            entry = json.loads(raw)
            self.assertNotIn("very private text", raw)
            self.assertNotIn("private answer", raw)
            self.assertEqual(entry["messages"]["roles"], ["user"])
            self.assertEqual(entry["response"]["character_count"], 14)

    def test_journal_keeps_exactly_the_latest_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api.jsonl"
            journal = ApiJournal(
                str(path),
                max_entries=2,
                enabled=True,
                include_content=True,
            )
            for number in range(3):
                journal.record(
                    provider="test",
                    model="model",
                    messages=[{"role": "user", "content": str(number)}],
                    response=f"answer-{number}",
                    kind="reply",
                    chat_id=42,
                )

            entries = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(entries), 2)
            self.assertEqual(
                [entry["response"] for entry in entries],
                ["answer-1", "answer-2"],
            )

    def test_existing_journal_is_trimmed_when_opened(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api.jsonl"
            path.write_text(
                "\n".join(json.dumps({"response": number}) for number in range(4))
                + "\n",
                encoding="utf-8",
            )

            ApiJournal(str(path), max_entries=2, enabled=True)

            entries = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(entries, [{"response": 2}, {"response": 3}])

    def test_existing_journal_removes_intermediate_and_embedded_media(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api.jsonl"
            payload = "data:image/jpeg;base64," + "A" * 1000
            entries = [
                {"kind": "reply", "response": "", "messages": []},
                {
                    "kind": "reply",
                    "response": "answer",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": payload}],
                        }
                    ],
                },
            ]
            path.write_text(
                "\n".join(json.dumps(item) for item in entries) + "\n",
                encoding="utf-8",
            )

            ApiJournal(str(path), max_entries=30, enabled=True, include_content=True)

            raw = path.read_text(encoding="utf-8")
            self.assertEqual(len(raw.splitlines()), 1)
            self.assertNotIn("A" * 100, raw)
            self.assertIn("data URL omitted", raw)


class GroupMemorySelectionTests(unittest.TestCase):
    def test_other_participant_memory_requires_name_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(str(Path(directory) / "memory.json"))
            store.set_chat_type(-1, "group")
            store.merge(-1, 1, "Алиса", ["общий факт"], ["любит чай"])
            store.merge(-1, 2, "Борис", [], ["любит джаз"])

            self.assertIn("общий факт", store.chat_context(-1))
            self.assertEqual(store.referenced_user_context(-1, 1, "как дела?"), "")
            self.assertIn(
                "любит джаз",
                store.referenced_user_context(-1, 1, "а что любит Борис?"),
            )


class RateLimiterTests(unittest.TestCase):
    def test_limit_is_per_chat_and_user_and_expires(self) -> None:
        limiter = RequestRateLimiter(request_count=2, window_seconds=10)

        self.assertEqual(limiter.allow((1, 7), 100), (True, 0))
        self.assertEqual(limiter.allow((1, 7), 101), (True, 0))
        allowed, retry_after = limiter.allow((1, 7), 102)
        self.assertFalse(allowed)
        self.assertGreater(retry_after, 0)
        self.assertEqual(limiter.allow((1, 8), 102), (True, 0))
        self.assertEqual(limiter.allow((1, 7), 111), (True, 0))


class LogPrivacyTests(unittest.TestCase):
    def test_proxy_credentials_are_removed_from_log_description(self) -> None:
        self.assertEqual(
            safe_url_for_log("socks5://secret:password@127.0.0.1:10808/path?token=x"),
            "socks5://127.0.0.1:10808",
        )

    def test_runtime_log_is_written_to_file_without_console(self) -> None:
        root_logger = logging.getLogger()
        old_handlers = list(root_logger.handlers)
        old_level = root_logger.level
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "runtime.log"
            settings = SimpleNamespace(
                log_level="INFO",
                log_file=str(log_path),
                log_max_bytes=100_000,
                log_backup_count=2,
            )
            try:
                with patch.dict(os.environ, {"BOT_LOG_CONSOLE": "0"}):
                    configure_logging(settings)
                logging.getLogger("test-runtime-log").info("health check")
                for handler in root_logger.handlers:
                    handler.flush()
                self.assertIn("health check", log_path.read_text(encoding="utf-8"))
                self.assertEqual(len(root_logger.handlers), 1)
            finally:
                for handler in root_logger.handlers:
                    handler.close()
                root_logger.handlers[:] = old_handlers
                root_logger.setLevel(old_level)


class ConversationSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_ai_turn_is_removed_from_context(self) -> None:
        service = object.__new__(ReplyService)
        messages = [
            {
                "role": "user",
                "content": "request",
                "user_id": 7,
                "user_name": "User",
            }
        ]
        service.settings = SimpleNamespace(
            ai_provider="vsegpt",
            vsegpt_model="model",
            timezone="Europe/Moscow",
        )
        service.gigachat_client = None
        service.vsegpt_client = object()
        service.memory = SimpleNamespace(
            set_chat_type=Mock(),
            remember_user=Mock(),
            user_name=Mock(return_value="User"),
            is_private_chat=Mock(return_value=True),
            user_timezone=Mock(return_value="Europe/Moscow"),
        )
        service._prepare_history = AsyncMock(return_value=messages)
        service._general_system_message = Mock(return_value="system")
        service._api_messages = Mock(return_value=messages)
        service._vsegpt_reply_with_tools = AsyncMock(side_effect=RuntimeError("offline"))
        service._save_history = Mock()

        answer = await service.reply(1, 7, "User", "request", "general", "private")

        self.assertIn("Попробуйте ещё раз", answer)
        self.assertEqual(messages, [])
        service._save_history.assert_called_once()

    def test_idle_group_keeps_only_last_complete_exchange(self) -> None:
        service = object.__new__(ReplyService)
        service.settings = SimpleNamespace(group_context_idle_seconds=60)
        old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        messages = [
            {"role": "user", "content": "old topic", "created_at": old},
            {"role": "assistant", "content": "old answer", "created_at": old},
            {"role": "user", "content": "last question", "created_at": old},
            {"role": "assistant", "content": "last answer", "created_at": old},
        ]

        service._reset_idle_group_history(messages, datetime.now(timezone.utc))

        self.assertEqual([item["content"] for item in messages], ["last question", "last answer"])

    def test_forgetting_user_does_not_remove_another_users_answer(self) -> None:
        service = object.__new__(ReplyService)
        service.memory = SimpleNamespace(forget_user=Mock())
        service.history = {
            (-100, "general"): [
                {"role": "user", "content": "delete me", "user_id": 7},
                {"role": "user", "content": "keep me", "user_id": 8},
                {"role": "assistant", "content": "answer to 8"},
            ]
        }
        service.compacted_history = set()
        service.journal = SimpleNamespace(purge_chat=Mock())
        service._save_history = Mock()

        service.forget_user_data(7, 7)

        self.assertEqual(
            [item["content"] for item in service.history[(-100, "general")]],
            ["keep me", "answer to 8"],
        )

    def test_three_messages_from_one_member_keep_other_group_members(self) -> None:
        service = object.__new__(ReplyService)
        service.settings = SimpleNamespace(
            group_context_limit=8,
            soft_context_limit=8,
            max_english_history_messages=4,
        )
        messages = [
            {"role": "user", "content": "from other", "user_id": 2, "user_name": "Борис"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "one", "user_id": 1, "user_name": "Алиса"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "two", "user_id": 1, "user_name": "Алиса"},
            {"role": "assistant", "content": "answer two"},
            {"role": "user", "content": "three", "user_id": 1, "user_name": "Алиса"},
        ]

        result = service._api_messages(messages, 1, "general", False)

        self.assertTrue(any("Борис" in item["content"] for item in result))


if __name__ == "__main__":
    unittest.main()
