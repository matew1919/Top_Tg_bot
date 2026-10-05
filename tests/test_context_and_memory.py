from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from bot import (
    MediaAttachment,
    MemoryStore,
    ReplyService,
    clean_assistant_call,
    direct_media_history_text,
    message_mentions_bot,
    with_quoted_message,
)


class _HistoryStore:
    def save(self, history, compacted_history) -> None:
        self.last_history = history
        self.last_compacted_history = compacted_history


class _Memory:
    def set_chat_type(self, chat_id, chat_type) -> None:
        pass

    def remember_user(self, chat_id, user_id, user_name, chat_type) -> None:
        pass

    def user_name(self, chat_id, user_id):
        return None

    def is_private_chat(self, chat_id) -> bool:
        return chat_id > 0

    def personal_context(self, chat_id, user_id) -> str:
        return ""

    def context(self, chat_id, exclude_user_id=None) -> str:
        return ""


def _service() -> ReplyService:
    service = ReplyService.__new__(ReplyService)
    service.settings = SimpleNamespace(
        group_context_limit=5,
        group_context_idle_seconds=3600,
        max_english_history_messages=4,
        soft_context_limit=20,
        hard_context_limit=20,
        keep_last_messages=2,
    )
    service.history = {}
    service.compacted_history = set()
    service.history_store = _HistoryStore()
    service._extract_memory = AsyncMock(return_value=True)
    return service


def _message(role: str, content: str, created_at: datetime | None = None):
    item = {"role": role, "content": content}
    if role == "user":
        item.update({"user_id": 1, "user_name": "User"})
    if created_at is not None:
        item["created_at"] = created_at.isoformat(timespec="seconds")
    return item


class GroupContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_idle_group_keeps_last_exchange_then_new_message(self) -> None:
        service = _service()
        now = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)
        service._history_now = lambda: now
        key = (-100, "general")
        service.history[key] = [
            _message("user", "old question", now - timedelta(hours=2)),
            _message("assistant", "old answer", now - timedelta(hours=2)),
        ]

        result = await service._prepare_history(
            -100, 1, "User", "new question", "general", False
        )

        self.assertEqual(
            [item["content"] for item in result],
            ["old question", "old answer", "new question"],
        )
        service._extract_memory.assert_not_awaited()

    async def test_legacy_group_timestamp_is_treated_as_idle(self) -> None:
        service = _service()
        now = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)
        service._history_now = lambda: now
        key = (-100, "general")
        service.history[key] = [
            _message("user", "discard me"),
            _message("assistant", "discard answer"),
            _message("user", "keep me"),
            _message("assistant", "keep answer"),
        ]

        result = await service._prepare_history(
            -100, 1, "User", "new", "general", False
        )

        self.assertEqual(
            [item["content"] for item in result],
            ["keep me", "keep answer", "new"],
        )

    async def test_active_group_is_physically_limited_without_compaction(self) -> None:
        service = _service()
        now = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)
        service._history_now = lambda: now
        key = (-100, "general")
        service.history[key] = [
            _message("user", f"u{index}", now - timedelta(minutes=10))
            if index % 2 == 0
            else _message("assistant", f"a{index}", now - timedelta(minutes=10))
            for index in range(8)
        ]

        result = await service._prepare_history(
            -100, 1, "User", "new", "general", False
        )

        self.assertLessEqual(len(result), 5)
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[-1]["content"], "new")
        service._extract_memory.assert_not_awaited()


class PrivateContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_chat_keeps_existing_hard_compaction(self) -> None:
        service = _service()
        now = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)
        service._history_now = lambda: now
        key = (100, "general")
        service.history[key] = [
            _message("user", f"u{index}", now)
            if index % 2 == 0
            else _message("assistant", f"a{index}", now)
            for index in range(20)
        ]

        result = await service._prepare_history(
            100, 1, "User", "new", "general", True
        )

        service._extract_memory.assert_awaited_once()
        self.assertEqual(
            [item["role"] for item in result],
            ["user", "assistant", "user"],
        )
        self.assertEqual(result[-1]["content"], "new")


class MediaContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_is_sent_once_but_only_marker_is_persisted(self) -> None:
        service = _service()
        service.settings.ai_provider = "vsegpt"
        service.settings.vsegpt_model = "text-model"
        service.settings.vsegpt_vision_model = "vision-model"
        service.settings.vsegpt_max_tokens = 200
        service.settings.vision_prompt = "Look at the image."
        service.settings.system_prompt = "System"
        service.settings.english_coach_prompt = "Coach"
        service.settings.timezone = "Europe/Moscow"
        service.vsegpt_client = object()
        service.gigachat_client = None
        service.memory = _Memory()
        service._vsegpt_reply_with_tools = AsyncMock(return_value="image answer")
        media = MediaAttachment("photo", "image/jpeg", b"image bytes")

        answer = await service.reply(
            -100,
            1,
            "User",
            "[отправлено фото]\nПодпись: что тут?",
            "general",
            "group",
            media=media,
        )

        self.assertEqual(answer, "image answer")
        request = service._vsegpt_reply_with_tools.await_args.args[0]
        current_content = request[-1]["content"]
        self.assertIsInstance(current_content, list)
        self.assertEqual(current_content[0]["type"], "text")
        self.assertTrue(current_content[1]["image_url"].startswith("data:image/jpeg;base64,"))
        persisted = service.history[(-100, "general")]
        self.assertEqual(persisted[0]["content"], "[отправлено фото]\nПодпись: что тут?")
        self.assertNotIn("base64", str(persisted))
        self.assertEqual(persisted[1]["content"], "image answer")

    def test_caption_mention_and_reply_block_are_preserved_without_triggers(self) -> None:
        sender = SimpleNamespace(id=2, full_name="Other User", username=None)
        photo = SimpleNamespace(file_id="photo-id")
        source = SimpleNamespace(
            text=None,
            caption="исходная подпись",
            entities=None,
            caption_entities=[],
            photo=[photo],
            document=None,
            video=None,
            animation=None,
            voice=None,
            audio=None,
            sticker=None,
            from_user=sender,
        )
        direct = SimpleNamespace(**{**vars(source), "caption": "?? @PeterBot что тут?"})

        self.assertTrue(message_mentions_bot(direct, "PeterBot", 99))
        self.assertEqual(clean_assistant_call(direct.caption, "PeterBot"), "что тут?")
        self.assertEqual(
            direct_media_history_text(direct, "PeterBot"),
            "[отправлено фото]\nПодпись: что тут?",
        )
        self.assertEqual(
            with_quoted_message("объясни", source),
            "Сообщение от Other User, на которое отвечает пользователь:\n"
            "[отправлено фото]\nПодпись: исходная подпись\n\n"
            "Текущий вопрос пользователя:\nобъясни",
        )


class MemoryRewriteTests(unittest.TestCase):
    def test_mutable_memory_is_replaced_and_immutable_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(str(Path(directory) / "memory.json"))
            store.data = {
                "version": 1,
                "chat_types": {"-100": "group"},
                "chats": {
                    "-100": {
                        "immutable": {"facts": ["protected"]},
                        "mutable": {"facts": ["old chat fact"]},
                    }
                },
                "users": {
                    "-100": {
                        "1": {
                            "name": "User",
                            "immutable": {"facts": ["protected user fact"]},
                            "mutable": {"facts": ["old user fact"]},
                        },
                        "2": {
                            "name": "Other",
                            "immutable": {"facts": []},
                            "mutable": {"facts": ["must be removed when omitted"]},
                        },
                    }
                },
            }

            store.rewrite_from_extracted(
                -100,
                {
                    "chat_facts": ["new chat fact"],
                    "users": {"1": {"name": "User", "facts": ["new user fact"]}},
                },
            )

            chat = store.data["chats"]["-100"]
            users = store.data["users"]["-100"]
            self.assertEqual(chat["immutable"]["facts"], ["protected"])
            self.assertEqual(chat["mutable"]["facts"], ["new chat fact"])
            self.assertEqual(users["1"]["immutable"]["facts"], ["protected user fact"])
            self.assertEqual(users["1"]["mutable"]["facts"], ["new user fact"])
            self.assertEqual(users["2"]["mutable"]["facts"], [])


class MemoryToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_tool_starts_full_rewrite_with_owner_information(self) -> None:
        service = _service()
        context = SimpleNamespace(
            chat_id=-100,
            user_id=42,
            user_name="Matvei",
            timezone="Europe/Moscow",
        )

        result = await service._update_memory_tool(
            {"scope": "current_user", "new_information": "Любит короткие ответы."},
            context,
        )

        self.assertTrue(result["ok"])
        service._extract_memory.assert_awaited_once()
        call = service._extract_memory.await_args
        self.assertEqual(call.args[:2], (-100, []))
        source = call.kwargs["new_information"]
        self.assertIn("ID 42", source)
        self.assertIn("Matvei", source)
        self.assertIn("Любит короткие ответы", source)


if __name__ == "__main__":
    unittest.main()
