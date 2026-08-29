import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

from telegram import Chat, Message, MessageEntity, User

from bot import (
    ApiJournal,
    ImageInput,
    MediaProcessor,
    PreparedMedia,
    REMINDER_MAX_DELIVERY_ATTEMPTS,
    ReminderStore,
    ReplyService,
    _process_incoming_messages,
    api_log_command,
    atomic_write_text,
    mentioned_bot,
    parse_reminder,
)
from media_processing import extract_supported_video_urls, is_supported_video_url
from tools_registry import ToolContext


class ReminderParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 8, 25, 12, 0, tzinfo=ZoneInfo("Europe/Moscow"))

    def test_plain_tool_request_uses_local_fallback(self) -> None:
        parsed = parse_reminder("напомни через 5 минут позвонить", self.now)

        self.assertIsNotNone(parsed)
        due_at, text = parsed
        self.assertEqual(due_at, self.now + timedelta(minutes=5))
        self.assertEqual(text, "позвонить")

    def test_command_request_may_include_remind_verb(self) -> None:
        parsed = parse_reminder("/н напомни завтра 13:00 купить продукты", self.now)

        self.assertIsNotNone(parsed)
        due_at, text = parsed
        self.assertEqual(due_at, datetime(2026, 8, 26, 13, 0, tzinfo=self.now.tzinfo))
        self.assertEqual(text, "купить продукты")

    def test_explicit_past_time_is_rejected(self) -> None:
        self.assertIsNone(parse_reminder("/н сегодня 10:00 тест", self.now))
        self.assertIsNone(
            ReplyService._scheduled_from_text(
                "TIME: 2026-08-25 10:00\nTEXT: тест",
                self.now,
                "тест",
            )
        )


class PersistenceTests(unittest.TestCase):
    def test_atomic_write_preserves_old_file_when_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text("old", encoding="utf-8")

            with patch.object(Path, "replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    atomic_write_text(path, "new")

            self.assertEqual(path.read_text(encoding="utf-8"), "old")
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_failed_delivery_backs_off_and_eventually_stops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ReminderStore(str(Path(directory) / "reminders.json"))
            now = datetime(2026, 8, 25, 12, 0, tzinfo=ZoneInfo("Europe/Moscow"))
            reminder = store.add(1, 2, "User", now - timedelta(minutes=1), "test")
            reminder_id = str(reminder["id"])

            self.assertEqual(len(store.claim_due(now)), 1)
            self.assertEqual(store.fail_delivery(reminder_id, now, RuntimeError("offline")), "active")
            self.assertEqual(store.claim_due(now), [])
            self.assertEqual(len(store.claim_due(now + timedelta(seconds=61))), 1)

            status = "active"
            for attempt in range(1, REMINDER_MAX_DELIVERY_ATTEMPTS):
                status = store.fail_delivery(
                    reminder_id,
                    now + timedelta(hours=attempt),
                    RuntimeError("offline"),
                )
            self.assertEqual(status, "failed")
            self.assertEqual(store.claim_due(now + timedelta(days=1)), [])


class ToolFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_reminder_is_confirmed_when_final_model_call_fails(self) -> None:
        service = object.__new__(ReplyService)
        service.settings = SimpleNamespace(vsegpt_model="test", vsegpt_max_tokens=200)
        call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="create_reminder",
                arguments=json.dumps({"request": "через час тест", "status_text": "Создаю"}),
            ),
        )
        first_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call], content=None))]
        )
        service._vsegpt_chat = AsyncMock(
            side_effect=[first_response, RuntimeError("final response unavailable")]
        )
        service.tools = SimpleNamespace(
            schemas=[],
            schemas_for=lambda _names: [],
            execute=AsyncMock(
                return_value=json.dumps(
                    {
                        "ok": True,
                        "reminder_id": "reminder-1",
                        "text": "тест",
                        "scheduled_for_display": "25.08.2026 13:00",
                    },
                    ensure_ascii=False,
                )
            ),
        )
        service._record_vsegpt_final_reply = AsyncMock()

        with patch("bot.LOGGER.exception"):
            answer = await service._vsegpt_reply_with_tools(
                [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "request"},
                ],
                ToolContext(
                    chat_id=1,
                    user_id=2,
                    user_name="User",
                    timezone="Europe/Moscow",
                ),
            )

        self.assertIn("Напоминание создано", answer)
        self.assertIn("25.08.2026 13:00", answer)
        service.tools.execute.assert_awaited_once()
        self.assertEqual(service._vsegpt_chat.await_count, 2)
        self.assertTrue(
            all(
                call.kwargs["record_journal"] is False
                for call in service._vsegpt_chat.await_args_list
            )
        )
        service._record_vsegpt_final_reply.assert_awaited_once()


class ToolSelectionTests(unittest.TestCase):
    def test_stable_or_subjective_questions_do_not_enable_search(self) -> None:
        self.assertEqual(
            ReplyService._tool_names_for_text("Лунный рыцарь это Marvel или DC?"),
            (),
        )
        self.assertEqual(
            ReplyService._tool_names_for_text("Что лучше: Star Wars или Marvel?"),
            (),
        )

    def test_current_rating_and_reminder_enable_only_needed_tools(self) -> None:
        self.assertEqual(
            ReplyService._tool_names_for_text("Какая оценка сериала сейчас?"),
            ("web_search",),
        )
        self.assertEqual(
            ReplyService._tool_names_for_text("Напомни завтра позвонить"),
            ("create_reminder",),
        )


class JournalAccessTests(unittest.IsolatedAsyncioTestCase):
    async def test_group_chat_cannot_read_api_journal(self) -> None:
        replies: list[str] = []

        async def reply_text(text: str) -> None:
            replies.append(text)

        update = SimpleNamespace(
            effective_message=SimpleNamespace(reply_text=reply_text),
            effective_chat=SimpleNamespace(id=-1, type="group"),
        )

        await api_log_command(update, SimpleNamespace())

        self.assertEqual(replies, ["Команда /log доступна только в личном чате."])


class MediaSecurityTests(unittest.TestCase):
    def test_only_expected_youtube_hosts_are_accepted(self) -> None:
        self.assertTrue(is_supported_video_url("https://youtube.com/shorts/abc"))
        self.assertTrue(is_supported_video_url("https://youtu.be/abc?t=1"))
        self.assertTrue(is_supported_video_url("https://youtube.com/watch?v=abc"))
        self.assertFalse(is_supported_video_url("http://127.0.0.1/video"))
        self.assertFalse(is_supported_video_url("https://youtube.com.evil.test/video"))
        self.assertFalse(is_supported_video_url("https://user@youtube.com/video"))
        self.assertFalse(is_supported_video_url("https://youtube.com/playlist?list=abc"))
        self.assertFalse(is_supported_video_url("https://youtube.com/@channel"))
        self.assertFalse(is_supported_video_url("file:///etc/passwd"))

    def test_supported_links_are_bounded_and_deduplicated(self) -> None:
        text = (
            "https://youtu.be/first https://example.com/no "
            "https://youtu.be/first https://youtube.com/shorts/second"
        )
        self.assertEqual(extract_supported_video_urls(text), ["https://youtu.be/first"])
        self.assertEqual(
            extract_supported_video_urls(text, limit=2),
            ["https://youtu.be/first", "https://youtube.com/shorts/second"],
        )

    def test_api_journal_omits_base64_payloads(self) -> None:
        payload = "data:image/jpeg;base64," + "A" * 1000
        safe = ApiJournal._safe(
            {"content": [{"type": "image_url", "image_url": payload}]}
        )

        serialized = json.dumps(safe)
        self.assertNotIn("A" * 100, serialized)
        self.assertIn("data URL omitted", serialized)

    def test_images_are_attached_only_to_latest_user_turn(self) -> None:
        request = ReplyService._attach_images_to_latest_user_message(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "current"},
            ],
            [ImageInput(b"jpeg", "image/jpeg", "photo")],
        )

        self.assertEqual(request[1]["content"], "old")
        self.assertEqual(request[3]["content"][0], {"type": "text", "text": "current"})
        self.assertTrue(request[3]["content"][1]["image_url"].startswith("data:image/jpeg"))


class MentionParsingTests(unittest.TestCase):
    def test_caption_mention_after_emoji_uses_utf16_offsets(self) -> None:
        user = User(id=1, first_name="User", is_bot=False)
        message = Message(
            message_id=1,
            date=datetime.now(timezone.utc),
            chat=Chat(id=-1, type="group"),
            from_user=user,
            caption="😀 @Peter посмотри",
            caption_entities=[MessageEntity(type="mention", offset=3, length=6)],
        )

        self.assertTrue(mentioned_bot(message, "Peter", 99))


class MultimodalReplyTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _message(**overrides):
        defaults = {
            "message_id": 1,
            "chat": SimpleNamespace(id=-100, type="group"),
            "from_user": SimpleNamespace(
                id=7,
                is_bot=False,
                full_name="Иван",
                username="ivan",
            ),
            "sender_chat": None,
            "text": None,
            "caption": None,
            "entities": (),
            "caption_entities": (),
            "photo": (),
            "voice": None,
            "audio": None,
            "video": None,
            "video_note": None,
            "animation": None,
            "document": None,
            "reply_to_message": None,
            "media_group_id": None,
            "reply_text": AsyncMock(),
            "parse_entity": lambda entity: "",
            "parse_caption_entity": lambda entity: "",
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    async def _run(self, message, prepare_side_effect=None):
        settings = SimpleNamespace(
            media_max_transcript_chars=12000,
        )
        service = SimpleNamespace(settings=settings, reply=AsyncMock(return_value="готово"))

        async def prepare_message(*args, **kwargs):
            prepared = kwargs["prepared"]
            if prepare_side_effect:
                prepare_side_effect(prepared)
            return True

        processor = SimpleNamespace(
            prepare_telegram_message=AsyncMock(side_effect=prepare_message),
            prepare_video_url=AsyncMock(),
        )
        bot = SimpleNamespace(id=99, username="Peter", send_chat_action=AsyncMock())
        application = SimpleNamespace(
            bot_data={"reply_service": service, "media_processor": processor}
        )
        context = SimpleNamespace(application=application, bot=bot)

        async def passthrough(_bot, _chat_id, operation):
            return await operation

        with patch("bot.run_with_typing", side_effect=passthrough):
            await _process_incoming_messages([message], context)
        return service, processor

    async def test_group_photo_uses_question_trigger_without_analysis_command(self) -> None:
        photo = SimpleNamespace(file_id="photo", file_size=100)
        message = self._message(photo=[photo], caption="?? что на фото?")

        def add_image(prepared):
            prepared.images.append(ImageInput(b"jpeg", "image/jpeg", "фото"))

        service, processor = await self._run(message, add_image)

        processor.prepare_telegram_message.assert_awaited_once()
        service.reply.assert_awaited_once()
        call = service.reply.await_args
        self.assertEqual(call.args[4], "general")
        self.assertEqual(call.kwargs["images"][0].label, "фото")
        self.assertIn("?? что на фото?", call.args[3])

    async def test_plain_group_photo_is_ignored(self) -> None:
        photo = SimpleNamespace(file_id="photo", file_size=100)
        message = self._message(photo=[photo])

        service, processor = await self._run(message)

        processor.prepare_telegram_message.assert_not_awaited()
        service.reply.assert_not_awaited()

    async def test_english_caption_does_not_trigger_group_media(self) -> None:
        photo = SimpleNamespace(file_id="photo", file_size=100)
        message = self._message(photo=[photo], caption="hello")

        service, processor = await self._run(message)

        processor.prepare_telegram_message.assert_not_awaited()
        service.reply.assert_not_awaited()

    async def test_analysis_command_does_not_replace_normal_group_triggers(self) -> None:
        photo = SimpleNamespace(file_id="photo", file_size=100)
        message = self._message(photo=[photo], caption="/analyze")

        service, processor = await self._run(message)

        processor.prepare_telegram_message.assert_not_awaited()
        service.reply.assert_not_awaited()

    async def test_group_photo_uses_bot_mention_without_analysis_command(self) -> None:
        photo = SimpleNamespace(file_id="photo", file_size=100)
        message = self._message(photo=[photo], caption="@Peter что на фото?")

        def add_image(prepared):
            prepared.images.append(ImageInput(b"jpeg", "image/jpeg", "фото"))

        service, processor = await self._run(message, add_image)

        processor.prepare_telegram_message.assert_awaited_once()
        service.reply.assert_awaited_once()
        call = service.reply.await_args
        self.assertEqual(call.args[4], "general")
        self.assertNotIn("@Peter", call.args[3])
        self.assertEqual(call.kwargs["images"][0].label, "фото")

    async def test_plain_private_photo_is_processed(self) -> None:
        photo = SimpleNamespace(file_id="photo", file_size=100)
        private_chat = SimpleNamespace(id=100, type="private")
        message = self._message(photo=[photo], chat=private_chat)

        def add_image(prepared):
            prepared.images.append(ImageInput(b"jpeg", "image/jpeg", "фото"))

        service, processor = await self._run(message, add_image)

        processor.prepare_telegram_message.assert_awaited_once()
        service.reply.assert_awaited_once()
        self.assertEqual(service.reply.await_args.args[4], "general")

    async def test_mention_in_reply_includes_original_message(self) -> None:
        replied = self._message(
            message_id=5,
            text="Это исходное сообщение",
            from_user=SimpleNamespace(
                id=8,
                is_bot=False,
                full_name="Вася",
                username="vasya",
            ),
        )
        message = self._message(
            message_id=6,
            text="@Peter что думаешь?",
            reply_to_message=replied,
        )

        service, _ = await self._run(message)

        prompt = service.reply.await_args.args[3]
        self.assertIn("Это исходное сообщение", prompt)
        self.assertIn("Вася", prompt)
        self.assertNotIn("@Peter", prompt)
        self.assertFalse(service.reply.await_args.kwargs["allow_tools"])

    async def test_media_content_cannot_trigger_stateful_tools(self) -> None:
        settings = SimpleNamespace(
            ai_provider="gigachat",
            media_max_files_per_turn=6,
            media_max_turn_bytes=25_000_000,
            media_max_frames=6,
        )
        processor = MediaProcessor(SimpleNamespace(settings=settings))
        message = self._message(photo=[SimpleNamespace(file_id="photo", file_size=100)])
        prepared = PreparedMedia()

        await processor.prepare_telegram_message(
            message,
            SimpleNamespace(),
            1,
            source_label="photo",
            prepared=prepared,
        )

        self.assertFalse(prepared.allow_tools)


class VideoPreparationTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_subtitles_skip_paid_transcription(self) -> None:
        settings = SimpleNamespace(
            media_max_frames=6,
            media_processing_timeout=30,
            media_max_transcript_chars=12000,
        )
        service = SimpleNamespace(settings=settings, transcribe_audio=AsyncMock())
        processor = MediaProcessor(service)
        prepared = PreparedMedia()

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "video.mp4"
            source.write_bytes(b"test")
            with patch("bot.extract_video_frames", new=AsyncMock(return_value=[])):
                await processor._prepare_video_file(
                    source,
                    directory,
                    duration=10,
                    source_label="YouTube",
                    chat_id=1,
                    prepared=prepared,
                    known_transcript="готовые субтитры",
                )

        service.transcribe_audio.assert_not_awaited()
        self.assertIn("готовые субтитры", prepared.notes[0])


if __name__ == "__main__":
    unittest.main()
