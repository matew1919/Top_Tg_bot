import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from telegram.error import NetworkError, BadRequest

from search_status import SearchStatus, StatusMessages


class SearchStatusTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "statuses.json"
        self.bot = SimpleNamespace(delete_message=AsyncMock())
        self.store = StatusMessages(self.bot, self.path)
        self.sent = SimpleNamespace(chat=SimpleNamespace(id=-100), message_id=123)
        self.message = SimpleNamespace(reply_text=AsyncMock(return_value=self.sent))

    async def test_one_status_is_kept_during_search_then_deleted(self):
        status = SearchStatus(self.store, self.message)
        await status.show("Ща гляну")
        await status.show("Ещё ищу")
        self.message.reply_text.assert_awaited_once()
        await self.store.delete((-100, 123))
        self.bot.delete_message.assert_not_awaited()
        await status.finish()
        self.bot.delete_message.assert_awaited_once_with(chat_id=-100, message_id=123)
        self.assertFalse(self.store.pending)

    async def test_delete_failure_survives_restart_and_retries(self):
        status = SearchStatus(self.store, self.message)
        await status.show("Ща гляну")
        self.bot.delete_message.side_effect = NetworkError("offline")
        await status.finish()
        self.assertIn((-100, 123), self.store.pending)
        restored = StatusMessages(self.bot, self.path)
        self.bot.delete_message.side_effect = None
        await restored.delete((-100, 123))
        self.assertFalse(restored.pending)
        self.assertFalse(StatusMessages(self.bot, self.path).pending)

    async def test_late_send_acknowledgement_is_cleaned_after_turn_finishes(self):
        release = asyncio.Event()

        async def slow_send(text):
            await release.wait()
            return self.sent

        self.message.reply_text.side_effect = slow_send
        status = SearchStatus(self.store, self.message)
        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(status.show("Ща гляну"), timeout=0.01)
        await status.finish()
        release.set()
        await self.store.close()
        self.bot.delete_message.assert_awaited_once_with(chat_id=-100, message_id=123)
        self.assertFalse(self.store.pending)

    async def test_already_deleted_status_does_not_stay_in_queue(self):
        status = SearchStatus(self.store, self.message)
        await status.show("Ща гляну")
        self.bot.delete_message.side_effect = BadRequest("Message to delete not found")
        await status.finish()
        self.assertFalse(self.store.pending)
