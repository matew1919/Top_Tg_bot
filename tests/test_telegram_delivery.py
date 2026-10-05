from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from telegram.error import BadRequest, TimedOut

from bot import (
    TELEGRAM_REQUEST_POOL_SIZE,
    reply_text_with_retry,
    run_with_typing,
    telegram_request_with_retry,
)


class TelegramRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_network_failure_is_retried_until_success(self) -> None:
        request = AsyncMock(
            side_effect=[TimedOut("pool busy"), TimedOut("proxy slow"), "sent"]
        )

        with patch("bot.asyncio.sleep", new=AsyncMock()) as sleep:
            result = await telegram_request_with_retry(request, "test")

        self.assertEqual(result, "sent")
        self.assertEqual(request.await_count, 3)
        self.assertEqual(sleep.await_count, 2)

    async def test_non_network_telegram_error_is_not_retried(self) -> None:
        request = AsyncMock(side_effect=BadRequest("bad message"))

        with self.assertRaises(BadRequest):
            await telegram_request_with_retry(request, "test")

        request.assert_awaited_once()

    async def test_reply_wrapper_creates_a_fresh_request_for_retry(self) -> None:
        message = AsyncMock()
        message.reply_text.side_effect = [TimedOut("pool busy"), "message"]

        with patch("bot.asyncio.sleep", new=AsyncMock()):
            result = await reply_text_with_retry(message, "hello")

        self.assertEqual(result, "message")
        self.assertEqual(message.reply_text.await_count, 2)
        message.reply_text.assert_awaited_with("hello")

    def test_outbound_pool_has_room_for_parallel_requests(self) -> None:
        self.assertGreaterEqual(TELEGRAM_REQUEST_POOL_SIZE, 8)


class TypingIndicatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_operation_finishes_before_typing_request(self) -> None:
        bot = AsyncMock()

        result = await run_with_typing(bot, 123, asyncio.sleep(0, result="done"))

        self.assertEqual(result, "done")
        bot.send_chat_action.assert_not_awaited()

    async def test_stuck_typing_request_does_not_block_operation(self) -> None:
        bot = AsyncMock()

        async def stuck_action(**kwargs):
            del kwargs
            await asyncio.Event().wait()

        bot.send_chat_action.side_effect = stuck_action
        with (
            patch("bot.TELEGRAM_TYPING_DELAY_SECONDS", 0),
            patch("bot.TELEGRAM_TYPING_REQUEST_TIMEOUT_SECONDS", 0.01),
            patch("bot.TELEGRAM_TYPING_INTERVAL_SECONDS", 0.01),
        ):
            result = await asyncio.wait_for(
                run_with_typing(bot, 123, asyncio.sleep(0.03, result="done")),
                timeout=0.2,
            )

        self.assertEqual(result, "done")
        self.assertGreaterEqual(bot.send_chat_action.await_count, 1)


if __name__ == "__main__":
    unittest.main()
