import asyncio
import threading
import unittest
import httpx
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from telegram.error import NetworkError, RetryAfter

from bot import ReplyService, telegram_keepalive_loop, stop_background_tasks, reply_text_with_retry
from tools_registry import ToolContext, web_search


class ResponseLatencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_uncertain_send_is_not_repeated(self):
        for cause in (httpx.ReadTimeout("lost response"), httpx.WriteError("partial write"),
                      httpx.RemoteProtocolError("disconnected"), None):
            with self.subTest(cause=type(cause).__name__):
                error = NetworkError("unconfirmed")
                error.__cause__ = cause
                message = SimpleNamespace(reply_text=AsyncMock(side_effect=error))
                with self.assertRaises(NetworkError):
                    await reply_text_with_retry(message, "answer")
                message.reply_text.assert_awaited_once()

    async def test_connect_failure_and_rate_limit_can_be_retried(self):
        connection_error = NetworkError("not connected")
        connection_error.__cause__ = httpx.ConnectTimeout("connect timed out")
        for error in (connection_error, RetryAfter(1)):
            message = SimpleNamespace(reply_text=AsyncMock(side_effect=[error, "sent"]))
            with patch("bot.asyncio.sleep", new=AsyncMock()):
                self.assertEqual(await reply_text_with_retry(message, "answer"), "sent")
            self.assertEqual(message.reply_text.await_count, 2)

    async def test_keepalive_reconnects_after_network_failure_without_sending(self):
        bot = SimpleNamespace(
            get_me=AsyncMock(side_effect=[NetworkError("offline"), object()]),
            send_message=AsyncMock(),
        )
        application = SimpleNamespace(bot=bot)
        with patch("bot.asyncio.sleep", new=AsyncMock(side_effect=[None, asyncio.CancelledError])) as sleep:
            with self.assertRaises(asyncio.CancelledError):
                await telegram_keepalive_loop(application)
        self.assertEqual(bot.get_me.await_count, 2)
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [5.0, 30.0])
        bot.send_message.assert_not_awaited()

    async def test_shutdown_cancels_connection_keepalive(self):
        task = asyncio.create_task(asyncio.Event().wait())
        application = SimpleNamespace(bot_data={"telegram_keepalive_task": task})
        await stop_background_tasks(application)
        self.assertTrue(task.cancelled())

    async def test_stalled_progress_message_does_not_block_tool_or_answer(self):
        service = object.__new__(ReplyService)
        service.settings = SimpleNamespace(vsegpt_model="test", vsegpt_max_tokens=200)
        call = SimpleNamespace(
            id="search-1",
            function=SimpleNamespace(
                name="web_search",
                arguments='{"query":"test", "status_text":"Ищу"}',
            ),
        )
        service._vsegpt_chat = AsyncMock(side_effect=[
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                tool_calls=[call], content=None,
            ))]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                tool_calls=[], content="Готовый ответ",
            ))]),
        ])
        service.tools = SimpleNamespace(
            schemas_for=lambda names: [],
            execute=AsyncMock(return_value='{"ok":true,"results":[]}'),
        )
        service._record_vsegpt_final_reply = AsyncMock()
        cancelled = asyncio.Event()

        async def stalled_status(_text):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with patch("bot.TOOL_STATUS_TIMEOUT_SECONDS", 0.01), patch("bot.LOGGER.warning"):
            answer = await asyncio.wait_for(service._vsegpt_reply_with_tools(
                [{"role":"system", "content":"system"}, {"role":"user", "content":"test"}],
                ToolContext(1, 2, "User", "Europe/Moscow"),
                ("web_search",),
                status_callback=stalled_status,
            ), timeout=1)
        self.assertEqual(answer, "Готовый ответ")
        self.assertTrue(cancelled.is_set())
        service.tools.execute.assert_awaited_once()
        service._record_vsegpt_final_reply.assert_awaited_once()

    async def test_search_falls_back_after_first_backend_failure(self):
        client = MagicMock()
        client.text.side_effect = [TimeoutError("offline"), [
            {"title":"Found", "href":"https://example.com", "body":"Fact"},
        ]]
        with patch("tools_registry.DDGS") as factory:
            factory.return_value.__enter__.return_value = client
            result = await web_search({"query":"test"}, ToolContext(1, 2, "User", "UTC"))
        self.assertEqual(result["backend"], "brave")
        self.assertEqual([call.kwargs["backend"] for call in client.text.call_args_list],
                         ["duckduckgo", "brave"])
        self.assertTrue(result["ok"])

    async def test_search_deadline_returns_even_if_backend_is_stalled(self):
        release = threading.Event()
        client = MagicMock()

        def stalled_backend(*args, **kwargs):
            release.wait(timeout=1)
            return []

        client.text.side_effect = stalled_backend
        try:
            with patch("tools_registry.DDGS") as factory, patch(
                "tools_registry.SEARCH_TOTAL_TIMEOUT_SECONDS", 0.02
            ):
                factory.return_value.__enter__.return_value = client
                with self.assertRaises(TimeoutError):
                    await web_search({"query":"test"}, ToolContext(1, 2, "User", "UTC"))
                self.assertFalse(release.is_set())
        finally:
            release.set()
