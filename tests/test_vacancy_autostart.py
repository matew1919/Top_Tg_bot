from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot import start_background_tasks, stop_background_tasks, vacancies_command


class VacancyAutostartTests(unittest.IsolatedAsyncioTestCase):
    async def test_bot_starts_and_stops_vacancy_task(self) -> None:
        application = SimpleNamespace(
            bot_data={
                "vacancy_service": SimpleNamespace(
                    settings=SimpleNamespace(vacancy_autostart=True)
                )
            }
        )
        reminder_started = asyncio.Event()
        vacancy_started = asyncio.Event()

        async def fake_reminder_loop(_application: object) -> None:
            reminder_started.set()
            await asyncio.Event().wait()

        async def fake_vacancy_loop(_application: object) -> None:
            vacancy_started.set()
            await asyncio.Event().wait()

        with (
            patch("bot.reminder_loop", fake_reminder_loop),
            patch("bot.vacancy_loop", fake_vacancy_loop),
        ):
            await start_background_tasks(application)
            await asyncio.wait_for(reminder_started.wait(), timeout=1)
            await asyncio.wait_for(vacancy_started.wait(), timeout=1)
            self.assertEqual(application.bot_data["vacancy_task"].get_name(), "vacancy-loop")
            await stop_background_tasks(application)

        self.assertTrue(application.bot_data["reminder_task"].done())
        self.assertTrue(application.bot_data["vacancy_task"].done())

    async def test_autostart_can_be_disabled(self) -> None:
        application = SimpleNamespace(
            bot_data={
                "vacancy_service": SimpleNamespace(
                    settings=SimpleNamespace(vacancy_autostart=False)
                )
            }
        )

        async def idle_loop(_application: object) -> None:
            await asyncio.Event().wait()

        with patch("bot.reminder_loop", idle_loop):
            await start_background_tasks(application)
            self.assertNotIn("vacancy_task", application.bot_data)
            await stop_background_tasks(application)


    async def test_disabled_parser_overrides_autostart(self) -> None:
        application = SimpleNamespace(bot_data={"vacancy_service": SimpleNamespace(
            settings=SimpleNamespace(vacancy_autostart=True, vacancy_enabled=False)
        )})
        async def idle_loop(_application):
            await asyncio.Event().wait()
        with patch("bot.reminder_loop", idle_loop):
            await start_background_tasks(application)
            self.assertNotIn("vacancy_task", application.bot_data)
            await stop_background_tasks(application)

    async def test_disabled_parser_blocks_manual_poll(self) -> None:
        service = SimpleNamespace(settings=SimpleNamespace(vacancy_enabled=False), poll=AsyncMock())
        message = SimpleNamespace(reply_text=AsyncMock())
        update = SimpleNamespace(effective_message=message, effective_chat=SimpleNamespace(id=1))
        context = SimpleNamespace(application=SimpleNamespace(bot_data={"vacancy_service":service}), args=["check"])
        await vacancies_command(update, context)
        service.poll.assert_not_awaited()
        message.reply_text.assert_awaited_once_with("Парсер вакансий отключён в настройках бота.")


if __name__ == "__main__":
    unittest.main()
