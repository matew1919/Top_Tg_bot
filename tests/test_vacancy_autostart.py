from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from bot import start_background_tasks, stop_background_tasks


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


if __name__ == "__main__":
    unittest.main()
