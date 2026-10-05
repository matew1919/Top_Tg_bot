from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

from vacancy_monitor import VacancyMonitor, VacancyMonitorSettings
from vacancy_parser import Vacancy, VacancyConfig, VacancyFilter


def vacancy(external_id: str) -> Vacancy:
    return Vacancy(
        source="HH",
        external_id=external_id,
        title="Python Developer",
        company="Example",
        url=f"https://hh.ru/vacancy/{external_id}",
        location="Удалённо",
        salary="Зарплата не указана",
        salary_from=None,
        salary_to=None,
        currency="",
        description="Python backend",
        published_at=datetime.now(timezone.utc),
        remote=True,
    )


def settings(*, first_run_notify: bool = False) -> VacancyMonitorSettings:
    return VacancyMonitorSettings(
        enabled=True,
        parser_config=VacancyConfig(
            sources=("hh",),
            filter=VacancyFilter(keywords=("python",), remote_only=True),
            poll_seconds=60,
            per_source=20,
            store_file="state.json",
            first_run_notify=first_run_notify,
            user_agent="test/1.0",
        ),
        target_chat_id=1152363667,
        source_proxy_url=None,
        send_status_every_check=False,
        max_notifications_per_check=20,
    )


class VacancyMonitorTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_check_creates_baseline_then_only_sends_new_items(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            monitor = VacancyMonitor(settings(), Path(directory))
            monitor.parser.fetch = AsyncMock(return_value=[vacancy("1")])
            sender = AsyncMock()

            self.assertEqual(await monitor.check(sender), (1, 1, 0))
            sender.assert_not_awaited()
            self.assertTrue(monitor.store.is_seen("HH:1"))

            monitor.parser.fetch.return_value = [vacancy("1"), vacancy("2")]
            self.assertEqual(await monitor.check(sender), (2, 2, 1))
            self.assertEqual(sender.await_count, 1)
            self.assertTrue(monitor.store.is_seen("HH:2"))

    async def test_failed_send_does_not_lose_vacancy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            monitor = VacancyMonitor(settings(first_run_notify=True), Path(directory))
            item = vacancy("9")
            monitor.parser.fetch = AsyncMock(return_value=[item])
            sender = AsyncMock(side_effect=RuntimeError("telegram unavailable"))

            with self.assertRaises(RuntimeError):
                await monitor.check(sender)

            self.assertFalse(monitor.store.is_seen(item.key))
            self.assertIn("telegram unavailable", monitor.last_error or "")

    async def test_source_failure_and_recovery_are_recorded_without_telegram_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            monitor_settings = settings()
            monitor_settings = VacancyMonitorSettings(
                **{
                    **monitor_settings.__dict__,
                    "parser_config": VacancyConfig(
                        **{
                            **monitor_settings.parser_config.__dict__,
                            "sources": ("hh", "habr"),
                        }
                    ),
                }
            )
            monitor = VacancyMonitor(monitor_settings, Path(directory))
            monitor.parser.fetch = AsyncMock(return_value=[vacancy("1")])
            monitor.parser.last_source_errors = {"habr": "timeout"}
            monitor.parser.last_source_counts = {"hh": 1}
            sender = AsyncMock()

            for _ in range(3):
                await monitor.check(sender)

            sender.assert_not_awaited()
            self.assertEqual(monitor.store.health["sources"]["habr"]["consecutive_failures"], 3)

            monitor.parser.last_source_errors = {}
            monitor.parser.last_source_counts = {"hh": 1, "habr": 1}
            await monitor.check(sender)

            sender.assert_not_awaited()
            self.assertEqual(monitor.store.health["sources"]["habr"]["consecutive_failures"], 0)

    async def test_health_survives_monitor_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            monitor = VacancyMonitor(settings(), project)
            monitor.parser.fetch = AsyncMock(return_value=[vacancy("1")])

            await monitor.check(AsyncMock())
            restored = VacancyMonitor(settings(), project)

            self.assertEqual(restored.last_total, 1)
            self.assertEqual(restored.last_matches, 1)
            self.assertIsNotNone(restored.last_success_at)


if __name__ == "__main__":
    unittest.main()
