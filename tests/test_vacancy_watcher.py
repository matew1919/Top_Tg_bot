from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vacancy_parser import Vacancy
from vacancy_watcher import StandaloneVacancyWatcher, load_settings


class FakeSender:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str]] = []

    async def send(self, chat_id: int, text: str) -> None:
        self.messages.append((chat_id, text))


class StandaloneVacancyWatcherTests(unittest.IsolatedAsyncioTestCase):
    def make_settings(self, directory: str, first_run_notify: bool = False):
        path = Path(directory) / "config.json"
        path.write_text(
            json.dumps(
                {
                    "sources": ["hh"],
                    "keywords": ["python"],
                    "required_keywords": ["python"],
                    "poll_seconds": 300,
                    "per_source": 10,
                    "store_file": str(Path(directory) / "state.json"),
                    "first_run_notify": first_run_notify,
                    "target_chat_id": 123,
                    "send_status_every_check": True,
                }
            ),
            encoding="utf-8",
        )
        return load_settings(path)

    @staticmethod
    def vacancy() -> Vacancy:
        return Vacancy(
            source="HH",
            external_id="1",
            title="Python developer",
            company="Example",
            url="https://hh.ru/vacancy/1",
            location="Удалённо",
            salary="Зарплата не указана",
            salary_from=None,
            salary_to=None,
            currency="",
            description="Python backend",
            remote=True,
        )

    async def test_first_run_creates_baseline_without_vacancy_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sender = FakeSender()
            watcher = StandaloneVacancyWatcher(self.make_settings(directory), sender)

            async def fake_fetch():
                return [self.vacancy()], 1

            watcher.fetch_matches = fake_fetch
            result = await watcher.check_and_notify()

            self.assertEqual(result, (1, 1, 0))
            self.assertEqual(len(sender.messages), 1)
            self.assertIn("начальная база", sender.messages[0][1])

    async def test_new_vacancy_is_sent_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sender = FakeSender()
            watcher = StandaloneVacancyWatcher(
                self.make_settings(directory, first_run_notify=True),
                sender,
            )

            async def fake_fetch():
                return [self.vacancy()], 1

            watcher.fetch_matches = fake_fetch
            first = await watcher.check_and_notify()
            second = await watcher.check_and_notify()

            self.assertEqual(first, (1, 1, 1))
            self.assertEqual(second, (1, 1, 0))
            vacancy_messages = [text for _, text in sender.messages if "Новая вакансия" in text]
            self.assertEqual(len(vacancy_messages), 1)


if __name__ == "__main__":
    unittest.main()
