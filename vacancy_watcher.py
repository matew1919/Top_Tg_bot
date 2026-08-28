"""Run the vacancy parser as a service independent from the Telegram bot."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

from vacancy_parser import (
    Vacancy,
    VacancyConfig,
    VacancyFilter,
    VacancyParser,
    VacancyStore,
    filter_vacancies,
    format_vacancy,
)


LOGGER = logging.getLogger("vacancy_watcher")
PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class WatcherSettings:
    parser_config: VacancyConfig
    target_chat_id: int
    source_proxy_url: str | None
    telegram_proxy_url: str | None
    send_status_every_check: bool
    max_notifications_per_check: int


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _optional_string(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def load_settings(path: Path) -> WatcherSettings:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Конфигурация должна быть JSON-объектом")

    vacancy_filter = VacancyFilter(
        keywords=_string_tuple(data.get("keywords")),
        required_keywords=_string_tuple(data.get("required_keywords")),
        excluded_keywords=_string_tuple(data.get("excluded_keywords")),
        locations=_string_tuple(data.get("locations")),
        remote_only=bool(data.get("remote_only", False)),
        min_salary=max(0, int(data.get("min_salary", 0))),
        max_age_hours=max(0, int(data.get("max_age_hours", 72))),
        require_salary=bool(data.get("require_salary", False)),
    )
    parser_config = VacancyConfig(
        sources=_string_tuple(data.get("sources")) or ("hh", "habr"),
        filter=vacancy_filter,
        poll_seconds=max(60, int(data.get("poll_seconds", 1800))),
        per_source=min(100, max(1, int(data.get("per_source", 50)))),
        store_file=str(data.get("store_file") or "vacancy_watcher_state.json"),
        first_run_notify=bool(data.get("first_run_notify", False)),
        user_agent=str(data.get("user_agent") or "StandaloneVacancyWatcher/1.0"),
    )
    target_chat_id = int(data.get("target_chat_id", 0))
    if not target_chat_id:
        raise ValueError("В target_chat_id должен быть указан Telegram chat ID")

    return WatcherSettings(
        parser_config=parser_config,
        target_chat_id=target_chat_id,
        source_proxy_url=_optional_string(data.get("source_proxy_url")),
        telegram_proxy_url=_optional_string(data.get("telegram_proxy_url")),
        send_status_every_check=bool(data.get("send_status_every_check", True)),
        max_notifications_per_check=min(
            50,
            max(1, int(data.get("max_notifications_per_check", 20))),
        ),
    )


class TelegramSender:
    def __init__(self, token: str, proxy_url: str | None) -> None:
        self.url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.client = httpx.AsyncClient(
            timeout=20,
            proxy=proxy_url,
            follow_redirects=True,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def send(self, chat_id: int, text: str) -> None:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = await self.client.post(
                    self.url,
                    json={
                        "chat_id": chat_id,
                        "text": text,
                        "disable_web_page_preview": True,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or payload.get("ok") is not True:
                    raise RuntimeError(str(payload.get("description") or "Telegram вернул ошибку"))
                return
            except (httpx.HTTPError, RuntimeError, ValueError) as error:
                last_error = error
                if attempt < 3:
                    await asyncio.sleep(attempt * 2)
        raise RuntimeError(f"Telegram: {last_error}")


class StandaloneVacancyWatcher:
    def __init__(self, settings: WatcherSettings, sender: TelegramSender | None) -> None:
        self.settings = settings
        self.sender = sender
        self.parser = VacancyParser(settings.parser_config, settings.source_proxy_url)
        store_path = Path(settings.parser_config.store_file)
        if not store_path.is_absolute():
            store_path = PROJECT_DIR / store_path
        self.store = VacancyStore(store_path)

    async def fetch_matches(self) -> tuple[list[Vacancy], int]:
        vacancies = await self.parser.fetch()
        matches = filter_vacancies(vacancies, self.settings.parser_config.filter)
        return matches, len(vacancies)

    async def dry_run(self) -> tuple[list[Vacancy], int]:
        matches, total = await self.fetch_matches()
        LOGGER.info("Проверено: %s; подходит: %s", total, len(matches))
        for vacancy in matches[:10]:
            print(format_vacancy(vacancy))
            print()
        return matches, total

    async def check_and_notify(self) -> tuple[int, int, int]:
        if self.sender is None:
            raise RuntimeError("Telegram sender is not configured")

        matches, total = await self.fetch_matches()
        new_vacancies = [item for item in matches if not self.store.is_seen(item.key)]
        baseline_created = False

        if not self.store.baseline_done and not self.settings.parser_config.first_run_notify:
            self.store.mark_seen(item.key for item in matches)
            self.store.finish_baseline()
            new_vacancies = []
            baseline_created = True
        else:
            self.store.mark_seen(item.key for item in new_vacancies)
            if not self.store.baseline_done:
                self.store.finish_baseline()

        limit = self.settings.max_notifications_per_check
        for vacancy in new_vacancies[:limit]:
            await self.sender.send(self.settings.target_chat_id, format_vacancy(vacancy))
        if len(new_vacancies) > limit:
            await self.sender.send(
                self.settings.target_chat_id,
                f"Ещё найдено вакансий: {len(new_vacancies) - limit}. Показаны первые {limit}.",
            )

        if self.settings.send_status_every_check:
            if baseline_created:
                status = "Мониторинг вакансий запущен. Текущая выдача сохранена как начальная база."
            elif new_vacancies:
                status = f"Найдено новых вакансий: {len(new_vacancies)}."
            else:
                status = "Новых вакансий не найдено."
            await self.sender.send(
                self.settings.target_chat_id,
                f"{status}\nПроверено: {total}. Подходит под фильтр: {len(matches)}.",
            )

        LOGGER.info(
            "Проверено: %s; подходит: %s; новых: %s",
            total,
            len(matches),
            len(new_vacancies),
        )
        return total, len(matches), len(new_vacancies)


class ProcessLock:
    def __init__(self, path: Path, port: int = 47654) -> None:
        self.path = path
        self.port = port
        self.socket: socket.socket | None = None

    def __enter__(self) -> "ProcessLock":
        lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            lock.bind(("127.0.0.1", self.port))
            lock.listen(1)
        except OSError as error:
            lock.close()
            raise RuntimeError("Мониторинг вакансий уже запущен") from error
        self.socket = lock
        self.path.write_text(str(os.getpid()), encoding="utf-8")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.socket is not None:
            self.socket.close()


async def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    env_path = Path(args.env_file).expanduser().resolve()
    load_dotenv(env_path, override=False)
    settings = load_settings(config_path)

    if args.dry_run:
        watcher = StandaloneVacancyWatcher(settings, None)
        await watcher.dry_run()
        return 0

    token = str(os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise RuntimeError(f"TELEGRAM_BOT_TOKEN не найден в {env_path}")

    sender = TelegramSender(token, settings.telegram_proxy_url)
    watcher = StandaloneVacancyWatcher(settings, sender)
    try:
        if args.delay_first and not args.once:
            LOGGER.info(
                "Следующая проверка через %s секунд",
                settings.parser_config.poll_seconds,
            )
            await asyncio.sleep(settings.parser_config.poll_seconds)
        while True:
            try:
                await watcher.check_and_notify()
            except Exception as error:
                LOGGER.exception("Проверка вакансий завершилась ошибкой")
                try:
                    await sender.send(
                        settings.target_chat_id,
                        f"⚠️ Проверка вакансий не выполнена: {error}",
                    )
                except Exception:
                    LOGGER.exception("Не удалось отправить уведомление об ошибке")
            if args.once:
                return 0
            await asyncio.sleep(settings.parser_config.poll_seconds)
    finally:
        await sender.close()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone HH/Habr vacancy watcher")
    parser.add_argument("--config", default=str(PROJECT_DIR / "vacancy_watcher_config.json"))
    parser.add_argument("--env-file", default=str(PROJECT_DIR / ".env"))
    parser.add_argument("--once", action="store_true", help="Проверить один раз и завершить")
    parser.add_argument(
        "--delay-first",
        action="store_true",
        help="В постоянном режиме подождать один интервал перед первой проверкой",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Проверить источники без Telegram и изменения состояния",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # httpx logs request URLs; Telegram URLs contain the bot token.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    lock_path = PROJECT_DIR / "vacancy_watcher.pid"
    try:
        with ProcessLock(lock_path):
            return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        LOGGER.error("Не удалось запустить мониторинг: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
