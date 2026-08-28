"""In-process vacancy monitoring independent from Telegram handler details."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable

from vacancy_parser import (
    Vacancy,
    VacancyConfig,
    VacancyFilter,
    VacancyParser,
    VacancyStore,
    filter_vacancies,
    format_vacancy,
)


LOGGER = logging.getLogger(__name__)
SendMessage = Callable[[int, str], Awaitable[None]]


@dataclass(frozen=True)
class VacancyMonitorSettings:
    enabled: bool
    parser_config: VacancyConfig
    target_chat_id: int
    source_proxy_url: str | None
    send_status_every_check: bool
    max_notifications_per_check: int


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _boolean(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _optional_text(value: object) -> str | None:
    clean = str(value or "").strip()
    return clean or None


def load_vacancy_monitor_settings(path: Path) -> VacancyMonitorSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Vacancy configuration must be a JSON object")
    enabled = _boolean(raw.get("enabled"), True)
    target_chat_id = int(raw.get("target_chat_id") or 0)
    if enabled and not target_chat_id:
        raise ValueError("target_chat_id is required when vacancy monitoring is enabled")
    vacancy_filter = VacancyFilter(
        keywords=_strings(raw.get("keywords")),
        required_keywords=_strings(raw.get("required_keywords")),
        excluded_keywords=_strings(raw.get("excluded_keywords")),
        locations=_strings(raw.get("locations")),
        remote_only=_boolean(raw.get("remote_only")),
        min_salary=max(0, int(raw.get("min_salary") or 0)),
        max_age_hours=max(0, int(raw.get("max_age_hours") or 72)),
        require_salary=_boolean(raw.get("require_salary")),
    )
    parser_config = VacancyConfig(
        sources=_strings(raw.get("sources")) or ("hh", "habr"),
        filter=vacancy_filter,
        poll_seconds=max(60, int(raw.get("poll_seconds") or 1800)),
        per_source=min(100, max(1, int(raw.get("per_source") or 50))),
        store_file=str(raw.get("store_file") or "runtime/vacancy_state.json"),
        first_run_notify=_boolean(raw.get("first_run_notify")),
        user_agent=str(
            raw.get("user_agent")
            or "TopTgBotVacancyMonitor/2.0 (contact the project owner)"
        ),
    )
    return VacancyMonitorSettings(
        enabled=enabled,
        parser_config=parser_config,
        target_chat_id=target_chat_id,
        source_proxy_url=_optional_text(raw.get("source_proxy_url")),
        send_status_every_check=_boolean(raw.get("send_status_every_check")),
        max_notifications_per_check=min(
            50, max(1, int(raw.get("max_notifications_per_check") or 20))
        ),
    )


class VacancyMonitor:
    def __init__(self, settings: VacancyMonitorSettings, project_directory: Path) -> None:
        self.settings = settings
        self.parser = VacancyParser(
            settings.parser_config,
            proxy_url=settings.source_proxy_url,
        )
        store_path = Path(settings.parser_config.store_file)
        if not store_path.is_absolute():
            store_path = project_directory / store_path
        self.store = VacancyStore(store_path)
        self._check_lock = asyncio.Lock()
        self.last_started_at: datetime | None = None
        self.last_finished_at: datetime | None = None
        self.next_check_at: datetime | None = None
        self.last_total = 0
        self.last_matches = 0
        self.last_new = 0
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return self.settings.enabled and not self.store.paused

    async def current(self) -> tuple[list[Vacancy], int]:
        async with self._check_lock:
            vacancies = await self.parser.fetch()
            return filter_vacancies(vacancies, self.settings.parser_config.filter), len(
                vacancies
            )

    async def check(self, sender: SendMessage, *, force_status: bool = False) -> tuple[int, int, int]:
        async with self._check_lock:
            self.last_started_at = datetime.now(timezone.utc)
            self.last_error = None
            try:
                vacancies = await self.parser.fetch()
                matches = filter_vacancies(
                    vacancies, self.settings.parser_config.filter
                )
                unseen = [item for item in matches if not self.store.is_seen(item.key)]
                baseline_created = False
                if not self.store.baseline_done and not self.settings.parser_config.first_run_notify:
                    self.store.mark_seen(item.key for item in matches)
                    self.store.finish_baseline()
                    unseen = []
                    baseline_created = True
                elif not self.store.baseline_done:
                    self.store.finish_baseline()

                sent = 0
                for vacancy in unseen[: self.settings.max_notifications_per_check]:
                    await sender(
                        self.settings.target_chat_id,
                        format_vacancy(vacancy),
                    )
                    self.store.mark_seen((vacancy.key,))
                    sent += 1
                remaining = len(unseen) - sent
                if remaining > 0:
                    await sender(
                        self.settings.target_chat_id,
                        f"Ещё новых вакансий в очереди: {remaining}. Они придут в следующих проверках.",
                    )

                self.last_total = len(vacancies)
                self.last_matches = len(matches)
                self.last_new = sent
                source_errors = self.parser.last_source_errors
                should_report = (
                    force_status
                    or self.settings.send_status_every_check
                    or bool(source_errors)
                )
                if should_report:
                    if baseline_created:
                        result = "Текущая выдача сохранена как начальная база."
                    elif sent:
                        result = f"Отправлено новых вакансий: {sent}."
                    else:
                        result = "Новых вакансий не найдено."
                    source_note = ""
                    if source_errors:
                        source_note = "\nОшибки источников: " + ", ".join(
                            sorted(source_errors)
                        )
                    await sender(
                        self.settings.target_chat_id,
                        f"{result}\nПроверено: {len(vacancies)}. Подходит: {len(matches)}.{source_note}",
                    )
                return len(vacancies), len(matches), sent
            except Exception as error:
                self.last_error = (str(error) or type(error).__name__)[:300]
                raise
            finally:
                self.last_finished_at = datetime.now(timezone.utc)

    async def run(self, sender: SendMessage) -> None:
        while True:
            if self.enabled:
                try:
                    await self.check(sender)
                except Exception:
                    LOGGER.exception("Vacancy check failed")
                    try:
                        await sender(
                            self.settings.target_chat_id,
                            "Проверка вакансий не выполнена. Автоматическая проверка продолжит работу.",
                        )
                    except Exception:
                        LOGGER.exception("Could not deliver vacancy failure notification")
            self.next_check_at = datetime.now(timezone.utc) + timedelta(
                seconds=self.settings.parser_config.poll_seconds
            )
            await asyncio.sleep(self.settings.parser_config.poll_seconds)

    def status_text(self) -> str:
        state = "включён" if self.enabled else "на паузе"
        last_check = (
            self.last_finished_at.astimezone().strftime("%d.%m.%Y %H:%M:%S")
            if self.last_finished_at
            else "ещё не выполнялась"
        )
        next_check = (
            self.next_check_at.astimezone().strftime("%d.%m.%Y %H:%M:%S")
            if self.next_check_at
            else "не запланирована"
        )
        error = f"\nПоследняя ошибка: {self.last_error}" if self.last_error else ""
        return (
            f"Мониторинг вакансий: {state}\n"
            f"Получатель: {self.settings.target_chat_id}\n"
            f"Источники: {', '.join(self.settings.parser_config.sources)}\n"
            f"Интервал: {self.settings.parser_config.poll_seconds // 60} мин.\n"
            f"Последняя проверка: {last_check}\n"
            f"Следующая проверка: {next_check}\n"
            f"Последний результат: проверено {self.last_total}, подходит {self.last_matches}, новых {self.last_new}"
            f"{error}"
        )
