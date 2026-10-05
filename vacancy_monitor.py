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
    source_failure_alert_after: int = 3


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
        hh_api_enabled=_boolean(raw.get("hh_api_enabled")),
        request_timeout_seconds=max(
            2.0, min(60.0, float(raw.get("request_timeout_seconds") or 8.0))
        ),
        source_attempts=max(1, min(5, int(raw.get("source_attempts") or 3))),
        source_retry_base_seconds=max(
            0.1, min(10.0, float(raw.get("source_retry_base_seconds") or 1.0))
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
        source_failure_alert_after=max(
            1, min(20, int(raw.get("source_failure_alert_after") or 3))
        ),
    )


def _parse_health_datetime(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


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
        health = self.store.health
        self.last_started_at = _parse_health_datetime(health.get("last_started_at"))
        self.last_finished_at = _parse_health_datetime(health.get("last_finished_at"))
        self.last_success_at = _parse_health_datetime(health.get("last_success_at"))
        self.next_check_at: datetime | None = None
        self.last_total = int(health.get("last_total") or 0)
        self.last_matches = int(health.get("last_matches") or 0)
        self.last_new = int(health.get("last_new") or 0)
        self.last_error = str(health.get("last_error") or "") or None

    @property
    def enabled(self) -> bool:
        return self.settings.enabled and not self.store.paused

    async def current(self) -> tuple[list[Vacancy], int]:
        async with self._check_lock:
            vacancies = await self.parser.fetch()
            return filter_vacancies(vacancies, self.settings.parser_config.filter), len(
                vacancies
            )

    def _update_source_health(self) -> None:
        health = dict(self.store.health)
        sources_value = health.get("sources", {})
        sources = dict(sources_value) if isinstance(sources_value, dict) else {}
        now = datetime.now(timezone.utc).isoformat()
        configured = {
            "habr" if source.strip().lower() == "habr_career" else source.strip().lower()
            for source in self.settings.parser_config.sources
        }
        for source in configured:
            previous_value = sources.get(source, {})
            previous = dict(previous_value) if isinstance(previous_value, dict) else {}
            if source in self.parser.last_source_errors:
                previous["consecutive_failures"] = int(
                    previous.get("consecutive_failures") or 0
                ) + 1
                previous["last_error"] = self.parser.last_source_errors[source]
                previous["last_error_at"] = now
            elif source in self.parser.last_source_counts:
                previous["consecutive_failures"] = 0
                previous["last_error"] = None
                previous["last_success_at"] = now
                previous["last_count"] = self.parser.last_source_counts[source]
            sources[source] = previous
        health["sources"] = sources
        self.store.update_health(health)

    def _record_check_result(self, *, success: bool) -> None:
        now = datetime.now(timezone.utc)
        self.last_finished_at = now
        health = dict(self.store.health)
        health.update(
            {
                "last_started_at": (
                    self.last_started_at.isoformat() if self.last_started_at else None
                ),
                "last_finished_at": now.isoformat(),
                "last_total": self.last_total,
                "last_matches": self.last_matches,
                "last_new": self.last_new,
                "last_error": self.last_error,
            }
        )
        if success:
            health["last_success_at"] = now.isoformat()
            health["consecutive_failures"] = 0
            self.last_success_at = now
        else:
            health["consecutive_failures"] = int(
                health.get("consecutive_failures") or 0
            ) + 1
        self.store.update_health(health)

    def _log_health_changes(self) -> None:
        health = dict(self.store.health)
        sources_value = health.get("sources", {})
        sources = dict(sources_value) if isinstance(sources_value, dict) else {}
        threshold = self.settings.source_failure_alert_after
        messages: list[str] = []
        changed = False
        for source, raw in sources.items():
            if not isinstance(raw, dict):
                continue
            state = dict(raw)
            failures = int(state.get("consecutive_failures") or 0)
            alerted = bool(state.get("alert_active"))
            if failures >= threshold and not alerted:
                messages.append(
                    f"Источник вакансий {source} недоступен уже {failures} проверки подряд. Остальные источники продолжают работать.",
                )
                state["alert_active"] = True
                changed = True
            elif failures == 0 and alerted:
                messages.append(f"Источник вакансий {source} снова работает.")
                state["alert_active"] = False
                changed = True
            sources[source] = state

        failures = int(health.get("consecutive_failures") or 0)
        monitor_alerted = bool(health.get("monitor_alert_active"))
        if failures >= threshold and not monitor_alerted:
            messages.append(
                f"Мониторинг вакансий не завершился уже {failures} раза подряд. Автоматические попытки продолжаются.",
            )
            health["monitor_alert_active"] = True
            changed = True
        elif failures == 0 and monitor_alerted:
            messages.append(
                "Мониторинг вакансий восстановился и снова выполняет проверки."
            )
            health["monitor_alert_active"] = False
            changed = True
        if changed:
            LOGGER.info("Vacancy health: %s", " | ".join(messages))
            health["sources"] = sources
            self.store.update_health(health)

    async def check(self, sender: SendMessage, *, force_status: bool = False) -> tuple[int, int, int]:
        async with self._check_lock:
            self.last_started_at = datetime.now(timezone.utc)
            self.last_error = None
            source_health_updated = False
            try:
                vacancies = await self.parser.fetch()
                self._update_source_health()
                source_health_updated = True
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
                should_report = (
                    force_status
                    or self.settings.send_status_every_check
                )
                if should_report:
                    if baseline_created:
                        result = "Текущая выдача сохранена как начальная база."
                    elif sent:
                        result = f"Отправлено новых вакансий: {sent}."
                    else:
                        result = "Новых вакансий не найдено."
                    await sender(
                        self.settings.target_chat_id,
                        f"{result}\nПроверено: {len(vacancies)}. Подходит: {len(matches)}.",
                    )
                self._record_check_result(success=True)
                try:
                    self._log_health_changes()
                except Exception:
                    LOGGER.exception("Could not record vacancy health change")
                return len(vacancies), len(matches), sent
            except Exception as error:
                self.last_error = (str(error) or type(error).__name__)[:300]
                if not source_health_updated:
                    self._update_source_health()
                self._record_check_result(success=False)
                try:
                    self._log_health_changes()
                except Exception:
                    LOGGER.exception("Could not record vacancy health change")
                raise

    async def run(self, sender: SendMessage) -> None:
        loop = asyncio.get_running_loop()
        next_deadline = loop.time()
        while True:
            if self.enabled:
                try:
                    await self.check(sender)
                except Exception:
                    LOGGER.exception("Vacancy check failed")
            interval = self.settings.parser_config.poll_seconds
            next_deadline += interval
            now = loop.time()
            if next_deadline <= now:
                missed = int((now - next_deadline) // interval) + 1
                next_deadline += missed * interval
            delay = max(0.0, next_deadline - now)
            self.next_check_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
            await asyncio.sleep(delay)

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
        last_success = (
            self.last_success_at.astimezone().strftime("%d.%m.%Y %H:%M:%S")
            if self.last_success_at
            else "ещё не было"
        )
        health = self.store.health
        sources_value = health.get("sources", {})
        source_lines: list[str] = []
        if isinstance(sources_value, dict):
            for source in sorted(sources_value):
                source_state = sources_value[source]
                if not isinstance(source_state, dict):
                    continue
                failures = int(source_state.get("consecutive_failures") or 0)
                count = int(source_state.get("last_count") or 0)
                source_lines.append(
                    f"{source}: {count}, ошибок подряд {failures}"
                )
        sources_status = (
            "\nПо источникам: " + "; ".join(source_lines) if source_lines else ""
        )
        return (
            f"Мониторинг вакансий: {state}\n"
            f"Получатель: {self.settings.target_chat_id}\n"
            f"Источники: {', '.join(self.settings.parser_config.sources)}\n"
            f"Интервал: {self.settings.parser_config.poll_seconds // 60} мин.\n"
            f"Последняя проверка: {last_check}\n"
            f"Последняя успешная: {last_success}\n"
            f"Следующая проверка: {next_check}\n"
            f"Последний результат: проверено {self.last_total}, подходит {self.last_matches}, новых {self.last_new}"
            f"{sources_status}"
            f"{error}"
        )
