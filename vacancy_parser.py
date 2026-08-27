"""Low-cost vacancy collection for HH.ru and Habr Career.

The module deliberately does not call an LLM. Sources are fetched, vacancies are
filtered locally, and a small JSON store prevents duplicate Telegram messages.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone as dt_timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup


LOGGER = logging.getLogger(__name__)
HH_API_URL = "https://api.hh.ru/vacancies"
HH_SEARCH_URL = "https://hh.ru/search/vacancy"
HABR_VACANCIES_URL = "https://career.habr.com/vacancies"


@dataclass(frozen=True)
class Vacancy:
    source: str
    external_id: str
    title: str
    company: str
    url: str
    location: str
    salary: str
    salary_from: int | None
    salary_to: int | None
    currency: str
    description: str
    published_at: datetime | None = None
    remote: bool = False

    @property
    def key(self) -> str:
        return f"{self.source}:{self.external_id}"

    @property
    def searchable_text(self) -> str:
        return " ".join(
            (
                self.title,
                self.company,
                self.location,
                self.description,
                self.salary,
            )
        ).lower()


@dataclass(frozen=True)
class VacancyFilter:
    keywords: tuple[str, ...] = ()
    required_keywords: tuple[str, ...] = ()
    excluded_keywords: tuple[str, ...] = ()
    locations: tuple[str, ...] = ()
    remote_only: bool = False
    min_salary: int = 0
    max_age_hours: int = 72
    require_salary: bool = False

    def matches(self, vacancy: Vacancy, now: datetime) -> bool:
        haystack = vacancy.searchable_text
        include = tuple(item.lower().strip() for item in self.keywords if item.strip())
        required = tuple(item.lower().strip() for item in self.required_keywords if item.strip())
        excluded = tuple(item.lower().strip() for item in self.excluded_keywords if item.strip())
        locations = tuple(item.lower().strip() for item in self.locations if item.strip())

        if include and not any(item in haystack for item in include):
            return False
        if required and not all(item in haystack for item in required):
            return False
        if excluded and any(item in haystack for item in excluded):
            return False
        if locations and not any(item in haystack for item in locations):
            return False
        if self.remote_only and not vacancy.remote:
            return False
        if self.require_salary and vacancy.salary_from is None and vacancy.salary_to is None:
            return False
        if self.min_salary:
            highest_known = vacancy.salary_to or vacancy.salary_from
            if highest_known is None or highest_known < self.min_salary:
                return False
        if vacancy.published_at and self.max_age_hours > 0:
            published = vacancy.published_at
            if published.tzinfo is None:
                published = published.replace(tzinfo=dt_timezone.utc)
            if published < now - timedelta(hours=self.max_age_hours):
                return False
        return True


@dataclass(frozen=True)
class VacancyConfig:
    sources: tuple[str, ...]
    filter: VacancyFilter
    poll_seconds: int
    per_source: int
    store_file: str
    first_run_notify: bool
    user_agent: str


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _parse_iso(value: object) -> datetime | None:
    raw = _clean(value)
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _salary_text(salary_from: int | None, salary_to: int | None, currency: str) -> str:
    if salary_from is None and salary_to is None:
        return "Зарплата не указана"
    symbol = {"RUR": "₽", "RUB": "₽", "USD": "$", "EUR": "€"}.get(currency, currency)
    if salary_from is not None and salary_to is not None:
        return f"{salary_from:,}–{salary_to:,} {symbol}".replace(",", " ")
    if salary_from is not None:
        return f"от {salary_from:,} {symbol}".replace(",", " ")
    return f"до {salary_to:,} {symbol}".replace(",", " ")


def _to_int(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


class VacancyStore:
    """Persistent subscriptions and already-notified vacancy keys."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, object] = {
            "version": 1,
            "subscriptions": [],
            "seen": {},
            "baseline_done": False,
        }
        self._load()

    def _load(self) -> None:
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Could not load vacancy store %s; starting empty", self.path)
            return
        if isinstance(loaded, dict):
            self.data.update(loaded)
        if not isinstance(self.data.get("subscriptions"), list):
            self.data["subscriptions"] = []
        if not isinstance(self.data.get("seen"), dict):
            self.data["seen"] = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def subscribe(self, chat_id: int) -> None:
        subscriptions = self.data["subscriptions"]
        assert isinstance(subscriptions, list)
        if chat_id not in subscriptions:
            subscriptions.append(chat_id)
            self._save()

    def unsubscribe(self, chat_id: int) -> None:
        subscriptions = self.data["subscriptions"]
        assert isinstance(subscriptions, list)
        self.data["subscriptions"] = [item for item in subscriptions if item != chat_id]
        self._save()

    def subscribed_chat_ids(self) -> list[int]:
        subscriptions = self.data.get("subscriptions", [])
        if not isinstance(subscriptions, list):
            return []
        return [int(item) for item in subscriptions if isinstance(item, int) or str(item).lstrip("-").isdigit()]

    def is_seen(self, key: str) -> bool:
        seen = self.data.get("seen", {})
        return isinstance(seen, dict) and key in seen

    def mark_seen(self, keys: Iterable[str]) -> None:
        seen = self.data.setdefault("seen", {})
        if not isinstance(seen, dict):
            seen = {}
            self.data["seen"] = seen
        timestamp = datetime.now(dt_timezone.utc).isoformat()
        for key in keys:
            seen[str(key)] = timestamp
        # Keep the file small while preserving the newest entries.
        if len(seen) > 5000:
            newest = sorted(seen.items(), key=lambda item: str(item[1]), reverse=True)[:4000]
            self.data["seen"] = dict(newest)
        self._save()

    @property
    def baseline_done(self) -> bool:
        return bool(self.data.get("baseline_done"))

    def finish_baseline(self) -> None:
        self.data["baseline_done"] = True
        self._save()


class VacancyParser:
    def __init__(self, config: VacancyConfig, proxy_url: str | None = None) -> None:
        self.config = config
        self.proxy_url = proxy_url

    async def fetch(self) -> list[Vacancy]:
        headers = {
            "User-Agent": self.config.user_agent,
            "HH-User-Agent": self.config.user_agent,
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.7",
        }
        async with httpx.AsyncClient(
            headers=headers,
            timeout=25,
            follow_redirects=True,
            proxy=self.proxy_url,
        ) as client:
            result: list[Vacancy] = []
            for source in self.config.sources:
                normalized = source.strip().lower()
                try:
                    if normalized == "hh":
                        result.extend(await self._fetch_hh(client))
                    elif normalized in {"habr", "habr_career"}:
                        result.extend(await self._fetch_habr(client))
                    else:
                        LOGGER.warning("Unknown vacancy source: %s", source)
                except Exception:
                    LOGGER.exception("Vacancy source failed: %s", source)
            return result

    async def _fetch_hh(self, client: httpx.AsyncClient) -> list[Vacancy]:
        params: dict[str, object] = {
            "per_page": min(100, max(1, self.config.per_source)),
            "page": 0,
            "order_by": "publication_time",
        }
        keywords = [item.strip() for item in self.config.filter.keywords if item.strip()]
        if keywords:
            # The local filter remains authoritative; this query only narrows the API response.
            params["text"] = " OR ".join(keywords)
        if self.config.filter.max_age_hours > 0:
            params["period"] = min(30, max(1, (self.config.filter.max_age_hours + 23) // 24))
        try:
            response = await client.get(HH_API_URL, params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 403:
                raise
            LOGGER.warning("HH API returned 403; falling back to the public search page")
            return await self._fetch_hh_html(client, keywords)
        payload = response.json()
        items = payload.get("items", []) if isinstance(payload, dict) else []
        vacancies: list[Vacancy] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            salary = item.get("salary") if isinstance(item.get("salary"), dict) else {}
            salary_from = _to_int(salary.get("from"))
            salary_to = _to_int(salary.get("to"))
            currency = _clean(salary.get("currency"))
            schedule = item.get("schedule") if isinstance(item.get("schedule"), dict) else {}
            work_format = item.get("work_format") if isinstance(item.get("work_format"), list) else []
            format_text = " ".join(
                _clean(value.get("name")) for value in work_format if isinstance(value, dict)
            )
            remote = "remote" in _clean(schedule.get("id")).lower() or "удал" in (
                _clean(schedule.get("name")) + " " + format_text
            ).lower()
            area = item.get("area") if isinstance(item.get("area"), dict) else {}
            employer = item.get("employer") if isinstance(item.get("employer"), dict) else {}
            vacancy_id = _clean(item.get("id"))
            if not vacancy_id:
                continue
            vacancies.append(
                Vacancy(
                    source="HH",
                    external_id=vacancy_id,
                    title=_clean(item.get("name")) or "Без названия",
                    company=_clean(employer.get("name")) or "Компания не указана",
                    url=_clean(item.get("alternate_url") or item.get("url")),
                    location=_clean(area.get("name")) or "Локация не указана",
                    salary=_salary_text(salary_from, salary_to, currency),
                    salary_from=salary_from,
                    salary_to=salary_to,
                    currency=currency,
                    description=_clean(item.get("snippet")),
                    published_at=_parse_iso(item.get("published_at")),
                    remote=remote,
                )
            )
        LOGGER.info("HH returned %s vacancies", len(vacancies))
        return vacancies

    async def _fetch_hh_html(
        self,
        client: httpx.AsyncClient,
        keywords: list[str],
    ) -> list[Vacancy]:
        params: dict[str, object] = {
            "items_on_page": min(100, max(1, self.config.per_source)),
            "page": 0,
            "order_by": "publication_time",
        }
        if keywords:
            params["text"] = " OR ".join(keywords)
        if self.config.filter.max_age_hours > 0:
            params["search_period"] = min(30, max(1, (self.config.filter.max_age_hours + 23) // 24))
        response = await client.get(HH_SEARCH_URL, params=params)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        vacancies: list[Vacancy] = []
        now = datetime.now(dt_timezone.utc)
        cards = soup.select("[data-qa='vacancy-serp__vacancy']")
        if not cards:
            cards = soup.select("div.serp-item")
        for card in cards:
            title_link = card.select_one("a[data-qa='serp-item__title'], a.serp-item__title")
            if title_link is None:
                continue
            title = _clean(title_link.get_text(" ", strip=True))
            href = _clean(title_link.get("href"))
            vacancy_match = re.search(r"/vacancy/(\d+)", href)
            if not title or not vacancy_match:
                continue
            company_node = card.select_one(
                "[data-qa='vacancy-serp__vacancy-employer'], "
                ".vacancy-serp-item__meta-info-company"
            )
            location_node = card.select_one(
                "[data-qa='vacancy-serp__vacancy-address'], "
                ".vacancy-serp__vacancy-address"
            )
            salary_node = card.select_one(
                "[data-qa='vacancy-serp__vacancy-compensation'], "
                ".vacancy-serp__vacancy-compensation"
            )
            description_node = card.select_one(
                "[data-qa='vacancy-serp__vacancy_snippet_requirement'], "
                "[data-qa='vacancy-serp__vacancy_snippet_responsibility']"
            )
            card_text = _clean(card.get_text(" ", strip=True))
            salary_text = _clean(salary_node.get_text(" ", strip=True)) if salary_node else ""
            salary_from, salary_to, currency = self._parse_hh_salary(salary_text)
            remote = bool(re.search(r"удалённо|удаленно|remote", card_text, re.IGNORECASE))
            vacancies.append(
                Vacancy(
                    source="HH",
                    external_id=vacancy_match.group(1),
                    title=title,
                    company=(
                        _clean(company_node.get_text(" ", strip=True))
                        if company_node
                        else "Компания не указана"
                    ),
                    url=urljoin("https://hh.ru", href),
                    location=(
                        _clean(location_node.get_text(" ", strip=True))
                        if location_node
                        else "Локация не указана"
                    ),
                    salary=_salary_text(salary_from, salary_to, currency),
                    salary_from=salary_from,
                    salary_to=salary_to,
                    currency=currency,
                    description=(
                        _clean(description_node.get_text(" ", strip=True))
                        if description_node
                        else card_text[:1000]
                    ),
                    published_at=now,
                    remote=remote,
                )
            )
            if len(vacancies) >= self.config.per_source:
                break
        LOGGER.info("HH HTML search returned %s vacancies", len(vacancies))
        return vacancies

    @staticmethod
    def _parse_hh_salary(value: str) -> tuple[int | None, int | None, str]:
        if not value:
            return None, None, ""
        numbers = [int(re.sub(r"\D", "", item)) for item in re.findall(r"[\d\s]+", value) if re.sub(r"\D", "", item)]
        currency_match = re.search(r"₽|руб\.?|\$|USD|€|EUR", value, re.IGNORECASE)
        currency = currency_match.group(0).upper() if currency_match else ""
        if len(numbers) >= 2:
            return numbers[0], numbers[1], currency
        if numbers:
            if re.search(r"до", value, re.IGNORECASE):
                return None, numbers[0], currency
            return numbers[0], None, currency
        return None, None, currency

    async def _fetch_habr(self, client: httpx.AsyncClient) -> list[Vacancy]:
        response = await client.get(HABR_VACANCIES_URL, params={"page": 1})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        vacancies: list[Vacancy] = []
        seen_urls: set[str] = set()
        now = datetime.now(dt_timezone.utc)
        for link in soup.select('a[href*="/vacancies/"]'):
            href = _clean(link.get("href"))
            if not re.search(r"/vacancies/[^/?#]+", href):
                continue
            url = urljoin(HABR_VACANCIES_URL, href)
            if url in seen_urls:
                continue
            title = _clean(link.get_text(" ", strip=True))
            if not title or title.lower() in {"откликнуться", "respond"}:
                continue
            card = self._find_card(link)
            card_text = _clean(card.get_text(" ", strip=True)) if card else title
            company = "Компания не указана"
            if card:
                company_link = card.select_one('a[href*="/companies/"]')
                if company_link:
                    company = _clean(company_link.get_text(" ", strip=True)) or company
            salary_match = re.search(
                r"(?:от\s+)?([\d\s]+)(?:\s*[-–]\s*([\d\s]+))?\s*(₽|руб\.?|\$|USD|€|EUR)",
                card_text,
                re.IGNORECASE,
            )
            salary_from = salary_to = None
            currency = ""
            if salary_match:
                salary_from = int(re.sub(r"\D", "", salary_match.group(1)))
                salary_to = (
                    int(re.sub(r"\D", "", salary_match.group(2)))
                    if salary_match.group(2)
                    else None
                )
                currency = salary_match.group(3).upper()
            remote = bool(re.search(r"удалённо|удаленно|remote", card_text, re.IGNORECASE))
            location_match = re.search(
                r"(?:Можно удалённо|Можно удаленно|Удалённо|Удаленно|Remote|Middle|Senior|Junior|Lead|Intern)\s+(.+?)(?=\s+(?:[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё+.#0-9 -]{1,30})\s*$|$)",
                card_text,
                re.IGNORECASE,
            )
            location = _clean(location_match.group(1)) if location_match else "Локация не указана"
            external_id = url.rstrip("/").rsplit("/", 1)[-1]
            seen_urls.add(url)
            vacancies.append(
                Vacancy(
                    source="Habr",
                    external_id=external_id,
                    title=title,
                    company=company,
                    url=url,
                    location=location,
                    salary=_salary_text(salary_from, salary_to, currency),
                    salary_from=salary_from,
                    salary_to=salary_to,
                    currency=currency,
                    description=card_text[:1000],
                    published_at=now,
                    remote=remote,
                )
            )
            if len(vacancies) >= self.config.per_source:
                break
        LOGGER.info("Habr returned %s vacancies", len(vacancies))
        return vacancies

    @staticmethod
    def _find_card(link: object):
        current = link
        for _ in range(5):
            current = getattr(current, "parent", None)
            if current is None:
                return None
            text = _clean(current.get_text(" ", strip=True))
            classes = " ".join(current.get("class", [])) if hasattr(current, "get") else ""
            if "vacancy" in classes.lower() or len(text) >= 80:
                return current
        return None


def filter_vacancies(vacancies: Iterable[Vacancy], vacancy_filter: VacancyFilter) -> list[Vacancy]:
    now = datetime.now(dt_timezone.utc)
    return [vacancy for vacancy in vacancies if vacancy_filter.matches(vacancy, now)]


def format_vacancy(vacancy: Vacancy) -> str:
    parts = [f"Новая вакансия [{vacancy.source}]", vacancy.title]
    if vacancy.company:
        parts.append(f"Компания: {vacancy.company}")
    if vacancy.salary:
        parts.append(f"Зарплата: {vacancy.salary}")
    if vacancy.location:
        parts.append(f"Место: {vacancy.location}")
    parts.append(vacancy.url)
    return "\n".join(parts)
