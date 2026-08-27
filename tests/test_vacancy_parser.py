from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import httpx

from vacancy_parser import (
    HH_API_URL,
    HH_SEARCH_URL,
    Vacancy,
    VacancyConfig,
    VacancyFilter,
    VacancyParser,
    VacancyStore,
    filter_vacancies,
)


def parser_config(vacancy_filter: VacancyFilter) -> VacancyConfig:
    return VacancyConfig(
        sources=("hh", "habr"),
        filter=vacancy_filter,
        poll_seconds=300,
        per_source=20,
        store_file="vacancies.json",
        first_run_notify=False,
        user_agent="test-bot/1.0",
    )


class FakeClient:
    def __init__(self, responses: dict[str, httpx.Response]) -> None:
        self.responses = responses

    async def get(self, url: str, **kwargs: object) -> httpx.Response:
        del kwargs
        return self.responses[url]


class VacancyParserTests(unittest.IsolatedAsyncioTestCase):
    def test_resume_filter_accepts_matching_remote_python_role(self) -> None:
        vacancy_filter = VacancyFilter(
            keywords=("python", "django", "backend", "rest api"),
            required_keywords=("python",),
            excluded_keywords=("senior", "lead", "architect"),
            remote_only=True,
            min_salary=100000,
        )
        matching = Vacancy(
            source="HH",
            external_id="1",
            title="Python Backend Developer",
            company="Example",
            url="https://hh.ru/vacancy/1",
            location="Удалённо",
            salary="от 120 000 ₽",
            salary_from=120000,
            salary_to=None,
            currency="RUR",
            description="Django REST API PostgreSQL Docker",
            published_at=datetime.now(timezone.utc),
            remote=True,
        )
        senior = matching.__class__(**{**matching.__dict__, "title": "Senior Python Developer"})
        onsite = matching.__class__(**{**matching.__dict__, "remote": False})

        result = filter_vacancies([matching, senior, onsite], vacancy_filter)

        self.assertEqual([item.external_id for item in result], ["1"])

    async def test_hh_api_403_uses_html_fallback(self) -> None:
        html = """
        <div data-qa="vacancy-serp__vacancy">
          <a data-qa="serp-item__title" href="/vacancy/123">Python Developer</a>
          <div data-qa="vacancy-serp__vacancy-employer">Acme</div>
          <div data-qa="vacancy-serp__vacancy-address">Удалённо</div>
          <span data-qa="vacancy-serp__vacancy-compensation">от 120 000 ₽</span>
          <div data-qa="vacancy-serp__vacancy_snippet_requirement">Django REST API</div>
        </div>
        """
        api_request = httpx.Request("GET", HH_API_URL)
        search_request = httpx.Request("GET", HH_SEARCH_URL)
        client = FakeClient(
            {
                HH_API_URL: httpx.Response(403, request=api_request),
                HH_SEARCH_URL: httpx.Response(200, text=html, request=search_request),
            }
        )
        parser = VacancyParser(parser_config(VacancyFilter(keywords=("python",))), None)

        result = await parser._fetch_hh(client)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].external_id, "123")
        self.assertEqual(result[0].company, "Acme")
        self.assertTrue(result[0].remote)

    async def test_habr_html_parser_extracts_vacancy_card(self) -> None:
        html = """
        <div class="vacancy-card">
          <a href="/vacancies/456">Python Backend Developer</a>
          <a href="/companies/acme">Acme</a>
          <span>Можно удалённо</span>
          <span>от 150 000 ₽</span>
        </div>
        """
        request = httpx.Request("GET", "https://career.habr.com/vacancies")
        client = FakeClient(
            {
                "https://career.habr.com/vacancies": httpx.Response(
                    200, text=html, request=request
                )
            }
        )
        parser = VacancyParser(parser_config(VacancyFilter(keywords=("python",))), None)

        result = await parser._fetch_habr(client)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].external_id, "456")
        self.assertEqual(result[0].company, "Acme")
        self.assertTrue(result[0].remote)

    def test_store_remembers_seen_vacancies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = VacancyStore(Path(directory) / "vacancies.json")
            store.mark_seen(["HH:1"])

            self.assertTrue(store.is_seen("HH:1"))
            self.assertFalse(store.is_seen("HH:2"))


if __name__ == "__main__":
    unittest.main()
