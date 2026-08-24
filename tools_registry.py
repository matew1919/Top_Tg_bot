"""Native LLM tools and their registry.

The registry keeps provider schemas next to executable handlers, so adding a new
tool does not require changing the tool-calling loop.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Awaitable, Callable

from ddgs import DDGS


ToolHandler = Callable[[dict[str, object], "ToolContext"], Awaitable[dict[str, object]]]


@dataclass(frozen=True)
class ToolContext:
    chat_id: int
    user_id: int
    user_name: str
    timezone: str


@dataclass(frozen=True)
class RegisteredTool:
    description: str
    parameters: dict[str, object]
    handler: ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, object],
        handler: ToolHandler,
    ) -> None:
        self._tools[name] = RegisteredTool(description, parameters, handler)

    @property
    def schemas(self) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for name, tool in self._tools.items()
        ]

    async def execute(
        self,
        name: str,
        raw_arguments: str | dict[str, object] | None,
        context: ToolContext,
    ) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return json.dumps(
                {"ok": False, "error": f"Unknown tool: {name}"}, ensure_ascii=False
            )
        try:
            if isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                decoded = json.loads(raw_arguments or "{}")
                if not isinstance(decoded, dict):
                    raise ValueError("tool arguments must be a JSON object")
                arguments = decoded
            result = await tool.handler(arguments, context)
            return json.dumps(result, ensure_ascii=False)
        except Exception as error:
            return json.dumps(
                {"ok": False, "error": str(error) or type(error).__name__},
                ensure_ascii=False,
            )


async def web_search(
    arguments: dict[str, object], context: ToolContext, *, proxy: str | None = None
) -> dict[str, object]:
    del context
    query = str(arguments.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")

    def search() -> list[dict[str, str]]:
        raw_results = DDGS(proxy=proxy, timeout=10).text(
            query,
            max_results=5,
            backend="brave,wikipedia",
        )
        results: list[dict[str, str]] = []
        for item in raw_results or []:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": str(item.get("title") or "").strip(),
                    "url": str(item.get("href") or item.get("url") or "").strip(),
                    "snippet": str(item.get("body") or item.get("snippet") or "").strip()[:1000],
                }
            )
        return results

    results = await asyncio.to_thread(search)
    return {"ok": True, "query": query, "results": results}


def register_builtin_tools(
    registry: ToolRegistry,
    create_reminder: ToolHandler,
    *,
    search_proxy: str | None = None,
) -> None:
    async def search_handler(
        arguments: dict[str, object], context: ToolContext
    ) -> dict[str, object]:
        return await web_search(arguments, context, proxy=search_proxy)

    registry.register(
        "web_search",
        "Search the internet for current or factual information and return titles, URLs and snippets.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise internet search query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        search_handler,
    )
    registry.register(
        "create_reminder",
        "Create a reminder for the current Telegram user using the existing reminder system.",
        {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": (
                        "The user's complete reminder request, including the time and what to remind about."
                    ),
                }
            },
            "required": ["request"],
            "additionalProperties": False,
        },
        create_reminder,
    )
