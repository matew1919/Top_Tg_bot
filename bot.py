"""Telegram assistant bot.

The bot answers in group chats when:
1. it is mentioned with @bot_username (or addressed with a text mention); or
2. a message is written in Latin/English characters.

Set GIGACHAT_CREDENTIALS to enable useful generated answers. Without it the bot uses
a safe fallback response, which makes it possible to check Telegram wiring
before connecting an AI provider.
"""

from __future__ import annotations

import logging
import os
import re
import asyncio
import json
import threading
from base64 import b64encode
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone as dt_timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from gigachat import GigaChat
from openai import OpenAI
from telegram import Message, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


LOGGER = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_runtime_config() -> dict[str, object]:
    """Load non-secret bot settings from the project configuration file."""

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Configuration file is missing: {CONFIG_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in configuration file: {CONFIG_PATH}") from exc

    if not isinstance(config, dict):
        raise RuntimeError(f"Configuration root must be a JSON object: {CONFIG_PATH}")
    return config


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    ai_provider: str
    gigachat_credentials: str | None
    gigachat_scope: str
    gigachat_model: str
    gigachat_base_url: str
    gigachat_verify_ssl: bool
    gigachat_timeout: float
    gigachat_max_retries: int
    vsegpt_api_key: str | None
    vsegpt_model: str
    vsegpt_base_url: str
    vsegpt_max_tokens: int
    vsegpt_vision_model: str
    vision_prompt: str
    telegram_timeout: float
    system_prompt: str
    english_coach_prompt: str
    soft_context_limit: int
    hard_context_limit: int
    keep_last_messages: int
    memory_file: str
    memory_max_tokens: int
    memory_prompt: str
    max_english_history_messages: int
    reminders_file: str
    reminder_schedule_prompt: str
    reminder_prompt: str
    timezone: str
    reminder_poll_seconds: int
    log_level: str
    api_journal_file: str
    api_journal_max_entries: int
    api_journal_enabled: bool

    @classmethod
    def from_env(cls) -> "Settings":
        config = load_runtime_config()

        def value(name: str, default: object) -> object:
            return config.get(name, default)

        def text(name: str, default: str) -> str:
            return str(value(name, default)).strip()

        def integer(name: str, default: int) -> int:
            return int(value(name, default))

        def number(name: str, default: float) -> float:
            return float(value(name, default))

        def boolean(name: str, default: bool) -> bool:
            raw = value(name, default)
            if isinstance(raw, bool):
                return raw
            return str(raw).strip().lower() in {"1", "true", "yes", "on"}

        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

        provider = text("ai_provider", "gigachat").lower()
        if provider not in {"gigachat", "vsegpt"}:
            raise RuntimeError("AI_PROVIDER must be either 'gigachat' or 'vsegpt'")
        vsegpt_api_key = os.getenv("VSEGPT_API_KEY", "").strip() or None

        return cls(
            telegram_token=token,
            ai_provider=provider,
            # OPENAI_* fallbacks keep compatibility with the first .env file,
            # where the GigaChat authorization key was entered under that name.
            gigachat_credentials=(
                os.getenv("GIGACHAT_CREDENTIALS", "").strip()
                or os.getenv("OPENAI_API_KEY", "").strip()
                or None
            ),
            gigachat_scope=text("gigachat_scope", "GIGACHAT_API_PERS"),
            gigachat_model=text("gigachat_model", "GigaChat-2-Pro"),
            gigachat_base_url=text("gigachat_base_url", "https://api.giga.chat/v1"),
            gigachat_verify_ssl=bool(value("gigachat_verify_ssl", False)),
            gigachat_timeout=number("gigachat_timeout", 60),
            gigachat_max_retries=max(0, integer("gigachat_max_retries", 3)),
            vsegpt_api_key=vsegpt_api_key,
            vsegpt_model=text("vsegpt_model", "openai/gpt-4o-mini"),
            vsegpt_base_url=text("vsegpt_base_url", "https://api.vsegpt.ru/v1"),
            vsegpt_max_tokens=max(32, integer("vsegpt_max_tokens", 200)),
            vsegpt_vision_model=text("vsegpt_vision_model", "vis-google/gemini-3-flash-pre"),
            vision_prompt=text(
                "vision_prompt", "Reply briefly in Russian like in a messenger. Do not use Markdown."
            ),
            telegram_timeout=number("telegram_timeout", 30),
            system_prompt=text(
                "system_prompt",
                "Ты полезный ассистент в Telegram-чате. Отвечай кратко и по делу на русском языке.",
            ),
            english_coach_prompt=text(
                "english_coach_prompt",
                "Act as a strict but friendly English teacher. Be concise and use plain text.",
            ),
            soft_context_limit=max(2, integer("soft_context_limit", 8)),
            hard_context_limit=max(3, integer("hard_context_limit", 20)),
            keep_last_messages=max(1, integer("keep_last_messages", 2)),
            memory_file=text("memory_file", "memory.json") or "memory.json",
            memory_max_tokens=max(64, integer("memory_max_tokens", 150)),
            memory_prompt=text(
                "memory_prompt",
                "Изучи CURRENT MEMORY и FULL CONVERSATION HISTORY. Верни полную "
                "обновлённую память по чату и всем пользователям. Сохрани все факты "
                "из текущей памяти и SHARED USER PROFILES. Добавь новые факты. Не удаляй факт, если он не "
                "упомянут в истории. Верни только строки CHAT_FACT: факт или "
                "USER_FACT: user_id|name|факт. Если фактов нет, верни EMPTY.",
            ),
            max_english_history_messages=max(1, integer("max_english_history_messages", 1)),
            reminders_file=text("reminders_file", "reminders.json") or "reminders.json",
            reminder_schedule_prompt=text(
                "reminder_schedule_prompt",
                "Определи время напоминания и верни TIME и TEXT отдельными строками.",
            ),
            reminder_prompt=text(
                "reminder_prompt",
                "Сформулируй одну короткую фразу напоминания на русском без Markdown.",
            ),
            timezone=text("timezone", "Europe/Moscow") or "Europe/Moscow",
            reminder_poll_seconds=max(5, integer("reminder_poll_seconds", 15)),
            log_level=text("log_level", "INFO").upper(),
            api_journal_file=text("api_journal_file", "api_journal.jsonl")
            or "api_journal.jsonl",
            api_journal_max_entries=max(1, integer("api_journal_max_entries", 200)),
            api_journal_enabled=boolean("api_journal_enabled", True),
        )


def contains_english(text: str) -> bool:
    """Return True when a message contains an English word.

    The check deliberately works for mixed Russian/English messages too, e.g.
    "Я написал I has a car". URLs and @mentions do not count as English text.
    """

    text_without_urls = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.I)
    text_without_mentions = re.sub(r"@[A-Za-z0-9_]+", "", text_without_urls)
    return bool(re.search(r"\b[A-Za-z]{2,}(?:'[A-Za-z]+)?\b", text_without_mentions))


def mentioned_bot(message: Message, bot_username: str | None) -> bool:
    """Detect both @username mentions and Telegram text mentions."""

    if not message.text or not bot_username:
        return False

    username = bot_username.lower().lstrip("@")
    for entity in message.entities or ():
        if entity.type == "mention":
            mention = message.text[entity.offset : entity.offset + entity.length]
            if mention.lower().lstrip("@").split("@")[0] == username:
                return True
        elif entity.type == "text_mention" and message.from_user:
            # Telegram's text_mention entity points to the mentioned user.
            # The bot user is not necessarily the sender, so compare by id in
            # the handler where the current bot id is available instead.
            continue

    return bool(re.search(rf"(?<!\w)@{re.escape(username)}(?!\w)", message.text, re.I))


def remove_bot_mention(text: str, bot_username: str | None) -> str:
    if not bot_username:
        return text.strip()
    return re.sub(rf"(?<!\w)@{re.escape(bot_username.lstrip('@'))}(?!\w)", "", text, flags=re.I).strip()


def question_trigger(text: str) -> bool:
    """Return True for the explicit ?? assistant-call prefix."""

    return text.lstrip().startswith("??")


def ends_with_period(text: str) -> bool:
    """A trailing period explicitly suppresses the bot response."""

    return text.rstrip().endswith(".")


def contains_excluded_mention(text: str) -> bool:
    """Ignore messages addressed to inline media bots."""

    return bool(re.search(r"(?<!\w)@(?:pic|gif)(?!\w)", text, re.I))


def project_path(filename: str) -> Path:
    configured_path = Path(filename)
    return (
        configured_path
        if configured_path.is_absolute()
        else Path(__file__).resolve().parent / configured_path
    )


def timezone_now(timezone_name: str) -> datetime:
    try:
        timezone = ZoneInfo(timezone_name)
    except Exception:
        LOGGER.warning("Timezone data unavailable for %s; using fixed Europe/Moscow UTC+3", timezone_name)
        timezone = dt_timezone(timedelta(hours=3), name="MSK")
    return datetime.now(timezone)


class ApiJournal:
    """Store the exact model messages and generated text without API secrets."""

    def __init__(self, filename: str, max_entries: int, enabled: bool) -> None:
        self.path = project_path(filename)
        self.max_entries = max(1, max_entries)
        self.enabled = enabled
        self._lock = threading.Lock()
        LOGGER.info("API journal: %s%s", self.path, " (enabled)" if enabled else " (disabled)")

    @staticmethod
    def _safe(value: object) -> object:
        if isinstance(value, dict):
            return {str(key): ApiJournal._safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [ApiJournal._safe(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def record(
        self,
        *,
        provider: str,
        model: str,
        messages: object,
        response: object,
        kind: str,
        chat_id: int | None,
    ) -> None:
        if not self.enabled:
            return
        entry = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "provider": provider,
            "model": model,
            "kind": kind,
            "chat_id": chat_id,
            "messages": self._safe(messages),
            "response": self._safe(response),
        }
        serialized = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            try:
                existing = (
                    self.path.read_text(encoding="utf-8").splitlines()
                    if self.path.exists()
                    else []
                )
                existing.append(serialized)
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(
                    "\n".join(existing[-self.max_entries :]) + "\n",
                    encoding="utf-8",
                )
            except OSError:
                LOGGER.exception("Could not write API journal %s", self.path)

    def recent(self, chat_id: int | None, limit: int) -> list[dict[str, object]]:
        if not self.enabled or not self.path.exists():
            return []
        entries: list[dict[str, object]] = []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            LOGGER.exception("Could not read API journal %s", self.path)
            return []
        for line in reversed(lines):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            if chat_id is not None and item.get("chat_id") != chat_id:
                continue
            entries.append(item)
            if len(entries) >= limit:
                break
        return entries


class MemoryStore:
    """Small local JSON store for chat-wide and per-user facts."""

    def __init__(self, filename: str) -> None:
        # Keep a relative MEMORY_FILE next to bot.py, regardless of the
        # working directory selected by PyCharm or a terminal.
        self.path = project_path(filename)
        self.data: dict[str, object] = {
            "version": 1,
            "chats": {},
            "users": {},
            "profiles": {},
            "chat_types": {},
        }
        LOGGER.info("Memory file: %s", self.path)
        self._load()
        if not self.path.exists():
            self._save()

    def _load(self) -> None:
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.data.update(loaded)
                self._normalize()
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Could not load memory file %s; starting empty", self.path)

    @classmethod
    def _normalize_scope(cls, scope: dict[str, object]) -> None:
        """Add immutable/mutable buckets and migrate the old flat facts list."""

        immutable = scope.setdefault("immutable", {})
        mutable = scope.setdefault("mutable", {})
        if not isinstance(immutable, dict):
            scope["immutable"] = immutable = {}
        if not isinstance(mutable, dict):
            scope["mutable"] = mutable = {}
        if not isinstance(immutable.get("facts"), list):
            immutable["facts"] = []
        if not isinstance(mutable.get("facts"), list):
            mutable["facts"] = []

        # Facts written by the previous version were model-managed, so they
        # belong in mutable rather than becoming protected facts.
        legacy_facts = scope.pop("facts", None)
        if isinstance(legacy_facts, list):
            cls._add_facts(mutable, legacy_facts)

    def _normalize(self) -> None:
        self.data["version"] = 1
        chats = self.data.setdefault("chats", {})
        users = self.data.setdefault("users", {})
        profiles = self.data.setdefault("profiles", {})
        chat_types = self.data.setdefault("chat_types", {})
        if not isinstance(chats, dict):
            self.data["chats"] = chats = {}
        if not isinstance(users, dict):
            self.data["users"] = users = {}
        if not isinstance(profiles, dict):
            self.data["profiles"] = profiles = {}
        if not isinstance(chat_types, dict):
            self.data["chat_types"] = chat_types = {}
        for chat in chats.values():
            if isinstance(chat, dict):
                self._normalize_scope(chat)
        for chat_users in users.values():
            if not isinstance(chat_users, dict):
                continue
            for user in chat_users.values():
                if isinstance(user, dict):
                    self._normalize_scope(user)
        for profile in profiles.values():
            if isinstance(profile, dict):
                self._normalize_scope(profile)
        self._backfill_shared_profiles()

    @staticmethod
    def _looks_private_chat(chat_id: str) -> bool:
        """Telegram private chat IDs are positive; group/channel IDs are negative."""

        try:
            return int(chat_id) > 0
        except (TypeError, ValueError):
            return False

    def is_private_chat(self, chat_id: int) -> bool:
        chat_types = self.data.get("chat_types", {})
        chat_type = chat_types.get(str(chat_id)) if isinstance(chat_types, dict) else None
        if chat_type == "private":
            return True
        if chat_type in {"group", "supergroup", "channel"}:
            return False
        return self._looks_private_chat(str(chat_id))

    def set_chat_type(self, chat_id: int, chat_type: str | None) -> None:
        if chat_type not in {"private", "group", "supergroup", "channel"}:
            return
        chat_types = self.data.setdefault("chat_types", {})
        if not isinstance(chat_types, dict):
            self.data["chat_types"] = chat_types = {}
        if chat_types.get(str(chat_id)) != chat_type:
            chat_types[str(chat_id)] = chat_type
            self._save()

    def _backfill_shared_profiles(self) -> None:
        """Migrate old per-chat group facts into the cross-chat profile index."""

        profiles = self.data.get("profiles", {})
        users = self.data.get("users", {})
        if not isinstance(profiles, dict) or not isinstance(users, dict):
            return
        for stored_chat_id, chat_users in users.items():
            if self.is_private_chat(int(stored_chat_id)) or not isinstance(chat_users, dict):
                continue
            for stored_user_id, user in chat_users.items():
                if not isinstance(user, dict):
                    continue
                profile = profiles.setdefault(str(stored_user_id), {})
                if not isinstance(profile, dict):
                    continue
                self._normalize_scope(profile)
                if not profile.get("name") and user.get("name"):
                    profile["name"] = str(user["name"]).strip()
                self._add_facts(profile["mutable"], self._facts(user, "mutable"))
                self._add_facts(profile["immutable"], self._facts(user, "immutable"))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _add_facts(target: dict[str, object], facts: object, limit: int = 30) -> None:
        if not isinstance(facts, list):
            return
        existing = target.setdefault("facts", [])
        if not isinstance(existing, list):
            existing = []
            target["facts"] = existing
        for fact in facts:
            if isinstance(fact, str):
                clean = " ".join(fact.split())[:300]
                if clean and clean not in existing:
                    existing.append(clean)
        del existing[:-limit]

    def _sync_shared_profile(
        self,
        user_id: str,
        name: object,
        facts: object,
        immutable_facts: object = None,
    ) -> None:
        profiles = self.data.setdefault("profiles", {})
        if not isinstance(profiles, dict):
            self.data["profiles"] = profiles = {}
        profile = profiles.setdefault(str(user_id), {})
        if not isinstance(profile, dict):
            return
        self._normalize_scope(profile)
        if not profile.get("name") and name:
            profile["name"] = str(name).strip()
        self._add_facts(profile["mutable"], facts)
        self._add_facts(profile["immutable"], immutable_facts)

    def merge(
        self,
        chat_id: int,
        user_id: int,
        user_name: str,
        chat_facts: object,
        user_facts: object,
    ) -> None:
        chats = self.data.setdefault("chats", {})
        users = self.data.setdefault("users", {})
        shared_chat = not self.is_private_chat(chat_id)
        chat = chats.setdefault(str(chat_id), {})
        if isinstance(chat, dict):
            self._normalize_scope(chat)
            mutable_chat = chat["mutable"]
            if isinstance(mutable_chat, dict):
                self._add_facts(mutable_chat, chat_facts)
        chat_users = users.setdefault(str(chat_id), {})
        if isinstance(chat_users, dict):
            extracted_users = user_facts if isinstance(user_facts, dict) else {}
            if extracted_users and all(isinstance(value, dict) for value in extracted_users.values()):
                for extracted_id, info in extracted_users.items():
                    user = chat_users.setdefault(str(extracted_id), {})
                    if isinstance(user, dict) and isinstance(info, dict):
                        self._normalize_scope(user)
                        # A manually configured name in memory.json is stable;
                        # the model may only fill an empty name.
                        if not user.get("name"):
                            user["name"] = info.get("name") or ""
                        mutable_user = user["mutable"]
                        if isinstance(mutable_user, dict):
                            self._add_facts(mutable_user, info.get("facts"))
                        if shared_chat:
                            self._sync_shared_profile(
                                str(extracted_id),
                                user.get("name") or info.get("name"),
                                info.get("facts"),
                            )
            else:
                user = chat_users.setdefault(str(user_id), {"name": user_name})
                if isinstance(user, dict):
                    self._normalize_scope(user)
                    user["name"] = user_name or user.get("name", "")
                    mutable_user = user["mutable"]
                    if isinstance(mutable_user, dict):
                        self._add_facts(mutable_user, user_facts)
                    if shared_chat:
                        self._sync_shared_profile(str(user_id), user_name, user_facts)
        self._save()

    def user_name(self, chat_id: int, user_id: int) -> str | None:
        users = self.data.get("users", {})
        chat_users = users.get(str(chat_id), {}) if isinstance(users, dict) else {}
        user = chat_users.get(str(user_id), {}) if isinstance(chat_users, dict) else {}
        profiles = self.data.get("profiles", {})
        profile = profiles.get(str(user_id), {}) if isinstance(profiles, dict) else {}
        if self.is_private_chat(chat_id) and isinstance(profile, dict):
            name = profile.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        if isinstance(user, dict):
            name = user.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        if isinstance(profile, dict):
            name = profile.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        return None

    def remember_user(
        self, chat_id: int, user_id: int, user_name: str, chat_type: str | None
    ) -> None:
        """Remember a Telegram name locally and in shared profiles only for public chats."""

        users = self.data.setdefault("users", {})
        if not isinstance(users, dict):
            self.data["users"] = users = {}
        chat_users = users.setdefault(str(chat_id), {})
        if not isinstance(chat_users, dict):
            self.data["users"][str(chat_id)] = chat_users = {}
        user = chat_users.setdefault(str(user_id), {})
        if not isinstance(user, dict):
            return
        self._normalize_scope(user)
        changed = False
        if user_name and user.get("name") != user_name:
            user["name"] = user_name
            changed = True
        self.set_chat_type(chat_id, chat_type)
        if chat_type in {"group", "supergroup", "channel"}:
            profiles = self.data.setdefault("profiles", {})
            if not isinstance(profiles, dict):
                self.data["profiles"] = profiles = {}
            profile = profiles.setdefault(str(user_id), {})
            if isinstance(profile, dict):
                self._normalize_scope(profile)
                if user_name and not profile.get("name"):
                    profile["name"] = user_name
                    changed = True
        if changed:
            self._save()

    @staticmethod
    def _facts(scope: object, bucket: str) -> list[str]:
        if not isinstance(scope, dict):
            return []
        values = scope.get(bucket, {})
        if not isinstance(values, dict) or not isinstance(values.get("facts"), list):
            return []
        return [fact for fact in values["facts"] if isinstance(fact, str)]

    def _scope_context(self, scope: object, label: str) -> list[str]:
        blocks: list[str] = []
        immutable = self._facts(scope, "immutable")
        mutable = self._facts(scope, "mutable")
        if immutable:
            blocks.append(f"{label} immutable facts:\n- " + "\n- ".join(immutable))
        if mutable:
            blocks.append(f"{label} mutable facts:\n- " + "\n- ".join(mutable))
        return blocks

    def personal_context(self, chat_id: int, user_id: int) -> str:
        users = self.data.get("users", {})
        chat_users = users.get(str(chat_id), {}) if isinstance(users, dict) else {}
        user = chat_users.get(str(user_id), {}) if isinstance(chat_users, dict) else {}
        blocks: list[str] = []
        if isinstance(user, dict):
            name = user.get("name") or str(user_id)
            blocks.extend(self._scope_context(user, f"{name} personal in this chat"))
        profiles = self.data.get("profiles", {})
        profile = profiles.get(str(user_id), {}) if isinstance(profiles, dict) else {}
        if isinstance(profile, dict):
            name = profile.get("name") or str(user_id)
            blocks.extend(self._scope_context(profile, f"{name} shared profile"))
        return "\n\n".join(blocks)

    def context(self, chat_id: int, exclude_user_id: int | None = None) -> str:
        chats = self.data.get("chats", {})
        users = self.data.get("users", {})
        blocks: list[str] = []
        chat = chats.get(str(chat_id), {}) if isinstance(chats, dict) else {}
        blocks.extend(self._scope_context(chat, "Chat"))
        chat_users = users.get(str(chat_id), {}) if isinstance(users, dict) else {}
        if isinstance(chat_users, dict):
            for stored_user_id, user in chat_users.items():
                if exclude_user_id is not None and str(stored_user_id) == str(exclude_user_id):
                    continue
                if not isinstance(user, dict):
                    continue
                name = user.get("name") or stored_user_id
                blocks.extend(self._scope_context(user, f"{name} personal"))
                profiles = self.data.get("profiles", {})
                profile = profiles.get(str(stored_user_id), {}) if isinstance(profiles, dict) else {}
                if isinstance(profile, dict):
                    blocks.extend(self._scope_context(profile, f"{name} shared profile"))
        return "\n\n".join(blocks)

    def memory_snapshot(self, chat_id: int) -> str:
        """Return the complete current memory for the memory-rewrite request."""

        chats = self.data.get("chats", {})
        users = self.data.get("users", {})
        chat = chats.get(str(chat_id), {}) if isinstance(chats, dict) else {}
        chat_users = users.get(str(chat_id), {}) if isinstance(users, dict) else {}
        lines: list[str] = []

        if isinstance(chat, dict):
            lines.append("CHAT MEMORY:")
            for bucket in ("immutable", "mutable"):
                for fact in self._facts(chat, bucket):
                    lines.append(f"CHAT_{bucket.upper()}_FACT: {fact}")

        if isinstance(chat_users, dict):
            lines.append("USER MEMORY:")
            for stored_user_id, user in chat_users.items():
                if not isinstance(user, dict):
                    continue
                name = str(user.get("name") or "").strip()
                lines.append(f"USER: {stored_user_id}|{name}")
                for bucket in ("immutable", "mutable"):
                    for fact in self._facts(user, bucket):
                        lines.append(f"USER_{bucket.upper()}_FACT: {fact}")

            profiles = self.data.get("profiles", {})
            if isinstance(profiles, dict):
                lines.append("SHARED USER PROFILES:")
                for stored_user_id in chat_users:
                    profile = profiles.get(str(stored_user_id), {})
                    if not isinstance(profile, dict):
                        continue
                    name = str(profile.get("name") or "").strip()
                    lines.append(f"PROFILE: {stored_user_id}|{name}")
                    for bucket in ("immutable", "mutable"):
                        for fact in self._facts(profile, bucket):
                            lines.append(f"PROFILE_{bucket.upper()}_FACT: {fact}")

        return "\n".join(lines) if lines else "EMPTY"

    def rewrite_from_extracted(self, chat_id: int, extracted: dict[str, object]) -> None:
        """Rewrite model-managed memory while preserving facts omitted by the model."""

        chats = self.data.setdefault("chats", {})
        users = self.data.setdefault("users", {})
        chat = chats.setdefault(str(chat_id), {})
        if isinstance(chat, dict):
            self._normalize_scope(chat)
            mutable_chat = chat["mutable"]
            if isinstance(mutable_chat, dict):
                old_facts = self._facts(chat, "mutable")
                new_facts = extracted.get("chat_facts", [])
                mutable_chat["facts"] = []
                self._add_facts(mutable_chat, old_facts)
                self._add_facts(mutable_chat, new_facts)

        chat_users = users.setdefault(str(chat_id), {})
        extracted_users = extracted.get("users", {})
        if not isinstance(chat_users, dict) or not isinstance(extracted_users, dict):
            self._save()
            return

        # Existing users and their facts remain even if the model omitted them.
        for stored_user_id, user in list(chat_users.items()):
            if not isinstance(user, dict):
                continue
            self._normalize_scope(user)
            old_facts = self._facts(user, "mutable")
            info = extracted_users.get(str(stored_user_id), {})
            if not isinstance(info, dict):
                info = {}
            if not user.get("name") and info.get("name"):
                user["name"] = str(info["name"]).strip()
            mutable_user = user["mutable"]
            if isinstance(mutable_user, dict):
                mutable_user["facts"] = []
                self._add_facts(mutable_user, old_facts)
                self._add_facts(mutable_user, info.get("facts"))

        for extracted_id, info in extracted_users.items():
            if not isinstance(info, dict):
                continue
            user = chat_users.setdefault(str(extracted_id), {})
            if not isinstance(user, dict):
                continue
            self._normalize_scope(user)
            if not user.get("name"):
                user["name"] = str(info.get("name") or "").strip()
            mutable_user = user["mutable"]
            if isinstance(mutable_user, dict):
                self._add_facts(mutable_user, info.get("facts"))

        # Public chat facts become part of the cross-chat profile index.
        # Private-chat facts deliberately stay inside that private chat.
        if not self.is_private_chat(chat_id):
            for stored_user_id, user in chat_users.items():
                if not isinstance(user, dict):
                    continue
                self._sync_shared_profile(
                    str(stored_user_id),
                    user.get("name"),
                    self._facts(user, "mutable"),
                    self._facts(user, "immutable"),
                )

        self._save()


class ReminderStore:
    """Persistent one-shot reminder tasks."""

    def __init__(self, filename: str) -> None:
        self.path = project_path(filename)
        self.data: dict[str, object] = {"version": 1, "reminders": []}
        self._load()
        if not self.path.exists():
            self._save()
        LOGGER.info("Reminders file: %s", self.path)

    def _load(self) -> None:
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.data.update(loaded)
            reminders = self.data.get("reminders")
            if not isinstance(reminders, list):
                self.data["reminders"] = []
            else:
                # A process killed during delivery must not lose a reminder.
                kept: list[object] = []
                recovered = False
                for reminder in reminders:
                    if isinstance(reminder, dict) and reminder.get("status") == "completed":
                        recovered = True
                        continue
                    if isinstance(reminder, dict) and reminder.get("status") == "processing":
                        reminder["status"] = "active"
                        recovered = True
                    kept.append(reminder)
                self.data["reminders"] = kept
                if recovered:
                    self._save()
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Could not load reminders file %s; starting empty", self.path)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add(
        self,
        chat_id: int,
        user_id: int,
        user_name: str,
        due_at: datetime,
        text: str,
    ) -> dict[str, object]:
        reminder = {
            "id": uuid4().hex,
            "chat_id": chat_id,
            "user_id": user_id,
            "user_name": user_name,
            "due_at": due_at.isoformat(),
            "text": text,
            "status": "active",
            "created_at": datetime.now(due_at.tzinfo).isoformat(),
        }
        reminders = self.data.setdefault("reminders", [])
        if not isinstance(reminders, list):
            reminders = []
            self.data["reminders"] = reminders
        reminders.append(reminder)
        self._save()
        return reminder

    def claim_due(self, now: datetime) -> list[dict[str, object]]:
        reminders = self.data.get("reminders", [])
        if not isinstance(reminders, list):
            return []
        claimed: list[dict[str, object]] = []
        changed = False
        for reminder in reminders:
            if not isinstance(reminder, dict) or reminder.get("status") != "active":
                continue
            try:
                due_at = datetime.fromisoformat(str(reminder["due_at"]))
            except (KeyError, TypeError, ValueError):
                LOGGER.warning("Ignoring reminder with invalid due_at: %r", reminder)
                continue
            if due_at.tzinfo is None:
                due_at = due_at.replace(tzinfo=now.tzinfo)
            if due_at <= now:
                reminder["status"] = "processing"
                claimed.append(reminder.copy())
                changed = True
                LOGGER.info(
                    "Claiming due reminder %s (scheduled for %s)",
                    reminder.get("id", ""),
                    reminder.get("due_at", ""),
                )
        if changed:
            self._save()
        return claimed

    def remove(self, reminder_id: str) -> None:
        reminders = self.data.get("reminders", [])
        if not isinstance(reminders, list):
            return
        self.data["reminders"] = [
            reminder
            for reminder in reminders
            if not isinstance(reminder, dict) or reminder.get("id") != reminder_id
        ]
        self._save()

    def release(self, reminder_id: str) -> None:
        self._set_status(reminder_id, "active")

    def _set_status(self, reminder_id: str, status: str) -> None:
        reminders = self.data.get("reminders", [])
        if not isinstance(reminders, list):
            return
        for reminder in reminders:
            if isinstance(reminder, dict) and reminder.get("id") == reminder_id:
                reminder["status"] = status
                self._save()
                return


def parse_reminder(text: str, now: datetime) -> tuple[datetime, str] | None:
    match = re.match(r"^\s*/н(?:@[A-Za-z0-9_]+)?\s+(.+?)\s*$", text, re.IGNORECASE)
    if not match:
        return None
    body = match.group(1).strip()

    relative = re.match(
        r"^через\s+(\d+)\s*(минут(?:у|ы)?|час(?:а|ов)?|дн(?:я|ей)?)\s+(.+)$",
        body,
        re.IGNORECASE,
    )
    if relative:
        amount = int(relative.group(1))
        unit = relative.group(2).lower()
        if unit.startswith("мин"):
            delta = timedelta(minutes=amount)
        elif unit.startswith("час"):
            delta = timedelta(hours=amount)
        else:
            delta = timedelta(days=amount)
        return now + delta, relative.group(3).strip()

    day_time = re.match(
        r"^(сегодня|завтра)\s+(\d{1,2}):(\d{2})\s+(.+)$", body, re.IGNORECASE
    )
    if day_time:
        day = now.date() + timedelta(days=1 if day_time.group(1).lower() == "завтра" else 0)
        try:
            due_at = now.replace(
                year=day.year,
                month=day.month,
                day=day.day,
                hour=int(day_time.group(2)),
                minute=int(day_time.group(3)),
                second=0,
                microsecond=0,
            )
        except ValueError:
            return None
        return due_at, day_time.group(4).strip()

    date_time = re.match(
        r"^(\d{1,2})[./-](\d{1,2})(?:[./-](\d{4}))?\s+(\d{1,2}):(\d{2})\s+(.+)$",
        body,
    )
    if date_time:
        year = int(date_time.group(3) or now.year)
        try:
            due_at = now.replace(
                year=year,
                month=int(date_time.group(2)),
                day=int(date_time.group(1)),
                hour=int(date_time.group(4)),
                minute=int(date_time.group(5)),
                second=0,
                microsecond=0,
            )
        except ValueError:
            return None
        return due_at, date_time.group(6).strip()

    clock = re.match(r"^(\d{1,2}):(\d{2})\s+(.+)$", body)
    if clock:
        try:
            due_at = now.replace(
                hour=int(clock.group(1)),
                minute=int(clock.group(2)),
                second=0,
                microsecond=0,
            )
        except ValueError:
            return None
        if due_at <= now:
            due_at += timedelta(days=1)
        return due_at, clock.group(3).strip()
    return None


class ReplyService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.gigachat_client = (
            GigaChat(
                credentials=settings.gigachat_credentials,
                scope=settings.gigachat_scope,
                model=settings.gigachat_model,
                base_url=settings.gigachat_base_url,
                verify_ssl_certs=settings.gigachat_verify_ssl,
                timeout=settings.gigachat_timeout,
                max_retries=settings.gigachat_max_retries,
                retry_backoff_factor=1.0,
            )
            if settings.ai_provider == "gigachat" and settings.gigachat_credentials
            else None
        )
        self.vsegpt_client = (
            OpenAI(
                api_key=settings.vsegpt_api_key,
                base_url=settings.vsegpt_base_url,
                timeout=settings.gigachat_timeout,
                max_retries=0,
            )
            if settings.ai_provider == "vsegpt" and settings.vsegpt_api_key
            else None
        )
        self.history: dict[tuple[int, str], list[dict[str, object]]] = {}
        self.compacted_history: set[tuple[int, str]] = set()
        self.memory = MemoryStore(settings.memory_file)
        self.journal = ApiJournal(
            settings.api_journal_file,
            settings.api_journal_max_entries,
            settings.api_journal_enabled,
        )

    @staticmethod
    def _is_retryable_api_error(error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        return (
            status_code is None
            or status_code in {408, 409, 429}
            or (isinstance(status_code, int) and status_code >= 500)
        )

    @staticmethod
    def _vsegpt_response_text(response: object) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "") if message else ""
        return content if isinstance(content, str) else str(content or "")

    @staticmethod
    def _gigachat_response_text(response: object) -> str:
        messages = getattr(response, "messages", None) or []
        if not messages:
            return ""
        content = getattr(messages[0], "content", "")
        if isinstance(content, str):
            return content
        parts = content or []
        return "".join(str(getattr(part, "text", "") or "") for part in parts)

    async def _gigachat_chat(
        self,
        request: list[dict[str, object]],
        *,
        kind: str,
        chat_id: int | None,
    ):
        response = await asyncio.to_thread(
            self.gigachat_client.chat.create,
            {"model": self.settings.gigachat_model, "messages": request},
        )
        self.journal.record(
            provider="gigachat",
            model=self.settings.gigachat_model,
            kind=kind,
            chat_id=chat_id,
            messages=request,
            response=self._gigachat_response_text(response),
        )
        return response

    async def _vsegpt_chat(
        self,
        *,
        kind: str = "chat",
        chat_id: int | None = None,
        **kwargs: object,
    ):
        """Call VseGPT with bounded retries for network and server failures."""

        attempts = max(1, self.settings.gigachat_max_retries + 1)
        for attempt in range(attempts):
            try:
                response = await asyncio.to_thread(
                    self.vsegpt_client.chat.completions.create,
                    **kwargs,
                )
                self.journal.record(
                    provider="vsegpt",
                    model=str(kwargs.get("model", "")),
                    kind=kind,
                    chat_id=chat_id,
                    messages=kwargs.get("messages", []),
                    response=self._vsegpt_response_text(response),
                )
                return response
            except Exception as error:
                if attempt + 1 >= attempts or not self._is_retryable_api_error(error):
                    raise
                delay = min(8.0, 0.75 * (2**attempt))
                LOGGER.warning(
                    "VseGPT request failed (%s/%s): %s; retrying in %.2fs",
                    attempt + 1,
                    attempts,
                    error,
                    delay,
                )
                await asyncio.sleep(delay)

    def _api_messages(
        self, key: tuple[int, str], messages: list[dict[str, object]]
    ) -> list[dict[str, str]]:
        def format_message(message: dict[str, object]) -> str:
            content = str(message["content"])
            if message.get("role") != "user":
                return content
            name = str(message.get("user_name") or message.get("user_id") or "пользователь")
            name = name.replace('"', "'").strip() or "пользователь"
            return f'Пишет "{name}":\n{content}'

        if key not in self.compacted_history:
            return [
                {"role": str(message["role"]), "content": format_message(message)}
                for message in messages
            ]

        users = [message for message in messages if message.get("role") == "user"]
        assistants = [message for message in messages if message.get("role") == "assistant"]
        if not users:
            return []
        latest_user = users[-1]
        prior_users = users[:-1]
        compacted: list[dict[str, str]] = []
        if prior_users:
            compacted.append(
                {
                    "role": "user",
                    "content": "Earlier user messages are context only; do not answer them:\n"
                    + "\n".join(format_message(item) for item in prior_users),
                }
            )
        if assistants:
            compacted.append(
                
                {
                    "role": "assistant",
                    "content": "Previous assistant replies for context only:\n"
                    + "\n---\n".join(str(item["content"]) for item in assistants[-2:]),
                }
            )
        compacted.append({"role": "user", "content": format_message(latest_user)})
        return compacted

    @staticmethod
    def _facts_from_text(raw: str) -> dict[str, object] | None:
        """Parse the deliberately simple text format returned by the memory call."""

        if not isinstance(raw, str):
            return None
        text = raw.strip()
        if not text or text.upper() == "EMPTY":
            return {"chat_facts": [], "users": {}}

        chat_facts: list[str] = []
        users: dict[str, dict[str, object]] = {}
        for line in text.splitlines():
            line = line.strip().lstrip("-* ")
            if not line:
                continue
            chat_match = re.match(r"CHAT_FACT\s*:\s*(.+)$", line, re.IGNORECASE)
            if chat_match:
                chat_facts.append(chat_match.group(1).strip())
                continue
            user_match = re.match(r"USER_FACT\s*:\s*([^|]+)\|([^|]*)\|(.+)$", line, re.IGNORECASE)
            if user_match:
                user_id, name, fact = (part.strip() for part in user_match.groups())
                if user_id and fact:
                    user = users.setdefault(user_id, {"name": name, "facts": []})
                    facts = user.setdefault("facts", [])
                    if isinstance(facts, list):
                        facts.append(fact)

        if not chat_facts and not users:
            return None
        return {"chat_facts": chat_facts, "users": users}

    async def _extract_memory(
        self,
        chat_id: int,
        messages: list[dict[str, object]],
    ) -> bool:
        if not messages:
            return True
        LOGGER.info(
            "Rewriting memory for chat %s from full context (%s messages)",
            chat_id,
            len(messages),
        )
        transcript_parts: list[str] = []
        for item in messages:
            if item.get("role") == "user":
                name = str(item.get("user_name") or item.get("user_id") or "пользователь")
                name = name.replace('"', "'").strip() or "пользователь"
                transcript_parts.append(
                    f'Пишет "{name}" (ID {item.get("user_id", "")}): {item["content"]}'
                )
            else:
                transcript_parts.append(f'Бот ответил: {item["content"]}')
        transcript = "\n".join(transcript_parts)
        memory_snapshot = self.memory.memory_snapshot(chat_id)
        extractor_system = self.settings.memory_prompt
        try:
            request = [
                {"role": "system", "content": extractor_system},
                {
                    "role": "user",
                    "content": (
                        "CURRENT MEMORY:\n"
                        + memory_snapshot
                        + "\n\nFULL CONVERSATION HISTORY:\n"
                        + transcript
                    ),
                },
            ]
            if self.settings.ai_provider == "gigachat":
                response = await self._gigachat_chat(
                    request,
                    kind="memory",
                    chat_id=chat_id,
                )
                if not response.messages:
                    return False
                parts = response.messages[0].content or []
                raw = parts if isinstance(parts, str) else "".join(part.text or "" for part in parts)
            else:
                response = await self._vsegpt_chat(
                    model=self.settings.vsegpt_model,
                    messages=request,
                    max_tokens=self.settings.memory_max_tokens,
                    temperature=0,
                    kind="memory",
                    chat_id=chat_id,
                )
                if not response.choices:
                    return False
                raw = response.choices[0].message.content or ""
            extracted = self._facts_from_text(raw)
            if extracted is None:
                LOGGER.warning(
                    "Memory extraction returned unparseable text for chat %s: %r",
                    chat_id,
                    str(raw)[:300],
                )
                return False
            self.memory.rewrite_from_extracted(chat_id, extracted)
            LOGGER.info("Full memory rewritten for chat %s", chat_id)
            return True
        except Exception:
            LOGGER.exception("Memory extraction failed")
            return False

    async def _prepare_history(
        self,
        chat_id: int,
        user_id: int,
        user_name: str,
        text: str,
        mode: str,
    ) -> list[dict[str, object]]:
        messages = self.history.setdefault((chat_id, mode), [])
        history_limit = (
            self.settings.max_english_history_messages
            if mode == "english_coach"
            else self.settings.soft_context_limit
        )
        messages.append(
            {"role": "user", "content": text, "user_id": user_id, "user_name": user_name}
        )
        if mode == "english_coach":
            messages[:] = messages[-history_limit:]
            return messages

        hard_compaction_failed = False
        if len(messages) >= self.settings.hard_context_limit:
            LOGGER.info(
                "Hard context limit reached for chat %s, mode %s (%s entries)",
                chat_id,
                mode,
                len(messages),
            )
            keep = min(self.settings.keep_last_messages, len(messages))
            if await self._extract_memory(chat_id, messages):
                messages[:] = messages[-keep:]
                self.compacted_history.discard((chat_id, mode))
                LOGGER.info("Hard context compaction completed for chat %s", chat_id)
            else:
                hard_compaction_failed = True
                LOGGER.warning(
                    "Hard context compaction skipped for chat %s because memory extraction failed",
                    chat_id,
                )

        assistant_messages = [item for item in messages if item.get("role") == "assistant"]
        if not hard_compaction_failed and len(assistant_messages) >= self.settings.soft_context_limit:
            LOGGER.info(
                "Soft context compaction for chat %s, mode %s (%s assistant entries)",
                chat_id,
                mode,
                len(assistant_messages),
            )
            keep_assistant_ids = {id(item) for item in assistant_messages[-2:]}
            messages[:] = [
                item
                for item in messages
                if item.get("role") == "user" or id(item) in keep_assistant_ids
            ]
            self.compacted_history.add((chat_id, mode))
        return messages

    async def reply(
        self,
        chat_id: int,
        user_id: int,
        user_name: str,
        text: str,
        mode: str,
        chat_type: str | None = None,
    ) -> str:
        if self.settings.ai_provider == "gigachat" and not self.gigachat_client:
            return (
                "Я получил сообщение. Чтобы я формировал содержательные ответы, "
                "добавьте GIGACHAT_CREDENTIALS в .env."
            )
        if self.settings.ai_provider == "vsegpt" and not self.vsegpt_client:
            return (
                "Я получил сообщение. Чтобы я формировал содержательные ответы, "
                "добавьте VSEGPT_API_KEY в .env."
            )

        self.memory.set_chat_type(chat_id, chat_type)
        self.memory.remember_user(chat_id, user_id, user_name, chat_type)
        configured_name = self.memory.user_name(chat_id, user_id)
        speaker_name = configured_name or str(user_id)
        messages = await self._prepare_history(chat_id, user_id, speaker_name, text, mode)

        try:
            system_message = self.settings.english_coach_prompt if mode == "english_coach" else self.settings.system_prompt
            request_messages = [
                {"role": "system", "content": system_message},
                *self._api_messages((chat_id, mode), messages),
            ]
            if request_messages and request_messages[-1].get("role") == "user":
                context_parts: list[str] = []
                if mode != "english_coach":
                    personal_context = self.memory.personal_context(chat_id, user_id)
                    chat_context = self.memory.context(chat_id, exclude_user_id=user_id)
                    if personal_context:
                        context_parts.append(
                            "Персональная информация отправителя (контекст):\n"
                            + personal_context
                        )
                    if chat_context:
                        context_parts.append("Остальной контекст (справочно):\n" + chat_context)
                context_parts.append(
                    "Текущее время: "
                    + timezone_now(self.settings.timezone).strftime("%Y-%m-%d %H:%M (%Z)")
                )
                request_messages[-1]["content"] += "\n\n" + "\n\n".join(context_parts)

            if self.settings.ai_provider == "gigachat":
                response = await self._gigachat_chat(
                    request_messages,
                    kind="reply",
                    chat_id=chat_id,
                )
                if not response.messages:
                    raise RuntimeError("GigaChat returned an empty response")
                parts = response.messages[0].content or []
                answer = "".join(part.text or "" for part in parts).strip()
            else:
                response = await self._vsegpt_chat(
                    model=self.settings.vsegpt_model,
                    messages=request_messages,
                    max_tokens=self.settings.vsegpt_max_tokens,
                    kind="reply",
                    chat_id=chat_id,
                )
                if not response.choices:
                    raise RuntimeError("VseGPT returned an empty response")
                answer = (response.choices[0].message.content or "").strip()
        except Exception:
            LOGGER.exception("AI response failed")
            answer = "Не смог обработать сообщение. Попробуйте ещё раз позже."

        if answer:
            messages.append({"role": "assistant", "content": answer})
            if mode == "english_coach":
                messages[:] = messages[-self.settings.max_english_history_messages :]
            return answer
        return "Модель вернула пустой ответ. Попробуйте ещё раз."

    @staticmethod
    def _scheduled_from_text(
        raw: str, now: datetime, fallback_text: str
    ) -> tuple[datetime, str] | None:
        if not isinstance(raw, str):
            return None
        time_match = re.search(
            r"(?:^|\n)\s*TIME\s*:\s*(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}|NONE)",
            raw,
            re.IGNORECASE,
        )
        if not time_match or time_match.group(1).upper() == "NONE":
            return None
        try:
            due_at = datetime.strptime(time_match.group(1), "%Y-%m-%d %H:%M").replace(
                tzinfo=now.tzinfo
            )
        except ValueError:
            return None
        text_match = re.search(r"(?:^|\n)\s*TEXT\s*:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
        reminder_text = text_match.group(1).strip() if text_match else fallback_text
        return (due_at, reminder_text or fallback_text) if reminder_text else None

    async def resolve_reminder(
        self, command_text: str, now: datetime, chat_id: int | None = None
    ) -> tuple[datetime, str] | None:
        """Let the model understand natural time, with a local parser fallback."""

        fallback = parse_reminder(command_text, now)
        client_available = (
            self.settings.ai_provider == "vsegpt" and self.vsegpt_client
        ) or (self.settings.ai_provider == "gigachat" and self.gigachat_client)
        if not client_available:
            return fallback

        schedule_prompt = (
            self.settings.reminder_schedule_prompt
            + "\nCurrent local time: "
            + now.strftime("%Y-%m-%d %H:%M")
            + "\nTimezone: "
            + self.settings.timezone
        )
        request = [
            {"role": "system", "content": schedule_prompt},
            {"role": "user", "content": command_text},
        ]
        try:
            if self.settings.ai_provider == "gigachat":
                response = await self._gigachat_chat(
                    request,
                    kind="reminder_schedule",
                    chat_id=chat_id,
                )
                if not response.messages:
                    return fallback
                parts = response.messages[0].content or []
                raw = parts if isinstance(parts, str) else "".join(part.text or "" for part in parts)
            else:
                response = await self._vsegpt_chat(
                    model=self.settings.vsegpt_model,
                    messages=request,
                    max_tokens=min(self.settings.vsegpt_max_tokens, 100),
                    kind="reminder_schedule",
                    chat_id=chat_id,
                )
                if not response.choices:
                    return fallback
                raw = response.choices[0].message.content or ""
            command_body = re.sub(
                r"^\s*/н(?:@[A-Za-z0-9_]+)?\s*", "", command_text, flags=re.IGNORECASE
            ).strip()
            return self._scheduled_from_text(raw, now, command_body) or fallback
        except Exception:
            LOGGER.exception("Reminder schedule interpretation failed")
            return fallback

    async def reminder_text(self, text: str, chat_id: int | None = None) -> str | None:
        """Generate a reminder from only its original text, without chat context."""

        if self.settings.ai_provider == "gigachat" and not self.gigachat_client:
            return None
        if self.settings.ai_provider == "vsegpt" and not self.vsegpt_client:
            return None
        try:
            request = [
                {"role": "system", "content": self.settings.reminder_prompt},
                {"role": "user", "content": text},
            ]
            if self.settings.ai_provider == "gigachat":
                response = await self._gigachat_chat(
                    request,
                    kind="reminder",
                    chat_id=chat_id,
                )
                if not response.messages:
                    return None
                parts = response.messages[0].content or []
                answer = parts if isinstance(parts, str) else "".join(part.text or "" for part in parts)
            else:
                response = await self._vsegpt_chat(
                    model=self.settings.vsegpt_model,
                    messages=request,
                    max_tokens=min(self.settings.vsegpt_max_tokens, 80),
                    kind="reminder",
                    chat_id=chat_id,
                )
                if not response.choices:
                    return None
                answer = response.choices[0].message.content or ""
            return answer.strip() or None
        except Exception:
            LOGGER.exception("Reminder generation failed")
            return None

    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        speaker_name: str | None = None,
        chat_id: int | None = None,
    ) -> str:
        if not self.vsegpt_client:
            return "Для анализа изображений выберите AI_PROVIDER=vsegpt и добавьте VSEGPT_API_KEY."
        data_url = "data:image/jpeg;base64," + b64encode(image_bytes).decode("ascii")
        try:
            response = await self._vsegpt_chat(
                model=self.settings.vsegpt_vision_model,
                messages=[
                    {"role": "system", "content": self.settings.vision_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f'Контекст отправителя: пишет "{speaker_name}".\n{prompt}'
                                    if speaker_name
                                    else prompt
                                ),
                            },
                            {"type": "image_url", "image_url": data_url},
                        ],
                    }
                ],
                max_tokens=self.settings.vsegpt_max_tokens,
                kind="vision",
                chat_id=chat_id,
            )
            if not response.choices:
                raise RuntimeError("VseGPT returned an empty vision response")
            return (response.choices[0].message.content or "Не смог описать изображение.").strip()
        except Exception:
            LOGGER.exception("Image analysis failed")
            return "Не смог проанализировать изображение. Проверьте баланс и доступность vision-модели."


async def reminder_loop(application: Application) -> None:
    store: ReminderStore = application.bot_data["reminder_store"]
    service: ReplyService = application.bot_data["reply_service"]
    settings = service.settings
    LOGGER.info("Reminder scheduler started")
    try:
        while True:
            now = timezone_now(settings.timezone)
            for reminder in store.claim_due(now):
                reminder_id = str(reminder.get("id", ""))
                try:
                    generated = await service.reminder_text(
                        str(reminder.get("text", "")),
                        int(reminder["chat_id"]),
                    )
                    if not generated:
                        raise RuntimeError("Reminder model returned an empty response")
                    await application.bot.send_message(
                        chat_id=int(reminder["chat_id"]),
                        text=generated,
                    )
                    store.remove(reminder_id)
                    LOGGER.info("Reminder delivered: %s", reminder_id)
                except Exception:
                    LOGGER.exception("Reminder delivery failed: %s", reminder_id)
                    store.release(reminder_id)
            await asyncio.sleep(settings.reminder_poll_seconds)
    except asyncio.CancelledError:
        LOGGER.info("Reminder scheduler stopped")
        raise


async def start_background_tasks(application: Application) -> None:
    application.bot_data["reminder_task"] = asyncio.create_task(reminder_loop(application))


async def stop_background_tasks(application: Application) -> None:
    task = application.bot_data.get("reminder_task")
    if not isinstance(task, asyncio.Task):
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "Привет! Упомяните меня через @, начните сообщение с ?? или отправьте "
        "сообщение на английском языке — и я отвечу. Для анализа фото добавьте "
        "подпись ?? ваш вопрос. Для напоминания используйте /н. Подробнее: /help"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "Возможности бота:\n"
        "@бот текст — обычный ответ в чате.\n"
        "?? текст — обычный режим без упоминания.\n"
        "Английский текст без обращения — проверка и исправление английского.\n"
        "Фото с подписью ?? вопрос или /analyze вопрос — анализ изображения.\n"
        "/н свободное описание времени и задачи — напоминание. Например: "
        "/н напомни завтра вечером купить продукты.\n"
        "/start — краткое описание.\n"
        "/help — эта справка.\n\n"
        "/log [N] — последние запросы к API в этом чате.\n\n"
        "Память и напоминания сохраняются локально в файлах приложения."
    )


def _journal_value(value: object, limit: int = 1800) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) > limit:
        return text[:limit] + "\n...[обрезано в Telegram, полный текст в файле журнала]"
    return text


async def api_log_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    chat = update.effective_chat
    if not message or not chat:
        return
    service: ReplyService = context.application.bot_data["reply_service"]
    if not service.journal.enabled:
        await message.reply_text("Журнал API отключён в config.json.")
        return
    try:
        limit = max(1, min(10, int(context.args[0]))) if context.args else 5
    except ValueError:
        limit = 5
    entries = service.journal.recent(chat.id, limit)
    if not entries:
        await message.reply_text("В журнале этого чата пока нет запросов к модели.")
        return

    chunks: list[str] = []
    for number, entry in enumerate(reversed(entries), 1):
        chunks.append(
            f"Запрос {number} | {entry.get('timestamp', '')} | "
            f"{entry.get('kind', 'chat')} | {entry.get('provider', '')}/{entry.get('model', '')}\n"
            "Модель получила:\n"
            + _journal_value(entry.get("messages"))
            + "\n\nМодель вывела:\n"
            + _journal_value(entry.get("response"))
        )
    text = "\n\n====================\n\n".join(chunks)
    while text:
        part, text = text[:3800], text[3800:]
        await message.reply_text(part)


async def handle_reminder(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    chat = update.effective_chat
    if not message or not message.text or not chat:
        return
    if message.from_user and message.from_user.is_bot:
        return

    service: ReplyService = context.application.bot_data["reply_service"]
    now = timezone_now(service.settings.timezone)
    parsed = await service.resolve_reminder(message.text, now, chat.id)
    if parsed is None:
        await message.reply_text(
            "Не смог понять время. Напиши, например: /н напомни завтра вечером купить продукты"
        )
        return
    due_at, reminder_text = parsed
    if not reminder_text:
        await message.reply_text(
            "После времени нужно написать текст напоминания. Например: /н 18:30 позвонить другу"
        )
        return

    sender = message.from_user
    user_id = sender.id if sender else 0
    if sender:
        service.memory.remember_user(
            chat.id,
            user_id,
            sender.full_name or sender.username or str(user_id),
            chat.type,
        )
    speaker_name = service.memory.user_name(chat.id, user_id) or str(user_id)
    reminder_store: ReminderStore = context.application.bot_data["reminder_store"]
    reminder = reminder_store.add(chat.id, user_id, speaker_name, due_at, reminder_text)
    await message.reply_text(
        "Напоминание поставлено на "
        + due_at.strftime("%d.%m.%Y %H:%M")
        + "."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message or not message.text or not update.effective_chat:
        return
    if message.from_user and message.from_user.is_bot:
        return
    if ends_with_period(message.text):
        return
    if contains_excluded_mention(message.text):
        return

    bot = context.bot
    bot_username = bot.username
    is_mentioned = mentioned_bot(message, bot_username)

    # Also support Telegram's rich text_mention entity, which has no @text.
    if not is_mentioned:
        for entity in message.entities or ():
            if entity.type == "text_mention" and entity.user and entity.user.id == bot.id:
                is_mentioned = True
                break

    is_english = contains_english(message.text)
    is_question_trigger = question_trigger(message.text)
    if not is_mentioned and not is_question_trigger and not is_english:
        return

    prompt = remove_bot_mention(message.text, bot_username)
    if not prompt:
        prompt = "Расскажи, чем ты можешь помочь."

    service: ReplyService = context.application.bot_data["reply_service"]
    # An explicit @ mention or ?? prefix selects the general assistant mode.
    # English coaching is used only for an unad dressed English message.
    mode = "general" if is_mentioned or is_question_trigger else "english_coach"
    sender = message.from_user
    user_id = sender.id if sender else 0
    user_name = (sender.full_name or sender.username or str(user_id)) if sender else "unknown"
    answer = await service.reply(
        update.effective_chat.id,
        user_id,
        user_name,
        prompt,
        mode,
        update.effective_chat.type,
    )
    await message.reply_text(answer)


def photo_analysis_prompt(caption: str) -> str | None:
    match = re.match(r"^\s*(?:\?\?|/analyze(?:@[A-Za-z0-9_]+)?)\s*(.*)", caption, re.I)
    if not match:
        return None
    return match.group(1).strip() or "Кратко опиши изображение."


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message or not message.photo or (message.from_user and message.from_user.is_bot):
        return
    caption = message.caption or ""
    prompt = photo_analysis_prompt(caption)
    if prompt is None:
        return

    service: ReplyService = context.application.bot_data["reply_service"]
    if service.settings.ai_provider != "vsegpt":
        await message.reply_text("Анализ изображений доступен при AI_PROVIDER=vsegpt.")
        return

    photo = message.photo[-1]
    telegram_file = await context.bot.get_file(photo.file_id)
    image_bytes = bytes(await telegram_file.download_as_bytearray())

    sender = message.from_user
    service.memory.set_chat_type(update.effective_chat.id, update.effective_chat.type)
    speaker_name = (
        service.memory.user_name(update.effective_chat.id, sender.id)
        if sender
        else None
    ) or (str(sender.id) if sender else "пользователь")
    answer = await service.analyze_image(
        image_bytes,
        prompt,
        speaker_name,
        update.effective_chat.id,
    )
    await message.reply_text(answer)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    LOGGER.error("Unhandled Telegram error", exc_info=context.error)


def build_application(settings: Settings) -> Application:
    application = (
        ApplicationBuilder()
        .token(settings.telegram_token)
        .connect_timeout(settings.telegram_timeout)
        .read_timeout(settings.telegram_timeout)
        .write_timeout(settings.telegram_timeout)
        .pool_timeout(settings.telegram_timeout)
        .get_updates_connect_timeout(settings.telegram_timeout)
        .get_updates_read_timeout(settings.telegram_timeout + 10)
        .get_updates_write_timeout(settings.telegram_timeout)
        .get_updates_pool_timeout(settings.telegram_timeout)
        .post_init(start_background_tasks)
        .post_shutdown(stop_background_tasks)
        .build()
    )
    application.bot_data["reply_service"] = ReplyService(settings)
    application.bot_data["reminder_store"] = ReminderStore(settings.reminders_file)
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler(["log", "journal"], api_log_command))
    application.add_handler(
        MessageHandler(
            filters.TEXT & filters.Regex(r"^\s*/н(?:@[A-Za-z0-9_]+)?(?:\s+.*)?$"),
            handle_reminder,
        )
    )
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    return application


def main() -> None:
    load_dotenv()
    settings = Settings.from_env()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=settings.log_level,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    if settings.ai_provider == "gigachat" and not settings.gigachat_credentials:
        LOGGER.warning("GIGACHAT_CREDENTIALS is not set; fallback replies will be used")
    elif settings.ai_provider == "vsegpt" and not settings.vsegpt_api_key:
        LOGGER.warning("VSEGPT_API_KEY is not set; fallback replies will be used")
    else:
        model = settings.gigachat_model if settings.ai_provider == "gigachat" else settings.vsegpt_model
        LOGGER.info("Using %s model: %s", settings.ai_provider, model)

    # Some python-telegram-bot releases still expect an event loop before polling.
    asyncio.set_event_loop(asyncio.new_event_loop())

    LOGGER.info("Starting Telegram bot")
    # Keep retrying while the network/VPN is coming up during Windows startup.
    build_application(settings).run_polling(
        allowed_updates=Update.ALL_TYPES,
        bootstrap_retries=-1,
    )


if __name__ == "__main__":
    main()
