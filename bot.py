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
from dataclasses import dataclass

from dotenv import load_dotenv
from gigachat import GigaChat
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


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    gigachat_credentials: str | None
    gigachat_scope: str
    gigachat_model: str
    gigachat_base_url: str
    gigachat_verify_ssl: bool
    gigachat_timeout: float
    gigachat_max_retries: int
    telegram_timeout: float
    system_prompt: str
    english_coach_prompt: str
    max_history_messages: int

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

        return cls(
            telegram_token=token,
            # OPENAI_* fallbacks keep compatibility with the first .env file,
            # where the GigaChat authorization key was entered under that name.
            gigachat_credentials=(
                os.getenv("GIGACHAT_CREDENTIALS", "").strip()
                or os.getenv("OPENAI_API_KEY", "").strip()
                or None
            ),
            gigachat_scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip(),
            gigachat_model=(
                os.getenv("GIGACHAT_MODEL", "").strip()
                or os.getenv("OPENAI_MODEL", "GigaChat-2-Pro").strip()
            ),
            gigachat_base_url=os.getenv("GIGACHAT_BASE_URL", "https://api.giga.chat/v1").strip(),
            gigachat_verify_ssl=os.getenv("GIGACHAT_VERIFY_SSL_CERTS", "false").lower() == "true",
            gigachat_timeout=float(os.getenv("GIGACHAT_TIMEOUT", "60")),
            gigachat_max_retries=max(0, int(os.getenv("GIGACHAT_MAX_RETRIES", "3"))),
            telegram_timeout=float(os.getenv("TELEGRAM_TIMEOUT", "30")),
            system_prompt=os.getenv(
                "SYSTEM_PROMPT",
                "Ты полезный ассистент в Telegram-чате. "
                "Отвечай кратко и по делу на русском языке, если пользователь "
                "не попросил другой язык.",
            ).strip(),
            english_coach_prompt=os.getenv(
                "ENGLISH_COACH_PROMPT",
                "Ты носитель английского языка и опытный преподаватель. "
                "Проверь английское сообщение пользователя. Если есть ошибки, "
                "покажи исправленный вариант и кратко объясни каждую ошибку. "
                "Если текст корректен, покажи более естественный вариант. "
                "Не меняй смысл. Затем при необходимости добавь 1–3 коротких "
                "совета. Не пиши приветствия, похвалу или лишние объяснения. "
                "Примеры оставляй на английском.",
            ).strip(),
            max_history_messages=max(1, int(os.getenv("MAX_HISTORY_MESSAGES", "8"))),
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


class ReplyService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = (
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
            if settings.gigachat_credentials
            else None
        )
        self.history: dict[tuple[int, str], list[dict[str, str]]] = {}

    async def reply(self, chat_id: int, text: str, mode: str) -> str:
        if not self.client:
            return (
                "Я получил сообщение. Чтобы я формировал содержательные ответы, "
                "добавьте GIGACHAT_CREDENTIALS в .env."
            )

        messages = self.history.setdefault((chat_id, mode), [])
        messages.append({"role": "user", "content": text})
        messages[:] = messages[-self.settings.max_history_messages :]

        try:
            response = await asyncio.to_thread(
                self.client.chat.create,
                {
                    "model": self.settings.gigachat_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                self.settings.english_coach_prompt
                                + "\n\nКритичное правило: отвечай только готовым "
                                "улучшенным вариантом текста. После варианта дай "
                                "1–3 коротких полезных совета. Не добавляй "
                                "приветствия, оценки или воду."
                                if mode == "english_coach"
                                else self.settings.system_prompt
                            ),
                        },
                        *messages,
                    ],
                },
            )
            if not response.messages:
                raise RuntimeError("GigaChat returned an empty response")
            parts = response.messages[0].content or []
            answer = "".join(part.text or "" for part in parts).strip()
        except Exception:
            LOGGER.exception("AI response failed")
            answer = "Не смог обработать сообщение. Попробуйте ещё раз позже."

        if answer:
            messages.append({"role": "assistant", "content": answer})
            messages[:] = messages[-self.settings.max_history_messages :]
            return answer
        return "Не смог сформировать ответ."


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "Привет! Упомяните меня через @, начните сообщение с ?? или отправьте "
        "сообщение на английском языке — и я отвечу."
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
    # English coaching is used only for an unaddressed English message.
    mode = "general" if is_mentioned or is_question_trigger else "english_coach"
    answer = await service.reply(update.effective_chat.id, prompt, mode)
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
        .build()
    )
    application.bot_data["reply_service"] = ReplyService(settings)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    return application


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
    settings = Settings.from_env()
    if not settings.gigachat_credentials:
        LOGGER.warning("GIGACHAT_CREDENTIALS is not set; fallback replies will be used")
    else:
        LOGGER.info("Using GigaChat model: %s", settings.gigachat_model)

    # Python 3.14 no longer creates a default event loop automatically, while
    # some python-telegram-bot releases still expect one in run_polling().
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    LOGGER.info("Starting Telegram bot")
    build_application(settings).run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
