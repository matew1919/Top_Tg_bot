"""Persistent cleanup for temporary Telegram tool-status messages."""

import asyncio
import json
import logging
from pathlib import Path

from telegram.error import BadRequest

from runtime_storage import atomic_write_text, load_json_object

LOGGER = logging.getLogger(__name__)


class StatusMessages:
    def __init__(self, bot, path: Path):
        self.bot = bot
        self.path = path
        try:
            data = load_json_object(path)
        except FileNotFoundError:
            data = {}
        except (OSError, ValueError):
            LOGGER.exception("Could not load status cleanup queue")
            data = {}
        entries = data.get("messages", [])
        if not isinstance(entries, list):
            entries = []
        self.pending = {
            (int(item[0]), int(item[1]))
            for item in entries
            if isinstance(item, list) and len(item) == 2
            and all(isinstance(value, int) for value in item)
        }
        # Entries from a previous process are ready for cleanup immediately.
        self.active: set[tuple[int, int]] = set()
        self.sends: set[asyncio.Task] = set()
        self.deleting: set[tuple[int, int]] = set()

    def save(self):
        try:
            atomic_write_text(self.path, json.dumps({"messages": sorted(self.pending)}))
        except OSError:
            LOGGER.exception("Could not persist status cleanup queue; retaining it in memory")

    def track(self, key, *, active):
        self.pending.add(key)
        if active:
            self.active.add(key)
        self.save()

    async def delete(self, key):
        if key not in self.pending or key in self.active or key in self.deleting:
            return
        self.deleting.add(key)
        try:
            await asyncio.wait_for(
                self.bot.delete_message(chat_id=key[0], message_id=key[1]), timeout=3,
            )
        except BadRequest as error:
            if "message to delete not found" not in str(error).lower():
                LOGGER.warning("Status cleanup deferred: chat=%s message=%s", *key)
                return
        except Exception:
            LOGGER.warning("Status cleanup deferred: chat=%s message=%s", *key)
            return
        finally:
            self.deleting.discard(key)
        self.pending.discard(key)
        self.save()

    async def run(self):
        while True:
            for key in list(self.pending - self.active):
                await self.delete(key)
            await asyncio.sleep(10)

    async def close(self):
        # Allow late send acknowledgements to supply IDs before shutdown.
        if self.sends:
            await asyncio.gather(*list(self.sends), return_exceptions=True)


class SearchStatus:
    def __init__(self, store: StatusMessages, message):
        self.store = store
        self.message = message
        self.closed = False
        self.key = None
        self.task = None

    async def show(self, text):
        if self.task is None:
            self.task = asyncio.create_task(self._send(text))
            self.store.sends.add(self.task)
            self.task.add_done_callback(self.store.sends.discard)
        # Caller may time out; retain the send to capture a late message ID.
        await asyncio.shield(self.task)

    async def _send(self, text):
        try:
            sent = await asyncio.wait_for(self.message.reply_text(text), timeout=15)
            self.key = (sent.chat.id, sent.message_id)
            self.store.track(self.key, active=not self.closed)
            if self.closed:
                await self.store.delete(self.key)
        except Exception:
            LOGGER.warning("Tool status delivery failed or unconfirmed; not resending")

    async def finish(self):
        self.closed = True
        if self.key is not None:
            self.store.active.discard(self.key)
            await self.store.delete(self.key)
