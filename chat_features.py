"""User-controlled facts and a bounded, chat-scoped group transcript."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import re
import sqlite3
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from runtime_storage import atomic_write_text, load_json_object


def fact_key(text: str) -> str:
    return re.sub(r'\W+', ' ', text.casefold().replace('ё', 'е')).strip()


class MemoryControls:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self.data = {}
        try:
            self.data = load_json_object(path)
        except FileNotFoundError:
            pass

    @staticmethod
    def scope(chat_id, user_id):
        return f'{chat_id}:{user_id}'

    def excluded(self, chat_id, user_id):
        with self.lock:
            return list(self.data.get(self.scope(chat_id, user_id), []))

    def blocked(self, chat_id, user_id, text):
        key = fact_key(text)
        return any(fact_key(old) == key for old in self.excluded(chat_id, user_id))

    def update(self, chat_id, user_id, text, *, remove=False):
        with self.lock:
            scope = self.scope(chat_id, user_id)
            values = self.data.setdefault(scope, [])
            values[:] = [old for old in values if fact_key(old) != fact_key(text)]
            if not remove:
                values.append(text)
            atomic_write_text(self.path, json.dumps(self.data, ensure_ascii=False, indent=2))


class LegacyMemory:
    """Manage only the requesting user's facts in the current chat."""
    def __init__(self, store, controls):
        self.store, self.controls = store, controls

    def scope(self, chat_id, user_id):
        return self.store.data.setdefault('users', {}).setdefault(str(chat_id), {}).setdefault(str(user_id), {})

    def items(self, chat_id, user_id):
        scope = self.scope(chat_id, user_id)
        items = {}
        for bucket in ('immutable', 'mutable'):
            for text in self.store._facts(scope, bucket):
                token = f'l:{chat_id}:' + hashlib.sha256(text.encode()).hexdigest()[:16]
                items[token] = {'id': token, 'memory': text, 'chat_id':int(chat_id)}
        return list(items.values())

    def all_items(self, user_id):
        items = []
        for chat_id, users in self.store.data.get('users', {}).items():
            if str(user_id) in users:
                items.extend(self.items(chat_id,user_id))
        return items

    def add(self, chat_id, user_id, text):
        scope = self.scope(chat_id, user_id)
        self.store._normalize_scope(scope)
        self.controls.update(chat_id, user_id, text, remove=True)
        facts = scope['immutable']['facts']
        if text not in facts:
            facts.append(text)
        self.store._save()

    def erase(self, chat_id, user_id, token):
        found = next((item for item in self.items(chat_id, user_id) if item['id'] == token), None)
        if found is None:
            raise ValueError('Факт не найден в вашей памяти этого чата.')
        text = found['memory']
        scope = self.scope(chat_id, user_id)
        for bucket in ('immutable', 'mutable'):
            if isinstance(scope.get(bucket), dict):
                scope[bucket]['facts'] = [fact for fact in scope[bucket].get('facts', []) if fact != text]
        self.controls.update(chat_id, user_id, text)
        self.store._save()
        return text

    def enforce(self):
        for chat_id, users in self.store.data.get('users', {}).items():
            for user_id, scope in users.items():
                for bucket in ('immutable', 'mutable'):
                    if isinstance(scope.get(bucket), dict):
                        scope[bucket]['facts'] = [fact for fact in scope[bucket].get('facts', [])
                                                  if not self.controls.blocked(chat_id, user_id, fact)]


class GroupTranscript:
    def __init__(self, path: Path, days=7, max_rows=3000):
        self.path, self.days, self.max_rows = path, days, max_rows
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS messages(chat_id INTEGER, message_id INTEGER, timestamp REAL, user_id INTEGER, name TEXT, text TEXT, reply_to INTEGER, PRIMARY KEY(chat_id,message_id))')
            db.execute('CREATE INDEX IF NOT EXISTS message_time ON messages(chat_id,timestamp)')
            db.execute('DELETE FROM messages WHERE timestamp < ?', (time.time()-self.days*86400,))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def append(self, chat_id, message_id, timestamp, user_id, name, text, reply_to=None):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO messages VALUES(?,?,?,?,?,?,?)', (chat_id, message_id, timestamp, user_id, name, text[:5000], reply_to))
            db.execute('DELETE FROM messages WHERE timestamp < ?', (time.time()-self.days*86400,))
            db.execute('DELETE FROM messages WHERE chat_id=? AND message_id IN (SELECT message_id FROM messages WHERE chat_id=? ORDER BY timestamp DESC,message_id DESC LIMIT -1 OFFSET ?)', (chat_id, chat_id, self.max_rows))

    def forget_user(self, user_id):
        with self.connect() as db:
            db.execute("DELETE FROM messages WHERE user_id=?", (user_id,))

    def recent(self, chat_id, since, limit=400):
        with self.connect() as db:
            db.execute('DELETE FROM messages WHERE timestamp < ?', (time.time()-self.days*86400,))
            total = db.execute('SELECT COUNT(*) FROM messages WHERE chat_id=? AND timestamp>=?', (chat_id,since)).fetchone()[0]
            rows = db.execute('SELECT timestamp,user_id,name,text,reply_to FROM messages WHERE chat_id=? AND timestamp>=? ORDER BY timestamp DESC,message_id DESC LIMIT ?', (chat_id,since,limit)).fetchall()
        return list(reversed(rows)), total


def transcript_chunks(rows, budget=60000, chunk_size=12000, timezone_name="UTC"):
    selected, size = [], 0
    for row in reversed(rows):
        entry = dict(zip(('timestamp','user_id','name','text','reply_to'), row))
        entry['sent_at'] = datetime.fromtimestamp(row[0], ZoneInfo(timezone_name)).isoformat()
        line = json.dumps(entry, ensure_ascii=False)
        if size+len(line) > budget:
            break
        selected.append(line)
        size += len(line)+1
    chunks, current = [], ''
    for line in reversed(selected):
        if current and len(current)+len(line)+1 > chunk_size:
            chunks.append(current)
            current = ''
        current += line+'\n'
    if current:
        chunks.append(current)
    return chunks, len(selected)
