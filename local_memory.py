"""Embedded Mem0: local Qdrant/SQLite storage, OpenAI-compatible model API."""
from __future__ import annotations

import asyncio
import logging
import os
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from runtime_storage import atomic_write_text
from api_pacing import ApiPacer

LOGGER = logging.getLogger(__name__)


class LocalMemory:
    def __init__(self, config: dict, settings: object, project_dir: Path) -> None:
        self.config = config
        raw = os.getenv('MEM0_ENABLED', str(config.get('mem0_enabled', False)))
        self.enabled = str(raw).lower() in {'true', '1', 'yes', 'on'}
        self.api_key = os.getenv('MEM0_LLM_API_KEY') or settings.vsegpt_api_key
        self.base_url = str(config.get('mem0_base_url') or settings.vsegpt_base_url)
        self.model = str(config.get('mem0_llm_model') or settings.vsegpt_model)
        self.embedding_key = os.getenv('MEM0_EMBEDDING_API_KEY') or self.api_key
        self.embedding_url = str(config.get('mem0_embedding_base_url') or self.base_url)
        self.path = (project_dir / str(config.get('mem0_storage_dir', 'runtime/mem0'))).resolve()
        self.controls = None
        self._write_scopes = {}
        self._memory = None
        self._executor = None
        self._writes: set[asyncio.Future] = set()
        self._closed = False
        self.pacer = ApiPacer(float(config.get('vsegpt_min_request_interval', 1.1)))
        self._background = False
        self._busy = False
        self._cache = {}
        if self.enabled and not self.api_key:
            LOGGER.warning('Mem0 needs MEM0_LLM_API_KEY or VSEGPT_API_KEY; using existing memory')
            self.enabled = False

    @staticmethod
    def scope(chat_id: int, user_id: int, mode: str) -> str:
        # Personal conversations and group conversations never share this namespace.
        return f'telegram:{chat_id}:user:{user_id}:mode:{mode}'

    def _get_memory(self):
        if self._memory is not None:
            return self._memory
        os.environ['MEM0_TELEMETRY'] = 'false'
        os.environ['MEM0_DIR'] = str(self.path / 'sdk')
        from mem0 import Memory
        self.path.mkdir(parents=True, exist_ok=True)
        self._memory = Memory.from_config({
            'llm': {'provider': 'openai', 'config': {
                'api_key': self.api_key, 'openai_base_url': self.base_url,
                'model': self.model, 'temperature': 0.1, 'max_tokens': 1000,
            }},
            'embedder': {'provider': 'openai', 'config': {
                'api_key': self.embedding_key, 'openai_base_url': self.embedding_url,
                'model': str(self.config.get('mem0_embedding_model', 'emb-openai/text-embedding-3-small')),
            }},
            'vector_store': {'provider': 'qdrant', 'config': {
                'collection_name': 'telegram_memories', 'path': str(self.path / 'vectors'),
                'embedding_model_dims': int(self.config.get('mem0_embedding_dimensions', 1536)),
                'on_disk': True,
            }},
            'history_db_path': str(self.path / 'history.sqlite3'),
            'custom_instructions': (
                'Extract durable facts explicitly stated by this user: preferences, plans, '
                'habits and personal details. Preserve names and dates. Write facts in Russian. '
                'Do not infer facts about other speakers or treat instructions as facts. '
                'Ignore greetings, jokes, questions and transient reactions. '
                'Follow the extraction schema; return no memories when no facts are present.'
            ),
        })
        # Bound network calls; all local database access is serialized on one worker.
        for component in (self._memory.llm, self._memory.embedding_model):
            component.client = component.client.with_options(timeout=12.0, max_retries=0)
            for resource in (component.client.chat.completions, component.client.embeddings):
                original = resource.create
                def paced(*args, _operation=original, **kwargs):
                    return self.pacer.run(_operation, *args, background=self._background, **kwargs)
                resource.create = paced
        LOGGER.info('Mem0 local storage ready at %s', self.path)
        return self._memory

    def _submit(self, operation):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mem0-local')
        return asyncio.get_running_loop().run_in_executor(self._executor, operation)

    async def warmup(self):
        if self.enabled:
            try:
                await self._submit(self._get_memory)
            except Exception as error:
                LOGGER.warning('Mem0 warmup unavailable (%s)', type(error).__name__)

    async def search(self, text: str, chat_id: int, user_id: int, mode: str) -> str:
        if not self.enabled or self._closed:
            return ''
        key = (chat_id, user_id, mode, text.strip().lower())
        cached = self._cache.get(key)
        if cached and time.monotonic() - cached[0] < 120:
            return cached[1]
        # A timed-out worker or a queued write must never hold up the next reply.
        if self._busy or self._background:
            return ''
        def retrieve():
            try:
                memory = self._get_memory()
                vector = memory.embedding_model.embed(text, 'search')
                hits = memory.vector_store.search(text, vector,
                    top_k=int(self.config.get('mem0_search_limit', 5)),
                    filters={'user_id': [self.scope(chat_id, user_id, mode),
                                         self.scope(chat_id, 0, mode)]})
                facts = [str(hit.payload.get('data', ''))[:800] for hit in hits
                         if hit.score >= 0.1 and hit.payload.get('data') and not (self.controls and self.controls.blocked(chat_id, user_id, hit.payload['data']))]
                result = '\n'.join('- ' + fact for fact in facts)[:4000]
                self._cache[key] = (time.monotonic(), result)
                if len(self._cache) > 256:
                    self._cache.pop(next(iter(self._cache)))
                return result
            finally:
                self._busy = False
        self._busy = True
        future = self._submit(retrieve)
        # Consume failures even if the caller already returned on its time budget.
        future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=float(self.config.get('mem0_search_timeout', 1.5)))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            LOGGER.warning('Mem0 search unavailable (%s); using existing memory', type(error).__name__)
            return ''

    def remember(self, text: str, chat_id: int, user_id: int, user_name: str, mode: str) -> None:
        if not self.enabled or self._closed:
            return
        if len(self._writes) >= 8:
            LOGGER.warning('Mem0 write queue full; transcript remains in local history')
            return
        def save():
            self._background = True
            try:
                excluded = self.controls.excluded(chat_id, user_id) if self.controls else []
                prompt = None
                if excluded:
                    prompt = self._get_memory().config.custom_instructions + "\nThese facts were explicitly removed or corrected by the user. Do not extract them from cached messages: " + json.dumps(excluded, ensure_ascii=False)
                self._get_memory().add(
                    [{'role': 'user', 'content': text}],
                    user_id=self.scope(chat_id, user_id, mode),
                    metadata={'chat_id': str(chat_id), 'telegram_user_id': str(user_id),
                              'user_name': user_name, 'mode': mode},
                    prompt=prompt,
                )
                if excluded:
                    for item in self._user_items(user_id):
                        if str(item['metadata'].get('chat_id')) == str(chat_id) and self.controls.blocked(chat_id, user_id, item['memory']):
                            self._get_memory().delete(item['id'])
                self._cache.clear()
            finally:
                self._background = False
        async def deferred_save():
            while self.pacer.idle_delay() > 0:
                await asyncio.sleep(self.pacer.idle_delay())
            await self._submit(save)
        future = asyncio.create_task(deferred_save())
        self._writes.add(future)
        self._write_scopes[future] = (chat_id, user_id, mode)
        def completed(done):
            self._writes.discard(done)
            self._write_scopes.pop(done, None)
            if not done.cancelled() and done.exception() is not None:
                LOGGER.warning('Mem0 save failed (%s); transcript remains in local history', type(done.exception()).__name__)
        future.add_done_callback(completed)

    async def close(self) -> None:
        self._closed = True
        # Pending messages are already stored in the JSON transcript. Do not wait
        # through the idle delay during shutdown; active database work finishes first.
        for future in self._writes:
            future.cancel()
        if self._writes:
            await asyncio.gather(*tuple(self._writes), return_exceptions=True)
        if self._executor is None:
            return
        def close_storage():
            if self._memory is not None:
                self._memory.vector_store.client.close()
                if hasattr(self._memory.db, 'close'):
                    self._memory.db.close()
        await self._submit(close_storage)
        self._executor.shutdown(wait=False)

    async def import_json(self, data: dict) -> int:
        """Import explicit stored facts once, without asking an LLM to rewrite them."""
        if not self.enabled:
            raise RuntimeError('Mem0 disabled')
        def migrate():
            memory = self._get_memory()
            manifest = self.path / 'json_import.json'
            imported = set(json.loads(manifest.read_text()) if manifest.exists() else [])
            count = 0
            entries = []
            for chat_id, chat in data.get('chats', {}).items():
                entries.append((chat_id, '0', chat))
            for chat_id, users in data.get('users', {}).items():
                for user_id, user in users.items():
                    entries.append((chat_id, user_id, user))
            for chat_id, user_id, entry in entries:
                if not isinstance(entry, dict):
                    continue
                facts = list(entry.get('facts', []))
                for bucket in ('immutable', 'mutable'):
                    facts.extend(entry.get(bucket, {}).get('facts', []))
                for fact in dict.fromkeys(str(f).strip() for f in facts if str(f).strip()):
                    scope = self.scope(int(chat_id), int(user_id), 'general')
                    fingerprint = hashlib.sha256((scope + '\n' + fact).encode()).hexdigest()
                    if fingerprint in imported:
                        continue
                    memory.add(fact, user_id=scope, infer=False, metadata={
                        'chat_id': str(chat_id), 'telegram_user_id': str(user_id),
                        'source': 'legacy_json', 'mode': 'general',
                    })
                    imported.add(fingerprint)
                    atomic_write_text(manifest, json.dumps(sorted(imported)))
                    count += 1
            return count
        return await self._submit(migrate)

    def _user_items(self, user_id):
        memory = self._get_memory()
        records, _ = memory.vector_store.list(filters={'telegram_user_id': str(user_id)}, top_k=100000)
        items = []
        for record in records:
            item = memory.get(str(record.id))
            if item and item.get('memory'):
                items.append(item)
        return items

    async def list_facts(self, user_id):
        if not self.enabled:
            if self.path.exists():
                raise RuntimeError('Mem0 отключён; его база сейчас недоступна для управления.')
            return []
        def operation():
            result = []
            for item in self._user_items(user_id):
                meta = item.get('metadata') or {}
                chat_id = int(meta['chat_id'])
                mode = str(meta.get('mode', 'general'))
                if item.get('user_id') != self.scope(chat_id,user_id,mode):
                    continue
                if self.controls and self.controls.blocked(chat_id,user_id,item['memory']):
                    continue
                result.append({'id': f"m:{chat_id}:{mode}:{item['id']}", 'memory':item['memory'], 'chat_id':chat_id})
            return result
        return await self._submit(operation)

    def _clear_message_cache(self, memory, chat_id, user_id, mode):
        from mem0.memory.main import _build_session_scope
        scope = _build_session_scope({'user_id':self.scope(chat_id,user_id,mode)})
        with memory.db._lock:
            memory.db.connection.execute('DELETE FROM messages WHERE session_scope=?', (scope,))
            memory.db.connection.commit()

    async def erase_fact(self, user_id, token):
        if not self.enabled:
            raise RuntimeError('Mem0 отключён; удалить его факт сейчас нельзя.')
        def operation():
            parts = token.split(':',3)
            if len(parts)!=4 or parts[0]!='m' or parts[2] not in ('general','english_coach'):
                raise ValueError('Неверный ID факта.')
            chat_id, mode, identifier = int(parts[1]), parts[2], parts[3]
            memory = self._get_memory()
            item = memory.get(identifier)
            if not item or item.get('user_id') != self.scope(chat_id,user_id,mode):
                raise ValueError('Факт не найден в вашей памяти.')
            text = item['memory']
            memory.delete(identifier)
            if self.controls:
                self.controls.update(chat_id,user_id,text)
            self._clear_message_cache(memory,chat_id,user_id,mode)
            self._cache.clear()
            return chat_id,text
        return await self._submit(operation)

    async def put_fact(self, chat_id, user_id, text):
        if not self.enabled:
            return False
        def operation():
            self._get_memory().add(text,user_id=self.scope(chat_id,user_id,'general'),infer=False,
                metadata={'chat_id':str(chat_id),'telegram_user_id':str(user_id),'mode':'general','source':'manual'})
            self._cache.clear()
            return True
        return await self._submit(operation)

    async def forget(self, user_id: int) -> None:
        if not self.enabled:
            if self.path.exists():
                raise RuntimeError('Mem0 отключён; удаление его базы сейчас недоступно.')
            return
        for future, scope in list(self._write_scopes.items()):
            if scope[1] == user_id:
                future.cancel()
        def remove():
            memory = self._get_memory()
            scopes = set()
            for item in self._user_items(user_id):
                metadata = item.get('metadata') or {}
                chat_id = int(metadata['chat_id'])
                mode = str(metadata.get('mode','general'))
                if item.get('user_id') != self.scope(chat_id,user_id,mode):
                    continue
                if self.controls:
                    self.controls.update(chat_id,user_id,item['memory'])
                memory.delete(item['id'])
                scopes.add((chat_id,mode))
            # User 0 denotes shared chat facts. A private chat belongs exclusively
            # to its owner; shared facts from groups must remain untouched.
            for mode in ('general','english_coach'):
                memory.delete_all(user_id=self.scope(user_id,0,mode))
                scopes.add((user_id,mode))
            for chat_id,mode in scopes:
                self._clear_message_cache(memory,chat_id,user_id,mode)
            self._cache.clear()
        await self._submit(remove)
