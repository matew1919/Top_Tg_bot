"""Embedded Mem0: local Qdrant/SQLite storage, OpenAI-compatible model API."""
from __future__ import annotations

import asyncio
import logging
import os
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from runtime_storage import atomic_write_text

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
        self._memory = None
        self._executor = None
        self._writes: set[asyncio.Future] = set()
        self._closed = False
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
        LOGGER.info('Mem0 local storage ready at %s', self.path)
        return self._memory

    def _submit(self, operation):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mem0-local')
        return asyncio.get_running_loop().run_in_executor(self._executor, operation)

    async def search(self, text: str, chat_id: int, user_id: int, mode: str) -> str:
        if not self.enabled or self._closed:
            return ''
        def retrieve():
            memory = self._get_memory()
            result = memory.search(
                query=text, filters={'user_id': self.scope(chat_id, user_id, mode)},
                top_k=int(self.config.get('mem0_search_limit', 5)),
            )
            shared = memory.search(
                query=text, filters={'user_id': self.scope(chat_id, 0, mode)},
                top_k=2,
            )
            result['results'].extend(shared.get('results', []))
            return result
        future = self._submit(retrieve)
        try:
            result = await asyncio.wait_for(future, timeout=float(self.config.get('mem0_search_timeout', 4)))
            hits = result.get('results', []) if isinstance(result, dict) else result
            facts = [str(hit['memory'])[:800] for hit in hits if isinstance(hit, dict) and hit.get('memory')]
            return '\n'.join('- ' + fact for fact in facts)[:4000]
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
            self._get_memory().add(
                [{'role': 'user', 'content': text}],
                user_id=self.scope(chat_id, user_id, mode),
                metadata={'chat_id': str(chat_id), 'telegram_user_id': str(user_id),
                          'user_name': user_name, 'mode': mode},
            )
        future = self._submit(save)
        self._writes.add(future)
        def completed(done):
            self._writes.discard(done)
            if not done.cancelled() and done.exception() is not None:
                LOGGER.warning('Mem0 save failed (%s); transcript remains in local history', type(done.exception()).__name__)
        future.add_done_callback(completed)

    async def close(self) -> None:
        self._closed = True
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

    async def forget(self, user_id: int) -> None:
        if not self.enabled:
            return
        def remove():
            memory = self._get_memory()
            memories = memory.get_all(filters={'user_id': '*'}, top_k=100000)
            for item in memories.get('results', []):
                metadata = item.get('metadata') or {}
                if str(metadata.get('telegram_user_id')) == str(user_id):
                    memory.delete(item['id'])
        await self._submit(remove)
