import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import OpenAI
from local_memory import LocalMemory


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = SimpleNamespace(vsegpt_api_key='offline-test', vsegpt_base_url='https://mock.invalid/v1', vsegpt_model='test-model')
        self.config = {'mem0_enabled': True, 'mem0_search_timeout': 5}
        self.env = patch.dict(os.environ, {'MEM0_ENABLED': 'true', 'MEM0_TELEMETRY': 'false', 'MEM0_DIR': str(self.root / 'sdk')})
        self.env.start()
        self.calls = []
        def handle(request):
            self.calls.append(request.url.path)
            payload = json.loads(request.content)
            if request.url.path.endswith('/embeddings'):
                inputs = payload['input']
                if isinstance(inputs, str):
                    inputs = [inputs]
                return httpx.Response(200, json={'object': 'list', 'data': [{'object': 'embedding', 'index': i, 'embedding': [1.0] + [0.0]*1535} for i in range(len(inputs))], 'model': 'test', 'usage': {'prompt_tokens': 1, 'total_tokens': 1}})
            return httpx.Response(200, json={'id':'chat-offline','object':'chat.completion','created':1,'model':'test','choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content': json.dumps({'memory':[{'text':'Пользователь любит чай','entities':[]}]}, ensure_ascii=False)}}], 'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}})
        def client_factory(*args, **kwargs):
            kwargs['http_client'] = httpx.Client(transport=httpx.MockTransport(handle))
            return OpenAI(*args, **kwargs)
        self.patches = [patch('mem0.llms.openai.OpenAI', client_factory), patch('mem0.embeddings.openai.OpenAI', client_factory)]
        for item in self.patches:
            item.start()
        self.backends = []

    def backend(self, config=None):
        memory = LocalMemory(config or self.config, self.settings, self.root)
        self.backends.append(memory)
        return memory

    async def asyncTearDown(self):
        for memory in self.backends:
            if not memory._closed:
                await memory.close()
        for item in reversed(self.patches):
            item.stop()
        self.env.stop()
        self.temp.cleanup()

    async def test_inferred_fact_survives_restart_and_is_scoped(self):
        memory = self.backend()
        memory.remember('Я люблю чай', 100, 1, 'Ильсур', 'general')
        await asyncio.wait_for(asyncio.gather(*memory._writes), 10)
        self.assertIn('любит чай', await memory.search('Что я люблю?', 100, 1, 'general'))
        self.assertEqual('', await memory.search('Что я люблю?', 100, 2, 'general'))
        self.assertEqual('', await memory.search('Что я люблю?', -200, 1, 'general'))
        self.assertEqual('', await memory.search('Что я люблю?', 100, 1, 'english_coach'))
        await memory.close()
        restarted = self.backend()
        self.assertIn('любит чай', await restarted.search('Что я люблю?', 100, 1, 'general'))
        self.assertTrue((memory.path / 'history.sqlite3').exists())
        self.assertIn('/v1/chat/completions', self.calls)

    async def test_search_failure_returns_empty_context(self):
        memory = self.backend()
        with patch.object(memory, '_get_memory', side_effect=RuntimeError('offline')):
            self.assertEqual('', await memory.search('вопрос', 100, 1, 'general'))

    async def test_json_import_is_idempotent_and_forget_preserves_shared_facts(self):
        memory = self.backend()
        data = {'chats': {'-200': {'mutable': {'facts': ['В чате три участника']}}},
                'users': {'-200': {'1': {'immutable': {'facts': ['Любит чай']}}}}}
        self.assertEqual(2, await memory.import_json(data))
        self.assertEqual(0, await memory.import_json(data))
        self.assertIn('Любит чай', await memory.search('чай', -200, 1, 'general'))
        await memory.forget(1)
        recalled = await memory.search('чай', -200, 1, 'general')
        self.assertNotIn('Любит чай', recalled)
        self.assertIn('три участника', recalled)

    async def test_disabled_backend_creates_no_storage(self):
        with patch.dict(os.environ, {'MEM0_ENABLED':'false'}):
            memory = self.backend()
        memory.remember('Я люблю чай', 100, 1, 'Ильсур', 'general')
        self.assertEqual('', await memory.search('вопрос', 100, 1, 'general'))
        self.assertFalse(memory.path.exists())
