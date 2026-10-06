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

    async def test_disabled_backend_creates_no_storage(self):
        with patch.dict(os.environ, {'MEM0_ENABLED':'false'}):
            memory = self.backend()
        memory.remember('Я люблю чай', 100, 1, 'Ильсур', 'general')
        self.assertEqual('', await memory.search('вопрос', 100, 1, 'general'))
        self.assertFalse(memory.path.exists())


class ReplyIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def service(self):
        from unittest.mock import AsyncMock, Mock
        from bot import ReplyService
        service = ReplyService.__new__(ReplyService)
        service.settings = SimpleNamespace(ai_provider='vsegpt', english_coach_prompt='coach', timezone='Europe/Moscow')
        service.vsegpt_client = object()
        service.memory = Mock()
        service.memory.user_name.return_value = 'Ильсур'
        service.semantic_memory = Mock()
        service.semantic_memory.search = AsyncMock(return_value='Любит чай')
        service._prepare_history = AsyncMock(return_value=[])
        service._general_system_message = Mock(return_value='system')
        service._api_messages = Mock(return_value=[])
        service._save_history = Mock()
        service._vsegpt_reply_with_tools = AsyncMock(return_value='Ответ')
        return service

    async def test_reply_uses_retrieved_facts_and_schedules_save(self):
        service = self.service()
        self.assertEqual('Ответ', await service.reply(100, 1, 'Ильсур', 'Привет', 'general'))
        request = service._vsegpt_reply_with_tools.call_args.args[0]
        self.assertIn('Любит чай', request[0]['content'])
        service.semantic_memory.remember.assert_called_once_with('Привет', 100, 1, 'Ильсур', 'general')

    async def test_failed_reply_does_not_save_fact(self):
        service = self.service()
        service._vsegpt_reply_with_tools.side_effect = RuntimeError('API offline')
        await service.reply(100, 1, 'Ильсур', 'Привет', 'general')
        service.semantic_memory.remember.assert_not_called()


if __name__ == '__main__':
    unittest.main()
