import asyncio
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot import MemoryStore, RequestRateLimiter
from chat_features import GroupTranscript, LegacyMemory, MemoryControls, transcript_chunks
from feature_handlers import memory_command, forget_command, remember_command, summary_command, record_group_message


class LegacyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.controls = MemoryControls(root/'controls.json')
        self.store = MemoryStore(str(root/'memory.json'))
        self.store.controls = self.controls
        self.legacy = LegacyMemory(self.store,self.controls)

    def test_forgotten_fact_is_not_restored_and_other_user_is_preserved(self):
        self.legacy.add(-10,1,'Любит чай')
        self.legacy.add(-10,2,'Любит чай')
        token = self.legacy.items(-10,1)[0]['id']
        self.legacy.erase(-10,1,token)
        self.store.rewrite_from_extracted(-10,{'users':{'1':{'facts':['Любит чай']},'2':{'facts':['Любит чай']}}})
        self.assertEqual([],self.legacy.items(-10,1))
        self.assertEqual('Любит чай',self.legacy.items(-10,2)[0]['memory'])
        reloaded = MemoryControls(self.controls.path)
        self.assertTrue(reloaded.blocked(-10,1,'любит чай!'))
        self.legacy.add(-10,1,'Любит чай')
        self.assertFalse(self.controls.blocked(-10,1,'Любит чай'))

    def test_cross_user_and_chat_token_cannot_delete(self):
        self.legacy.add(-10,1,'Любит чай')
        token = self.legacy.items(-10,1)[0]['id']
        with self.assertRaises(ValueError):
            self.legacy.erase(-10,2,token)
        with self.assertRaises(ValueError):
            self.legacy.erase(-20,1,token)
        self.assertEqual(1,len(self.legacy.items(-10,1)))


class TranscriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'messages.sqlite3'
        self.store = GroupTranscript(self.path,max_rows=3)

    def test_retention_edits_restart_and_chat_isolation(self):
        now = time.time()
        self.store.append(-1,1,now-8*86400,1,'A','expired')
        for number in range(2,6):
            self.store.append(-1,number,now+number,1,'A',f'message {number}')
        self.store.append(-2,1,now,2,'B','Other chat')
        self.store.append(-1,5,now+5,1,'A','edited message')
        rows,total = GroupTranscript(self.path).recent(-1,now-10*86400)
        self.assertEqual(3,total)
        self.assertEqual(['message 3','message 4','edited message'],[row[3] for row in rows])
        self.assertNotIn('Other chat',str(rows))
        self.store.forget_user(1)
        self.assertEqual(0,self.store.recent(-1,0)[1])
        self.assertEqual(1,self.store.recent(-2,0)[1])

    def test_large_transcript_is_bounded_and_keeps_latest(self):
        rows=[(i,1,'A',str(i)+'x'*1000,None) for i in range(100)]
        chunks,count = transcript_chunks(rows,budget=5000,chunk_size=2200)
        self.assertLess(count,100)
        self.assertTrue(all(len(chunk)<=2200 for chunk in chunks))
        self.assertIn('99xxx',''.join(chunks))
        self.assertNotIn('"timestamp": 0,',''.join(chunks))


class HandlerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = MemoryStore(str(self.root/'memory.json'))
        self.controls = MemoryControls(self.root/'controls.json')
        self.store.controls = self.controls
        self.legacy = LegacyMemory(self.store,self.controls)
        self.semantic = SimpleNamespace(list_facts=AsyncMock(return_value=[]),erase_fact=AsyncMock(),
            put_fact=AsyncMock(return_value=True),forget=AsyncMock(),_cache={})
        self.service = SimpleNamespace(memory=self.store,memory_controls=self.controls,semantic_memory=self.semantic,
            settings=SimpleNamespace(ai_provider='vsegpt',vsegpt_model='mock',timezone='Asia/Yekaterinburg'),
            forget_user_data=lambda user,chat: self.store.forget_user(user,chat),
            _vsegpt_chat=AsyncMock(return_value=object()),_vsegpt_response_text=lambda response:'Обсудили поездку.')
        self.journal = GroupTranscript(self.root/'groups.sqlite3')
        self.message = SimpleNamespace(reply_text=AsyncMock(),text='Обычный разговор',caption=None,
            message_id=1,date=datetime.now(timezone.utc),from_user=SimpleNamespace(id=1,is_bot=False,full_name='A'),reply_to_message=None)
        self.update = SimpleNamespace(effective_message=self.message,effective_chat=SimpleNamespace(id=1,type='private'),effective_user=SimpleNamespace(id=1))
        self.context = SimpleNamespace(args=[],application=SimpleNamespace(bot_data={'reply_service':self.service,'group_transcript':self.journal}))

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def test_memory_lists_only_sender_facts_and_includes_mem0(self):
        self.legacy.add(-10,1,'Свой факт')
        self.legacy.add(-10,2,'Чужой секрет')
        self.semantic.list_facts.return_value=[{'id':'m:-10:general:id','chat_id':-10,'memory':'Факт Mem0'}]
        await memory_command(self.update,self.context)
        response=''.join(call.args[0] for call in self.message.reply_text.call_args_list)
        self.assertIn('Свой факт',response)
        self.assertIn('Факт Mem0',response)
        self.assertNotIn('Чужой секрет',response)
        self.semantic.list_facts.assert_awaited_once_with(1)

    async def test_group_memory_command_is_private(self):
        self.update.effective_chat.type='supergroup'
        await memory_command(self.update,self.context)
        self.semantic.list_facts.assert_not_awaited()
        self.assertIn('личном',self.message.reply_text.call_args.args[0])

    async def test_forget_removes_json_and_exact_mem0_duplicate(self):
        self.legacy.add(-10,1,'Любит чай')
        token=self.legacy.items(-10,1)[0]['id']
        self.semantic.list_facts.return_value=[{'id':'m:-10:general:id','chat_id':-10,'memory':'любит чай!'}]
        self.context.args=[token]
        await forget_command(self.update,self.context)
        self.assertEqual([],self.legacy.items(-10,1))
        self.semantic.erase_fact.assert_awaited_once_with(1,'m:-10:general:id')

    async def test_remember_survives_mem0_api_failure(self):
        self.semantic.put_fact.side_effect=RuntimeError('offline')
        self.context.args=['Любит','чай']
        await remember_command(self.update,self.context)
        self.assertEqual('Любит чай',self.legacy.items(1,1)[0]['memory'])
        self.assertIn('JSON',self.message.reply_text.call_args.args[0])

    async def test_edit_replaces_fact_in_its_source_chat(self):
        self.legacy.add(-10,1,'Любит чай')
        self.context.args=['edit',self.legacy.items(-10,1)[0]['id'],'Любит','кофе']
        await memory_command(self.update,self.context)
        self.assertEqual(['Любит кофе'],[x['memory'] for x in self.legacy.items(-10,1)])
        self.assertTrue(self.controls.blocked(-10,1,'Любит чай'))
        self.semantic.put_fact.assert_awaited_once_with(-10,1,'Любит кофе')

    async def test_mem0_failure_does_not_claim_full_deletion(self):
        self.legacy.add(1,1,'Факт')
        self.context.args=[self.legacy.items(1,1)[0]['id']]
        self.semantic.list_facts.side_effect=RuntimeError('offline')
        await forget_command(self.update,self.context)
        self.assertEqual(1,len(self.legacy.items(1,1)))
        self.assertIn('отложено',self.message.reply_text.call_args.args[0])

    async def test_confirm_erases_user_transcript_but_preserves_others(self):
        now=time.time()
        self.journal.append(-10,1,now,1,'A','Свой текст')
        self.journal.append(-10,2,now,2,'B','Чужой текст')
        self.context.args=['confirm']
        await forget_command(self.update,self.context)
        self.assertEqual(['Чужой текст'],[x[3] for x in self.journal.recent(-10,0)[0]])
        self.semantic.forget.assert_awaited_once_with(1)

    async def test_recorder_captures_ordinary_group_messages_and_edits(self):
        self.update.effective_chat=SimpleNamespace(id=-10,type='supergroup')
        await record_group_message(self.update,self.context)
        self.message.text='Правка'
        await record_group_message(self.update,self.context)
        rows,total=self.journal.recent(-10,0)
        self.assertEqual(1,total)
        self.assertEqual('Правка',rows[0][3])
        self.message.text='/summary'
        await record_group_message(self.update,self.context)
        self.assertEqual(1,self.journal.recent(-10,0)[1])

    async def test_summary_uses_only_current_group(self):
        now=time.time()
        self.journal.append(-10,1,now,1,'A','Обсудим поездку')
        self.journal.append(-20,1,now,2,'B','Секрет другой группы')
        self.update.effective_chat=SimpleNamespace(id=-10,type='group')
        await summary_command(self.update,self.context)
        request=self.service._vsegpt_chat.call_args.kwargs['messages']
        self.assertIn('Обсудим поездку',request[1]['content'])
        self.assertNotIn('Секрет',request[1]['content'])
        self.assertIn('1 записанным',self.message.reply_text.call_args.args[0])

    async def test_summary_empty_invalid_private_and_rate_limit_do_not_call_ai(self):
        await summary_command(self.update,self.context)
        self.update.effective_chat=SimpleNamespace(id=-10,type='group')
        self.context.args=['999']
        await summary_command(self.update,self.context)
        self.context.args=[]
        await summary_command(self.update,self.context)
        self.journal.append(-10,1,time.time(),1,'A','Текст')
        limiter=RequestRateLimiter(1,60)
        limiter.allow((-10,1),asyncio.get_running_loop().time())
        self.context.application.bot_data['request_rate_limiter']=limiter
        await summary_command(self.update,self.context)
        self.service._vsegpt_chat.assert_not_awaited()

    async def test_summary_failure_keeps_messages(self):
        self.journal.append(-10,1,time.time(),1,'A','Текст')
        self.update.effective_chat=SimpleNamespace(id=-10,type='group')
        self.service._vsegpt_chat.side_effect=RuntimeError('offline')
        await summary_command(self.update,self.context)
        self.assertEqual(1,self.journal.recent(-10,0)[1])
        self.assertIn('Не удалось',self.message.reply_text.call_args.args[0])

    async def test_summary_large_chat_batches_and_reports_truncation(self):
        now=time.time()
        for i in range(70):
            self.journal.append(-10,i,now+i,1,'A','Большое сообщение '+('x'*1500))
        self.update.effective_chat=SimpleNamespace(id=-10,type='group')
        await summary_command(self.update,self.context)
        self.assertGreater(self.service._vsegpt_chat.await_count,2)
        self.assertIn('ограничения объёма',self.message.reply_text.call_args.args[0])

if __name__=='__main__':
    unittest.main()

class RegistrationTests(unittest.TestCase):
    def test_commands_and_passive_recorder_registered_without_network(self):
        import os
        from bot import Settings, build_application
        from telegram.ext import CommandHandler, MessageHandler
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'TELEGRAM_BOT_TOKEN':'123456:offline-token','MEM0_ENABLED':'false'}):
            root=Path(directory)
            settings=Settings.from_env()
            import shutil
            shutil.copy2(Path(__file__).resolve().parents[1]/'vacancy_config.json',root/'vacancy_config.json')
            with patch('bot.project_path',side_effect=lambda filename:root/filename):
                app=build_application(settings)
            commands=set()
            for handler in app.handlers[0]:
                if isinstance(handler,CommandHandler):
                    commands.update(handler.commands)
            self.assertTrue({'memory','forget','remember','summary'}<=commands)
            self.assertIsInstance(app.handlers[-1][0],MessageHandler)
            self.assertEqual('record_group_message',app.handlers[-1][0].callback.__name__)

    def test_memory_control_backup_is_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'controls.json'
            controls=MemoryControls(path)
            controls.update(-1,1,'Первый факт')
            controls.update(-1,1,'Второй факт')
            path.write_text('invalid JSON')
            recovered=MemoryControls(path)
            self.assertTrue(recovered.blocked(-1,1,'Первый факт'))
