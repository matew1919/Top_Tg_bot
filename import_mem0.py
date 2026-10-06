"""One-off import of authoritative JSON facts into the bot's local Mem0 store."""
import argparse
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
from bot import Settings, load_runtime_config
from local_memory import LocalMemory


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--storage', required=True)
    parser.add_argument('--env-file', required=True)
    args = parser.parse_args()
    load_dotenv(args.env_file)
    config = load_runtime_config()
    config['mem0_storage_dir'] = str(Path(args.storage).resolve())
    backend = LocalMemory(config, Settings.from_env(), Path(__file__).resolve().parent)
    try:
        count = await backend.import_json(json.loads(Path(args.source).read_text()))
        result = await backend._submit(lambda: backend._get_memory().get_all(
            filters={'user_id': '*'}, top_k=100000))
        print(json.dumps({'imported': count, 'stored': len(result['results'])}))
        # Verify an actual embedding lookup using one stored fact, without printing it.
        personal = next((item for item in result['results']
                         if (item.get('metadata') or {}).get('telegram_user_id') != '0'), None)
        if personal:
            metadata = personal['metadata']
            recalled = await backend.search(personal['memory'], int(metadata['chat_id']),
                                            int(metadata['telegram_user_id']), 'general')
            if not recalled:
                raise RuntimeError('Mem0 real search returned no facts')
            print('Real embedding search: OK')
    finally:
        await backend.close()


if __name__ == '__main__':
    asyncio.run(main())
