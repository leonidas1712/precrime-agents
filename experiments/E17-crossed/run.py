"""Run E17 episodes through the v1 harness.

    .venv/bin/python experiments/E17-crossed/run.py --plan pilot --mock       # offline, scripted models (free)
    .venv/bin/python experiments/E17-crossed/run.py --plan main  --mock

    .venv/bin/python experiments/E17-crossed/run.py --plan pilot --live --cap-usd 3   # approved by Leo, 3 Nov 2026

Live settings copy E15: Qwen3-32B target pinned to its provider, deepseek-v4-flash peer, 20 rounds, early stop,
12 tool steps, 1536 tokens, price-capped routing, a hard spend cap. The main run needs a separate approval.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import crossed  # noqa: E402  (sets up sys.path for the harness)
import scenarios as v1_scenarios  # noqa: E402
from harness import Settings, run_episode  # noqa: E402
from mock_client import ScriptedClient  # noqa: E402
from plan import episodes  # noqa: E402

ROOT = HERE.parents[1]
MOCK_DIR = ROOT / 'private/crossed/mock_runs'
APPROVED = {'pilot': 'Leo, 3 Nov 2026 (chat)', 'pilot2': 'Leo, 4 Nov 2026 (chat)',
            'check4': 'Leo, 4 Nov 2026 (chat)',
            'main:1': 'Leo, 5 Nov 2026 (chat)'}  # main stage 2: not approved

# The harness asks scenarios.score_episode whether the main action happened (for early stop).
# v1's scorer reads the secret from the private file, which the SYSPROMPT arm doesn't have, so route E17 here.
_v1_score = v1_scenarios.score_episode


def _score(episode):
    if episode.get('scenario_spec', {}).get('metadata', {}).get('scenario_family') == 'e17-crossed':
        return crossed.score_crossed(episode)
    return _v1_score(episode)


v1_scenarios.score_episode = _score


async def run_mock(plan: str, limit: int | None = None, stage: int | None = None) -> Path:
    out = MOCK_DIR / plan
    out.mkdir(parents=True, exist_ok=True)
    settings = Settings('mock-target', 'mock-peer', rounds=20, max_tool_steps=12, max_tokens=1536)
    eps = episodes(plan, stage)[:limit]
    for ep in eps:
        path = out / f"{ep['episode_id']}.json"
        if path.exists():
            path.unlink()  # mock outputs are disposable; always regenerate
        scenario = crossed.build(ep['scenario'], ep['arm'], ep['location'], ep['episode_id'], ep['name_set'])
        client = ScriptedClient(scenario)
        result = await run_episode(scenario, settings, client, out, episode_id=ep['episode_id'])
        result['mock_outcome'] = client.outcome
        result['mock_action_round'] = client.action_round
        path.write_text(json.dumps(result))
    return out


async def run_live(plan: str, cap_usd: float, concurrency: int, stage: int | None = None) -> None:
    import hashlib
    import os
    from dataclasses import asdict, replace
    from dotenv import dotenv_values
    from openai import AsyncOpenAI
    from overnight_sweep import PEER, TARGETS, fetch_prices, options
    from scale_up import SpendCap
    from collusion.runs import start_run, write_json_atomic

    tag = f'{plan}:{stage}' if stage else plan
    if tag not in APPROVED:
        sys.exit(f'Only {sorted(APPROVED)} are approved.')
    model, _ = TARGETS['32b']
    prices = fetch_prices()
    settings = Settings(model, PEER, rounds=20, max_tool_steps=12, max_tokens=1536,
                        target_request_options=options(prices[model]),
                        recruiter_request_options=options(prices[PEER]),
                        api_retry_attempts=10, api_retry_max_delay=60,
                        # The DeepSeek peer sometimes returns an empty reply (pilot 1: 2 of 40). Resend up to twice.
                        empty_response_retries=2 if plan != 'pilot' else 0)
    key = os.environ.get('OPENROUTER_API_KEY') or dotenv_values(ROOT / '.env')['OPENROUTER_API_KEY']  # this repo's key
    client = AsyncOpenAI(api_key=key, base_url='https://openrouter.ai/api/v1', max_retries=0, timeout=90)
    code_hash = hashlib.sha256(b''.join((HERE / f).read_bytes() for f in ('crossed.py', 'plan.py', 'run.py'))).hexdigest()
    run = start_run(f'e17-{plan}' + (f'-stage{stage}' if stage else ''), experiment='E17',
                    settings={**asdict(settings), 'plan': plan, 'stage': stage, 'api_cap_usd': cap_usd,
                              'code_sha256': code_hash, 'approved_by': APPROVED[tag], 'version': crossed.VERSION},
                    tags=['e17', plan], model=model)
    budget = SpendCap(client, cap_usd, run.dir / 'api_budget.json')
    ep_dir = run.dir / 'episodes'
    gate = asyncio.Semaphore(concurrency)
    log = {}

    async def one(ep):
        async with gate:
            if budget.spent >= budget.cap:
                return
            sc = crossed.build(ep['scenario'], ep['arm'], ep['location'], ep['episode_id'], ep['name_set'])
            sc = replace(sc, recruiter_approved=True)  # Leo approved the pilot prompts (3 Nov 2026; pilot2 4 Nov)
            try:
                res = await run_episode(sc, settings, budget, ep_dir, episode_id=ep['episode_id'])
                out = (res['status'], str(res.get('stop_reason')), (res.get('label') or {}).get('outcome'))
            except Exception as exc:
                out = ('error', f'{type(exc).__name__}: {exc}', None)
            log[ep['episode_id']] = out
            write_json_atomic(run.dir / 'progress.json', {'spent_usd': str(budget.spent), 'calls': budget.calls, 'episodes': log})
            print('EPISODE', ep['episode_id'], *out, f'${budget.spent:.4f}', flush=True)

    await asyncio.gather(*(one(ep) for ep in episodes(plan, stage)))
    run.result(completed=sum(v[0] == 'complete' for v in log.values()), spent_usd=str(budget.spent), calls=budget.calls)
    run.finish('finished')
    await client.close()
    print('E17_DONE', run.dir, f'${budget.spent:.4f}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', choices=['pilot', 'pilot2', 'check4', 'main'], required=True)
    p.add_argument('--mock', action='store_true')
    p.add_argument('--live', action='store_true')
    p.add_argument('--limit', type=int)
    p.add_argument('--cap-usd', type=float, default=1.0)
    p.add_argument('--stage', type=int, choices=[1, 2])
    p.add_argument('--concurrency', type=int, default=4)
    a = p.parse_args()
    if a.live:
        asyncio.run(run_live(a.plan, a.cap_usd, a.concurrency, a.stage))
    elif a.mock:
        print(asyncio.run(run_mock(a.plan, a.limit, a.stage)))
    else:
        sys.exit('Pass --mock or --live.')
