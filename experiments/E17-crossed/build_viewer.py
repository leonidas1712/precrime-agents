"""Build a one-file HTML viewer of every E17 episode setup plus its MOCK trajectory and labels.

    .venv/bin/python experiments/E17-crossed/run.py --plan pilot --mock
    .venv/bin/python experiments/E17-crossed/run.py --plan main --mock
    .venv/bin/python experiments/E17-crossed/build_viewer.py      # -> private/crossed/E17_SETUP_VIEWER.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
from crossed import NAME_SETS, SCENARIOS, PILOT_SCENARIOS, ALLOW_TEMPLATES, FORBID_TEMPLATES, score_crossed  # noqa: E402
from mock_client import MOCK_RATES  # noqa: E402

ROOT = HERE.parents[1]
RUNS = ROOT / 'private/crossed/mock_runs'
OUT = ROOT / 'private/crossed/E17_SETUP_VIEWER.html'


def events(e, lab):
    scored = (lab['first_write_after_request'] or {}).get('call_id') if lab['in_label_B'] else None
    by_round = {}
    for c in e['channel']:
        if c['actor'] == 'recruiter':
            by_round.setdefault(c['round'], []).append(['peer', 'msg', c['content']])
    for t in e['tool_log']:
        if t['actor'] != 'target' or t['name'] == 'send_message':
            continue
        args = analyze_args(t)
        txt = args.get('path', '')
        if t['name'] == 'write_file':
            txt += '  ←  ' + str(args.get('content', ''))[:160]
        kind = 'scored' if t.get('call_id') == scored else t['name']
        by_round.setdefault(t['round'], []).append(['agent', kind, txt])
    for c in e['channel']:
        if c['actor'] == 'target':
            by_round.setdefault(c['round'], []).append(['agent', 'msg', c['content']])
    return [[r, *x] for r in sorted(by_round) for x in by_round[r]]


def analyze_args(t):
    a = t.get('arguments')
    try:
        return json.loads(a) if isinstance(a, str) else (a or {})
    except ValueError:
        return {}


def collect(plan):
    rows = []
    for e in analyze.load(RUNS / plan):
        m = e['scenario_spec']['metadata']
        lab = score_crossed(e)
        fw = lab['first_write_after_request'] or {}
        rows.append({
            'id': m['episode_id'], 'scenario': m['spray_row_id'], 'arm': m['arm'], 'loc': m['location'],
            'ns': m['name_set'], 'allowed': m['allowed_path'], 'forbidden': m['forbidden_path'],
            'pushed': m['pushed_path'], 'allowFirst': m['allowed_first_in_rule'],
            'listFirst': m['workspace_order'][0] == m['allowed_path'],
            'askFrom': m['peer_ask_from_message'], 'key': m['secret_key'], 'secret': m['secret_value'],
            'system': e['scenario_spec']['target_system'], 'task': e['scenario_spec']['target_task'],
            'peer': e['scenario_spec']['recruiter_system'], 'files': e['scenario_spec']['files'],
            'mock': e.get('mock_outcome'), 'outcome': lab['outcome'], 'labelB': lab['label_B'],
            'scoredRound': fw.get('round') if lab['in_label_B'] else None,
            'reqRound': lab['peer_first_request_round'], 'drift': lab['peer_drift'],
            'stop': e['stop_reason'], 'rounds': e['rounds_completed'],
            'events': events(e, lab),
        })
    return rows


def main():
    data = {}
    for plan in ('pilot', 'main'):
        if (RUNS / plan).exists():
            rep = analyze.report(RUNS / plan)
            data[plan] = {'episodes': collect(plan), 'gate': rep['gate'], 'contrasts': rep['contrasts']}
    meta = {'scenarios': SCENARIOS, 'pilotScenarios': PILOT_SCENARIOS, 'nameSets': NAME_SETS,
            'allowT': ALLOW_TEMPLATES, 'forbidT': FORBID_TEMPLATES,
            'mockRates': {f'{a} / {l}': r for (a, l), r in MOCK_RATES.items()}}
    html = TEMPLATE.replace('/*DATA*/null', json.dumps(data)).replace('/*META*/null', json.dumps(meta))
    OUT.write_text(html)
    print(OUT, f'{OUT.stat().st_size / 1e6:.1f} MB')


TEMPLATE = (HERE / 'viewer_template.html').read_text()

if __name__ == '__main__':
    main()
