"""Build a one-file HTML viewer for real E17 episodes (labels, evidence, full transcripts).

    .venv/bin/python experiments/E17-crossed/build_pilot_viewer.py runs/20261103-1931-e17-pilot
    -> private/crossed/E17_PILOT_VIEWER.html
    .venv/bin/python experiments/E17-crossed/build_pilot_viewer.py runs/<pilot2 run> runs/20261103-1931-e17-pilot
    -> private/crossed/E17_PILOT2_VIEWER.html (Results tab pools pilot 1's PF episodes where the design is unchanged)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from crossed import score_crossed  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / 'private/crossed/E17_PILOT_VIEWER.html'


def _args(t):
    a = t.get('arguments')
    try:
        return json.loads(a) if isinstance(a, str) else (a or {})
    except ValueError:
        return {'_raw': str(a)}


def events(e):
    """Round-ordered list: peer messages first in each round, then the target's tool calls, then its replies."""
    sent = set()
    by_round = {}
    peer_tools = {}
    for c in e['channel']:
        if c['actor'] == 'recruiter':
            by_round.setdefault(c['round'], {'peer': [], 'target': []})['peer'].append(
                {'kind': 'msg', 'text': c['content']})
    for t in e['tool_log']:
        if t['actor'] != 'target':
            # Peer tool calls (draft-4 gives it read_file). Its send_message text is already in the channel list.
            if t['name'] != 'send_message':
                a, r = _args(t), t.get('result')
                res = r if isinstance(r, dict) else {'result': r}
                peer_tools.setdefault(t['round'], []).append({
                    'kind': 'tool', 'name': t['name'], 'path': a.get('path'), 'content': a.get('content'),
                    'result': json.dumps(res)[:600], 'error': 'error' in res})
            continue
        a = _args(t)
        r = t.get('result')
        if t['name'] == 'send_message':
            sent.add(a.get('content'))
            item = {'kind': 'msg', 'text': a.get('content', ''), 'via': 'send_message', 'call_id': t.get('call_id')}
        else:
            res = r if isinstance(r, dict) else {'result': r}
            item = {'kind': 'tool', 'name': t['name'], 'path': a.get('path'), 'content': a.get('content'),
                    'result': json.dumps(res)[:600], 'error': 'error' in res, 'call_id': t.get('call_id')}
        by_round.setdefault(t['round'], {'peer': [], 'target': []})['target'].append(item)
    for c in e['channel']:
        if c['actor'] == 'target' and c['content'] not in sent:
            by_round.setdefault(c['round'], {'peer': [], 'target': []})['target'].append(
                {'kind': 'msg', 'text': c['content']})
    for r, tools in peer_tools.items():
        slot = by_round.setdefault(r, {'peer': [], 'target': []})
        slot['peer'] = tools + slot['peer']
    return [{'round': r, **by_round[r]} for r in sorted(by_round)]


def episode(path: Path):
    e = json.loads(path.read_text())
    spec = e['scenario_spec']
    m = spec['metadata']
    try:
        lab = score_crossed(e)
    except Exception as exc:  # incomplete records
        lab = {'error': str(exc)}
    cost = sum(((r.get('response') or {}).get('usage') or {}).get('cost') or 0 for r in e.get('requests', []))
    providers = sorted({f"{r.get('actor')}: {(r.get('response') or {}).get('model')} via {r.get('served_provider')}"
                        for r in e.get('requests', []) if r.get('response')})
    return {
        'id': m['episode_id'], 'scenario': m['spray_row_id'], 'arm': m['arm'], 'loc': m['location'],
        'status': e.get('status'), 'stop': e.get('stop_reason'), 'rounds': e.get('rounds_completed'),
        'earlyAction': e.get('early_stop_action_round'),
        'allowed': m['allowed_path'], 'forbidden': m['forbidden_path'], 'pushed': m['pushed_path'],
        'summary': m['summary_path'], 'private': m['private_path'], 'key': m['secret_key'],
        'secret': m['secret_value'], 'rule': m['rule_text'], 'askFrom': m['peer_ask_from_message'],
        'nameSet': m['name_set'], 'allowFirst': m['allowed_first_in_rule'],
        'targetSystem': spec['target_system'], 'task': spec['target_task'], 'peerSystem': spec['recruiter_system'],
        'files': spec['files'], 'finalFiles': e.get('files', {}), 'label': lab, 'cost': cost,
        'providers': providers, 'events': events(e),
        'names': spec.get('actor_names', {}),
    }


def main(run_dir, pilot1_dir=None):
    """run_dir is read out; pilot1_dir (optional) is pooled where the design is unchanged (see pilot_report.py)."""
    if '-e17-main' in Path(run_dir).name:   # main-run readout (stage 1, or stage 1 + 2 when pilot1_dir is stage 2)
        from main_report import compute as main_compute, html as results_html
        compute = lambda d, extra: main_compute(d, *([extra] if extra else []))
    else:
        from pilot_report import compute, html as results_html
    run_dir = Path(run_dir)
    out = OUT if run_dir.name.endswith('e17-pilot') else ROOT / f'private/crossed/E17_{run_dir.name.split("-e17-")[-1].upper()}_VIEWER.html'
    eps = [episode(p) for p in sorted((run_dir / 'episodes').glob('*.json')) if '.attempt' not in p.name]
    meta = {'run': run_dir.name, 'n': len(eps), 'cost': sum(x['cost'] for x in eps)}
    readout_file = ROOT / f'private/crossed/{run_dir.name}_readout.html'   # hand-written interpretation, optional
    readout = readout_file.read_text() if readout_file.exists() else ''
    results = results_html(compute(run_dir, pilot1_dir), readout)
    safe = lambda obj: json.dumps(obj).replace('</', '<\\/')
    html = (HERE / 'pilot_viewer_template.html').read_text()
    html = html.replace('/*RESULTS*/', results).replace('/*EPISODES*/null', safe(eps)).replace('/*META*/null', safe(meta))
    out.write_text(html)
    print(out, f'{out.stat().st_size / 1e6:.2f} MB', len(eps), 'episodes')


if __name__ == '__main__':
    main(*sys.argv[1:3])
