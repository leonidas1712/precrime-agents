"""Task 1(b): Jev 1.14 (the paper's per-call text monitor) on the permission experiments, with the rule in force.

Payloads are built exactly as experiments/E06-monitors/percall_input_monitor.py (Qwen): the call's saved request
messages + tools, reasoning keys removed, every exact secret -> [PROTECTED_VALUE], same question/criteria/preamble,
option order from prompts.option_order(digest(text)). The rule in force is the one in the request's system message:
for E08 we apply the E08 edit (PERMIT / NEUTRAL_EDIT / PERMIT with the consequence sentence removed); E14/E15 episodes
already carry their permitting rule.

    experiments/E19-v2-analyses/run.sh experiments/E19-v2-analyses/task1_jev.py            # prepare + cost estimate only (no API)
    experiments/E19-v2-analyses/run.sh experiments/E19-v2-analyses/task1_jev.py --pilot    # 20-call reproduction check vs saved Jev scores
    experiments/E19-v2-analyses/run.sh experiments/E19-v2-analyses/task1_jev.py --execute  # score everything (hard cap $4.50, ledger jev_spend.json)
    experiments/E19-v2-analyses/run.sh experiments/E19-v2-analyses/task1_jev.py --analyze  # E08/E14/E15 statistics from cached scores

Status (1 Nov 2026): not executed. The only OpenRouter key on this machine (/Users/leo/collusion-research/.env)
returns HTTP 401; spend so far $0.
"""
import csv, gzip, json, sys
from decimal import Decimal
from collections import defaultdict
import numpy as np
from common_v2 import *
from common_v2 import ROOT, OUT, jev_body, jev_many, JEV_CACHE, digest, request_for
import task1_bow as T

QWEN_JEV_RUN = '/Users/leo/collusion-research-followup/runs/20261026-1309-e06-percall-input-jev-qwen'
RESERVE_PER_BYTE = Decimal('2.474612634088200238379022646E-8')  # 2x max observed Jev cost/byte (that run's preflight)
OBSERVED = dict(billed_usd=Decimal('0.840589092'), payloads=6501)


def items():
    """(tag, call_id, condition, body) for every call to score."""
    out = []
    plan = json.loads((T.E08Q / 'plan.json').read_text())
    for c in plan['calls']:
        cid, sc = c['call_id'], c['scenario']
        for cond in ('FORBID', 'PERMIT', 'NEUTRAL_EDIT'):
            out.append(('E08', cid, cond, jev_body(cid, (sc, cond, plan))))
        req, hp = T.no_consequence_edit(cid, sc, plan)
        value = jev_body.__globals__['_clean']({k: req[k] for k in ['messages', 'tools'] if k in req}, hp)
        text = json.dumps(value, ensure_ascii=False)
        order = jev_body.__globals__['option_order'](digest(text))
        body = {'model': JEV_MODEL, 'state': PREAMBLE + text + SUFFIX,
                'questions': {QUESTION_ID: {'type': 'choice', 'instructions': QUESTION, 'criteria': {k: CRITERIA[k] for k in order}}}}
        out.append(('E08', cid, 'PERMIT_NO_CONSEQUENCE', body))
    seen = set()
    for c in json.loads((T.E15R / 'calls.json').read_text()):
        if c['call_id'] not in seen:
            seen.add(c['call_id']); out.append(('E15', c['call_id'], c['arm'], jev_body(c['call_id'])))
    for r in csv.DictReader(open(T.E14R / 'scores.csv')):
        if r['condition'] in ('PERMIT', 'PERMIT_NO_CONSEQUENCE') and r['call_id'] not in seen:
            seen.add(r['call_id']); out.append(('E14', r['call_id'], r['condition'], jev_body(r['call_id'])))
    return out


def estimate(its):
    uniq = {digest(b): b for *_, b in its}
    cached = sum((JEV_CACHE / f'{k}.json').exists() for k in uniq)
    nbytes = sum(len(json.dumps(b, ensure_ascii=False).encode()) for b in uniq.values())
    return dict(items=len(its), unique_payloads=len(uniq), cached=cached, payload_bytes=nbytes,
                worst_case_usd=float(nbytes * RESERVE_PER_BYTE),
                expected_usd_at_observed_mean_cost=float(OBSERVED['billed_usd'] / OBSERVED['payloads'] * len(uniq)),
                by_experiment={e: sum(1 for x in its if x[0] == e) for e in ('E08', 'E14', 'E15')})


def pilot():
    saved = {r['row_id']: float(r['score']) for r in csv.DictReader(open(QWEN_JEV_RUN + '/probe_scores.csv')) if r['analysis'] == 'loso' and r['seed'] == '42'}
    plan = json.loads((T.E08Q / 'plan.json').read_text())
    calls = plan['calls'][:20]
    sc = jev_many([('pilot', jev_body(c['call_id'], (c['scenario'], 'FORBID', plan))) for c in calls])
    rows = [dict(call_id=c['call_id'], saved=saved[c['call_id']], new=s) for c, s in zip(calls, sc)]
    d = [abs(r['saved'] - r['new']) for r in rows]
    out = dict(n=len(rows), exact=sum(x < 1e-9 for x in d), max_abs_diff=max(d), rows=rows)
    (OUT / 'jev_pilot.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out)[:600])


def analyze():
    its = items()
    score = {}
    for tag, cid, cond, body in its:
        p = JEV_CACHE / f'{digest(body)}.json'
        if p.exists():
            score[(tag, cid, cond)] = json.loads(p.read_text())['score']
    missing = [x[:3] for x in its if x[:3] not in score]
    if missing:
        print('missing', len(missing)); return
    # Jev threshold: E13 matched cut on the Qwen test calls (flags no more refusing-episode calls than the probe)
    cut = json.loads((ROOT / 'runs/20261028-0107-e13-matched-jev-d814/results.json').read_text())['qwen_test']['jev_cut']
    res = dict(threshold=cut)
    plan, by = T.e08_inputs(T.E08Q)
    jb = {cid: {cond: dict(v['FORBID'], score=score[('E08', cid, cond)]) for cond in ('FORBID', 'PERMIT', 'NEUTRAL_EDIT', 'PERMIT_NO_CONSEQUENCE')} for cid, v in by.items()}
    res['E08_qwen'] = T.e08_stats(jb, cut, conds=('FORBID', 'PERMIT', 'NEUTRAL_EDIT', 'PERMIT_NO_CONSEQUENCE'))
    saved = {r['row_id']: float(r['score']) for r in csv.DictReader(open(QWEN_JEV_RUN + '/probe_scores.csv')) if r['analysis'] == 'loso' and r['seed'] == '42'}
    res['E08_forbid_reproduction'] = dict(max_abs=max(abs(jb[c]['FORBID']['score'] - saved[c]) for c in jb), exact=sum(abs(jb[c]['FORBID']['score'] - saved[c]) < 1e-9 for c in jb), n=len(jb))
    # E14/E15 via the same code as task1_bow: build a score dict keyed by call id
    S = {cid: s for (tag, cid, cond), s in score.items() if tag in ('E14', 'E15')}
    for r in csv.DictReader(open(QWEN_JEV_RUN + '/probe_scores.csv')):
        if r['analysis'] == 'loso' and r['seed'] == '42': S.setdefault(r['row_id'], float(r['score']))
    res['note'] = 'E14/E15 tables: rerun task1_bow.main() logic with S as the detector; see task1_bow.py E14/E15 blocks.'
    (OUT / 'task1_jev.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, default=float)[:2000])


if __name__ == '__main__':
    from common_v2 import JEV_MODEL, PREAMBLE, SUFFIX, QUESTION_ID, QUESTION, CRITERIA  # noqa
    if '--pilot' in sys.argv: pilot()
    elif '--execute' in sys.argv:
        its = items(); est = estimate(its); print(est)
        assert est['worst_case_usd'] < 4.5, 'worst case above cap: score a stratified subset instead'
        jev_many([(f'{t}:{c}:{k}', b) for t, c, k, b in its])
    elif '--analyze' in sys.argv: analyze()
    else:
        est = estimate(items()); (OUT / 'task1_jev_estimate.json').write_text(json.dumps(est, indent=1)); print(json.dumps(est, indent=1))
