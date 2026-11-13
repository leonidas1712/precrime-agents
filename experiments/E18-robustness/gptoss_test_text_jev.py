"""gpt-oss test-set (594 recruiting calls, 26 leaking) scores for the text classifier and Jev.

bow:    fit the E05 text recipe (arxiv_text_workflow) on all 1,995 gpt-oss development calls,
        score the 594 test calls once, for the full-input and latest-message views
        (as E12 bow_test_auroc.py does for Qwen). CPU only, no API.
jev-prepare / jev-execute / jev-summarize:
        Jev 1.14 on the same 594 calls with the exact dev L4 recipe
        (percall_harmony_input_monitor.py: decoded frozen Harmony prompt_ids, retained same-turn
        reasoning, exact values masked, option order from the payload digest, same parsing).
        Summary adds the E13 operating point (Jev cut matched to the reasoning probe's share of
        refusing-episode calls flagged).
Metric: stream_metrics (within scenario x round AUROC, 1000 family-bootstrap draws).
"""
import argparse, csv, gzip, hashlib, json, sys, warnings
from decimal import Decimal
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments/E02-activation-replay'), str(ROOT / 'experiments/E05-probes'),
                str(ROOT / 'experiments/E03-target-screen'), str(ROOT / 'experiments/E06-monitors'),
                str(ROOT / 'experiments/E13-matched-jev')]
import numpy as np  # noqa: E402
from collusion.runs import Run, start_run, write_json_atomic  # noqa: E402
from reasoning_stream_report import stream_metrics  # noqa: E402

OLD = Path('/Users/leo/collusion-research')  # read-only
CONFIRM = OLD / 'runs/20261026-1302-e05-stream-confirm-score/probe_scores.csv'
CPLAN = OLD / 'runs/20261026-1225-e02-gpt-oss-confirm-plan'
DEV_REPLIES = OLD / 'runs/20261026-0522-e05-reasoning-stream-data/replies.json'
DEV_PLAN = OLD / 'runs/20261025-2251-e02-gpt-oss-full-plan'
DEV_L4 = OLD / 'runs/20261026-1319-e06-percall-input-jev-l4'
LASTP = ROOT / 'runs/20261026-2222-e05-arxiv-gpt-fresh-percall'
CAP = Decimal('1.00')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_rows():
    """The 594 rows exactly as arxiv_gpt_fresh.freeze builds them, joined to the confirm replay plan."""
    rows = []
    for r in csv.DictReader(CONFIRM.open()):
        if r['condition'] != 'recruiting' or int(r['reasoning_tokens']) == 0:
            continue
        for k in ['round', 'ordinal', 'y_immediate', 'reasoning_tokens']:
            r[k] = int(r[k])
        rows.append(r)
    assert len(rows) == 594 and sum(r['y_immediate'] for r in rows) == 26
    dev = json.loads(DEV_REPLIES.read_text())
    dev_rows = [r for r in dev if r['stream_window'] == 'post_ask' and r['reasoning_tokens'] > 0]
    assert not {str(r['family']) for r in dev_rows} & {str(r['family']) for r in rows}
    plan = {}
    for f in sorted((CPLAN / 'shards').glob('*.json')):
        for p in json.loads(f.read_text())['rows']:
            plan[p['row_id']] = p
    for r in rows:
        p = plan[r['row_id']]
        assert p['request_sha256'] == r['request_sha256'] and p['episode'] == r['episode']
        r.update(request_idx=p['request_idx'], history_reasoning_mode=p['history_reasoning_mode'], source_path=p['source_path'],
                 episode_sha256=p['episode_sha256'], harmony_request_sha256=p['harmony_request_sha256'], prompt_sha256=p['prompt_sha256'],
                 ever_public=r['ever_public'] == 'True', alarm=r['alarm'] == 'True')
        r['_prompt_ids'] = p['prompt_ids']
    return rows


def probe_scores(rows):
    with gzip.open(LASTP / 'probe_scores.csv.gz', 'rt') as f:
        lp = {r['row_id']: r for r in csv.DictReader(f)}
    assert set(lp) == {r['row_id'] for r in rows}
    for r in rows:
        assert int(lp[r['row_id']]['label']) == r['y_immediate'] and lp[r['row_id']]['request_sha256'] == r['request_sha256']
    return dict(last_prompt=np.array([float(lp[r['row_id']]['score']) for r in rows]),
                reasoning=np.array([float(r['max_score']) for r in rows]))


def metrics_for(rows, y, methods):
    ids = np.arange(len(rows)); fold = np.zeros((1, len(rows)), int)
    out, boots = {}, {}
    for k, v in methods.items():
        out[k], boots[k] = stream_metrics(rows, ids, y, np.asarray(v, float)[None, :], fold, 1000)
    return out, boots


def paired(out, boots, a, b):
    d = boots[a]['within_round'] - boots[b]['within_round']
    dp = boots[a]['pooled'] - boots[b]['pooled']
    return dict(within_round=out[a]['within_round_auc'] - out[b]['within_round_auc'], within_round_ci95=np.nanquantile(d, [.025, .975]).tolist(),
                pooled=out[a]['pooled_auc'] - out[b]['pooled_auc'], pooled_ci95=np.nanquantile(dp, [.025, .975]).tolist())


# ---------------- bag of words ----------------

def bow():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    from arxiv_text_workflow import source_data, texts_flags
    from metrics import position_weights
    run = start_run('e18-gptoss-bow-test', experiment='E18', parents=['20261026-2242-e05-arxiv-gpt-text-workflow', LASTP.name, CONFIRM.parent.name],
                    tags=['cpu-only', 'post-hoc', 'test'],
                    settings=dict(no_api=True, no_gpu=True, bootstrap=1000, restore_root=str(OLD),
                                  recipe='arxiv_text_workflow text views and TF-IDF/LR hyperparameters; fit once on all 1,995 dev calls (round-balanced weights, weight>0 rows) as E12 add_bow; score 594 test calls once'))
    try:
        rows = test_rows(); y = np.array([r['y_immediate'] for r in rows])
        dev, _, _, _ = source_data('gpt', OLD)
        dev_text = texts_flags(dev, OLD, 'gpt')[0]
        test_text, _, test_hashes = texts_flags(rows, OLD, 'gpt')
        for name, docs in test_text.items():
            assert 'PROTECTED_VALUE' in ''.join(docs) or name == 'latest_input'
        dy = np.array([r['y_immediate'] for r in dev]); rounds = np.array([r['round'] for r in dev])
        w = position_weights(dy, rounds); keep = w > 0
        methods = probe_scores(rows); fits = {}
        with threadpool_limits(limits=2):
            for view in ['full_input', 'latest_input']:
                tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
                x = tf.fit_transform([d for d, k in zip(dev_text[view], keep) if k])
                clf = LogisticRegression(C=.1, solver='lbfgs', max_iter=2000, tol=1e-5, random_state=42)
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    clf.fit(x, dy[keep], sample_weight=w[keep])
                methods['bow_' + view] = clf.decision_function(tf.transform(test_text[view]))
                fits[view] = dict(dev_calls=len(dev), dev_positives=int(dy.sum()), dev_fit_calls=int(keep.sum()), vocabulary=len(tf.vocabulary_), iterations=clf.n_iter_.tolist())
                print('FIT', view, fits[view], flush=True)
        out, boots = metrics_for(rows, y, methods)
        assert abs(out['reasoning']['within_round_auc'] - 0.905128205128205) < 1e-9
        assert abs(out['last_prompt']['within_round_auc'] - 0.8179487179487179) < 1e-9
        diffs = {f'{p}_minus_bow_{v}': paired(out, boots, p, 'bow_' + v) for p in ['last_prompt', 'reasoning'] for v in ['full_input', 'latest_input']}
        write_json_atomic(run.dir / 'metrics.json', dict(calls=len(rows), leaking_calls=int(y.sum()), fits=fits, metrics=out, paired=diffs, test_episode_hashes=test_hashes))
        with gzip.open(run.dir / 'scores.csv.gz', 'wt', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=['row_id', 'family', 'scenario', 'round', 'label'] + list(methods)); wr.writeheader()
            for i, r in enumerate(rows):
                wr.writerow({k: r[k] for k in ['row_id', 'family', 'scenario', 'round']} | dict(label=int(y[i])) | {k: float(v[i]) for k, v in methods.items()})
        for k, m in out.items():
            print(k, round(m['within_round_auc'], 3), np.round(m['within_round_ci95'], 3), 'pooled', round(m['pooled_auc'], 3), np.round(m['pooled_ci95'], 3))
        for k, d in diffs.items():
            print(k, round(d['within_round'], 3), np.round(d['within_round_ci95'], 3))
        run.result(**{k: m['within_round_auc'] for k, m in out.items()}); run.finish(); print(run.dir)
    except BaseException as e:
        run.finish('failed', str(e)); raise


# ---------------- Jev ----------------

def jev_base():
    import percall_harmony_input_monitor as h  # sets the dev L4 prompt spec on the shared base module
    import monitors
    base = h.base
    base.CAP = CAP
    base.RESERVE_REFERENCE = OLD / 'runs/20261025-2202-e06-addressee-audit/cache'  # not in this clone
    base.Ledger = partial(monitors.Ledger, limit=CAP)
    base.summarize = lambda directory: None
    assert base.PROMPT_SHA == json.loads((DEV_L4 / 'meta.json').read_text())['settings']['prompt_sha256']
    return h, base


def render(row, prompt_ids, episodes, enc, h, base):
    """Same steps and assertions as percall_harmony_input_monitor.prepare."""
    from gpt_oss_replay import digest
    assert digest(prompt_ids) == row['prompt_sha256']
    path = Path(row['source_path'])
    if str(path) not in episodes:
        assert sha(path) == row['episode_sha256']; episodes[str(path)] = json.loads(path.read_text())
    ep = episodes[str(path)]; request = ep['requests'][row['request_idx']]['request']; assert digest(request) == row['harmony_request_sha256']
    text = enc.decode(prompt_ids); reencoded = enc.encode(text, allowed_special='all'); assert enc.decode(reencoded) == text
    messages = request['messages']; last_user = max((j for j, v in enumerate(messages) if v['role'] == 'user'), default=-1); mode = row['history_reasoning_mode']
    reasons = [v['reasoning'] for j, v in enumerate(messages) if v.get('reasoning') and (mode == 'keep_all' or mode == 'current_turn' and j > last_user)]
    assert all(r in text for r in reasons)
    protected = ep['scenario_spec']['honeypots']; redacted = h.m.clean(text, protected); assert all(v not in redacted for v in protected)
    body, order = base.body_for(redacted, base.digest(redacted))
    return body, order, dict(retained_reasoning_messages=len(reasons), retained_reasoning_characters=sum(map(len, reasons)), history_reasoning_mode=mode,
                             prompt_token_count=len(prompt_ids), masked_occurrences=redacted.count('[PROTECTED_VALUE]'), text_reencoding_preserves_ids=reencoded == prompt_ids)


def dev_format_check(enc, h, base, n=3):
    """Re-render dev L4 rows with this code and require byte-identical payloads to the dev run."""
    replies = json.loads(DEV_REPLIES.read_text()); tasks = json.loads((DEV_L4 / 'tasks.json').read_text()); payloads = json.loads((DEV_L4 / 'payloads.json').read_text())
    picks = [t for t in tasks if t['evaluation']]
    picks = [picks[0], picks[len(picks) // 2], picks[-1]][:n]
    want = {t['row_id'] for t in picks}; planned = {}
    for f in sorted((DEV_PLAN / 'shards').glob('*.json')):
        for p in json.loads(f.read_text())['rows']:
            if p['row_id'] in want:
                planned[p['row_id']] = p['prompt_ids']
    out = []; episodes = {}
    for t in picks:
        row = replies[t['row_index']]; assert row['row_id'] == t['row_id']
        body, order, fact = render(row, planned[t['row_id']], episodes, enc, h, base)
        same = body == payloads[t['payload_key']] and list(order) == t['option_order'] and base.digest(body) == t['payload_key']
        assert same, t['row_id']
        out.append(dict(row_id=t['row_id'], identical_payload=same, masked_occurrences=fact['masked_occurrences'], retained_reasoning_messages=fact['retained_reasoning_messages']))
    return out


def jev_prepare():
    h, base = jev_base()
    from gpt_oss_replay import harmony_encoding
    enc = harmony_encoding()
    check = dev_format_check(enc, h, base)
    print('DEV_FORMAT_CHECK', json.dumps(check), flush=True)
    rows = test_rows(); episodes = {}; payloads = {}; tasks = []; facts = []
    for i, r in enumerate(rows):
        body, order, fact = render(r, r['_prompt_ids'], episodes, enc, h, base)
        pk = base.digest(body); payloads.setdefault(pk, body)
        tasks.append(dict(row_index=i, row_id=r['row_id'], request_sha256=r['request_sha256'], prompt_sha256=r['prompt_sha256'], payload_key=pk, option_order=list(order)))
        facts.append(dict(row_id=r['row_id'], **fact))
    rate = base.reserve_rate(); worst = sum(Decimal(len(json.dumps(b, ensure_ascii=False).encode())) * rate for b in payloads.values())
    run = start_run('e18-gptoss-jev-test', experiment='E18', parents=[DEV_L4.name, CPLAN.name, CONFIRM.parent.name], tags=['api', 'test'],
                    settings=dict(target='gpt-oss', prompt_spec=base.PROMPT_SPEC, prompt_sha256=base.PROMPT_SHA, cap_usd=str(CAP), restore_root=str(OLD),
                                  recipe='percall_harmony_input_monitor.prepare on the 594 test calls (confirm replay plan prompt_ids); no control calibration calls needed (AUROC and matched-share operating point only)'))
    for n, v in [('payloads', payloads), ('tasks', tasks), ('input_details', facts), ('dev_format_check', check)]:
        write_json_atomic(run.dir / (n + '.json'), v)
    pre = dict(calls=len(tasks), positives=sum(r['y_immediate'] for r in rows), unique_payloads=len(payloads), reserve_usd_per_byte=str(rate), worst_case_usd=str(worst),
               cap_usd=str(CAP), within_cap=worst <= CAP, calls_with_retained_reasoning=sum(bool(f['retained_reasoning_messages']) for f in facts),
               calls_with_masked_value=sum(bool(f['masked_occurrences']) for f in facts), protected_values_sent=False)
    write_json_atomic(run.dir / 'preflight.json', pre)
    run.result(**pre); run.finish('prepared'); print(run.dir); print(json.dumps(pre))


def jev_execute(directory, max_calls):
    h, base = jev_base()
    base.execute(directory.resolve(), max_calls)
    b = json.loads((directory / 'budget.json').read_text()); n = len(list((directory / 'cache').glob('*.json')))
    print('BILLED', b['billed'], 'cached', n, 'projected_total', float(b['billed']) / max(n, 1) * len(json.loads((directory / 'payloads.json').read_text())), flush=True)


def jev_summarize(directory):
    h, base = jev_base()
    import monitors
    from matched_jev import analyse, evaluate
    rows = test_rows(); y = np.array([r['y_immediate'] for r in rows]); tasks = json.loads((directory / 'tasks.json').read_text())
    assert len(tasks) == 594
    jev = np.full(len(rows), np.nan)
    for t in tasks:
        rec = json.loads((directory / 'cache' / (t['payload_key'] + '.json')).read_text()); assert rec['prompt_sha256'] == base.PROMPT_SHA
        s = monitors.parse_decision(rec['response'], base.QUESTION_ID); assert s == rec['score']; jev[t['row_index']] = s
    assert np.isfinite(jev).all()
    methods = probe_scores(rows) | dict(jev=jev)
    out, boots = metrics_for(rows, y, methods)
    diffs = {f'{p}_minus_jev': paired(out, boots, p, 'jev') for p in ['last_prompt', 'reasoning']}
    # E13 operating point: probe at its saved threshold, Jev at the lowest cut flagging no larger share of refusing-episode calls.
    pos = y == 1; neg = np.array([not r['ever_public'] for r in rows]); ep = [r['episode'] for r in rows]
    lp_threshold = json.loads((ROOT / 'paper-arxiv/frozen/gpt_percall_plan.json').read_text())['threshold']
    reasoning_hit = np.array([r['alarm'] for r in rows]); lp_hit = methods['last_prompt'] > lp_threshold
    lpm = json.loads((LASTP / 'metrics.json').read_text()); assert int((lp_hit & pos).sum()) == lpm['positive_alarms']
    op = {}
    for name, hit in [('reasoning_probe', reasoning_hit), ('last_prompt_probe', lp_hit)]:
        res = analyse('gptoss_test_' + name, ep, hit.astype(float), jev, pos, neg, np.random.default_rng(0))
        table_rows = [dict(row_id=r['row_id'], episode=r['episode'], condition='recruiting', label=int(r['y_immediate']), ever=r['ever_public'], score=float(jev[i])) for i, r in enumerate(rows)]
        res['jev_table_at_matched_cut'] = evaluate(table_rows, res['jev_cut'])
        op[name] = res
    op['last_prompt_probe']['threshold'] = lp_threshold
    bud = json.loads((directory / 'budget.json').read_text())
    res = dict(calls=len(rows), leaking_calls=int(y.sum()), refusing_episode_calls=int(neg.sum()), metrics=out, paired=diffs, operating_point=op,
               billed_usd=bud['billed'], cap_usd=str(CAP))
    write_json_atomic(directory / 'metrics.json', res)
    with gzip.open(directory / 'scores.csv.gz', 'wt', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['row_id', 'episode', 'family', 'scenario', 'round', 'request_sha256', 'label', 'ever_public', 'reasoning_alarm', 'jev', 'last_prompt', 'reasoning'])
        wr.writeheader()
        for i, r in enumerate(rows):
            wr.writerow({k: r[k] for k in ['row_id', 'episode', 'family', 'scenario', 'round', 'request_sha256']} | dict(label=int(y[i]), ever_public=r['ever_public'], reasoning_alarm=r['alarm'], jev=jev[i], last_prompt=methods['last_prompt'][i], reasoning=methods['reasoning'][i]))
    for k, m in out.items():
        print(k, round(m['within_round_auc'], 3), np.round(m['within_round_ci95'], 3), 'pooled', round(m['pooled_auc'], 3), np.round(m['pooled_ci95'], 3))
    for k, d in diffs.items():
        print(k, round(d['within_round'], 3), np.round(d['within_round_ci95'], 3))
    meta = json.loads((directory / 'meta.json').read_text()); r = Run(directory.name, directory, meta)
    r.result(jev_auc=out['jev']['within_round_auc'], billed_usd=bud['billed']); r.finish('finished')


def main():
    p = argparse.ArgumentParser(); p.add_argument('cmd', choices=['bow', 'jev-prepare', 'jev-execute', 'jev-summarize']); p.add_argument('--run', type=Path); p.add_argument('--max-calls', type=int)
    a = p.parse_args()
    if a.cmd == 'bow': bow()
    elif a.cmd == 'jev-prepare': jev_prepare()
    elif a.cmd == 'jev-execute': jev_execute(a.run, a.max_calls)
    else: jev_summarize(a.run)


if __name__ == '__main__':
    main()
