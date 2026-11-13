"""Process-pool workers and summaries for development (held-out-scenario, loso) refits.

Workers call the original E05 fit functions unchanged; only C (via common.set_C) varies.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import common
from common import set_C, SEEDS

THREADS = 2
_GPT = None


def _with_retry(run, marker_dir, C):
    """Run with the recipe's max_iter=2000; if lbfgs does not converge, retry at 20000 and leave a marker."""
    set_C(C)
    try:
        run()
    except RuntimeError as e:
        if 'onverge' not in str(e):
            raise
        set_C(C, max_iter=20000)
        run()
        Path(marker_dir).mkdir(parents=True, exist_ok=True)
        (Path(marker_dir) / 'max_iter_20000.json').write_text(json.dumps(dict(C=C, reason=str(e))))


def gpt_job(job):
    out, label, plan, C = job
    import lastprompt_percall as lp
    global _GPT
    if _GPT is None:
        _GPT = common.gpt_dev()
    rows, base, x, _ = _GPT
    dest = Path(out) / 'matched' / label / plan['mode'] / f"s{plan['seed']}-f{plan['fold']}"
    with threadpool_limits(limits=THREADS):
        _with_retry(lambda: lp.run_fit(Path(out), rows, base, x, plan, label, 'matched'), dest, C)
    return str(dest)


def qwen_job(job):
    out, label, plan, C = job
    import arxiv_qwen_refit as core
    dest = Path(out) / label / plan['mode'] / f"s{plan['seed']}-f{plan['fold']}"
    _with_retry(lambda: core.fit((Path(out), label, plan)), dest, C)
    return str(dest)


def reasoning_job(job):
    out, plan, C = job
    import reasoning_stream_fit as rsf
    set_C(C)
    dest = Path(out) / 'immediate' / 'all' / plan['mode'] / f"s{plan['seed']}-f{plan['fold']}"
    _with_retry(lambda: rsf.run_job(str(common.GPT_STREAM_DATA), str(out), plan, 'immediate', False), dest, C)
    return str(dest)


def qwen_prepare(out):
    """Rows + analysis_plan.json that arxiv_qwen_refit.fit reads; plans = saved development loso plans."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rows = common.qwen_rows()
    refit_plan = json.loads((common.Q_REFIT / 'analysis_plan.json').read_text())
    assert common.sha(common.Q_REFIT / 'plans.json') == refit_plan['input_hashes']['plans.json']
    saved = json.loads((common.Q_REFIT / 'rows.json').read_text())
    assert [r['index'] for r in saved] == [r['index'] for r in rows]
    plans = [p for p in json.loads((common.Q_REFIT / 'plans.json').read_text()) if p['mode'] == 'loso']
    assert len(plans) == 18
    strict = []
    for p in plans:  # arxiv_qwen_strict_next.main
        strict.append({**p, 'split': {k: [i for i in v if not rows[i]['y_immediate']] for k, v in p['split'].items()}})
    common_plan = dict(feature_path=common.Q_FEATURES)
    (out / 'rows.json').write_text(json.dumps(rows))
    (out / 'analysis_plan.json').write_text(json.dumps(common_plan))
    return rows, plans, strict


def gpt_plans():
    plans = [p for p in json.loads((common.GPT_FITS / 'plans.json').read_text())
             if p['window'] == 'post_ask' and p['variant'] == 'eventual' and p['mode'] == 'loso']
    assert len(plans) == 18
    return plans


def _collect(files, n, score_key='scores'):
    values = np.full((3, n), np.nan); folds = np.full((3, n), -1); layers = {}; raised = []
    for path in files:
        f = json.loads(path.read_text()); p = f.get('plan', f); si = SEEDS.index(p['seed'])
        z = np.load(path.parent / 'scores.npz'); ids = z['indices']
        assert np.isnan(values[si, ids]).all()
        values[si, ids] = z[score_key]; folds[si, ids] = p['fold']; layers[f"{p['seed']}/{p['fold']}"] = int(f['layer'])
        if (path.parent / 'max_iter_20000.json').exists():
            raised.append(f"{p['seed']}/{p['fold']}")
    return values, folds, layers, raised


def summarize(rows, folder, label, expected_rows, reps=1000, score_key='scores', mode='loso'):
    from reasoning_stream_report import stream_metrics
    files = sorted(Path(folder).glob(f'{mode}/s*-f*/fit.json'))
    assert len(files) == 18, (folder, len(files))
    values, folds, layers, raised = _collect(files, len(rows), score_key)
    ids = np.flatnonzero(np.isfinite(values).all(axis=0))
    y = np.array([rows[i]['y_' + label] for i in ids])
    if score_key == 'max_scores':  # reasoning summarize: post-ask only (drops pre-ask placebo rows)
        keep = np.array([rows[i]['stream_window'] == 'post_ask' for i in ids]); ids, y = ids[keep], y[keep]
    assert len(ids) == expected_rows, (folder, len(ids), expected_rows)
    m, _ = stream_metrics(rows, ids, y, values[:, ids], folds[:, ids], reps)
    out = common.summary(m)
    out.update(common.strata_counts(rows, ids, y), selected_layers=layers, max_iter_raised=raised)
    return out
