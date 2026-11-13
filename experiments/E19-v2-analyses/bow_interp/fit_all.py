"""Step 1-2: refit the paper's three text classifiers exactly, check AUROC against the saved runs, save vocab+coef+scores."""
import csv, gzip, json, pickle
import numpy as np
from common import *

SAVED = {'gpt': ROOT/'runs/20261026-2242-e05-arxiv-gpt-text-workflow', 'qwen': ROOT/'runs/20261026-2245-e05-arxiv-qwen-text-workflow'}
VIEW = {'gpt': 'full_input', 'qwen': 'latest_input'}


def loso(model, view):
    d = dev(model); rows = d['rows']; docs = d['texts'][view]
    y = np.array([r['y_immediate'] for r in rows]); rounds = np.array([r['round'] for r in rows])
    pred = np.full(len(rows), np.nan); fits = []
    for fold in sorted(set(d['folds'][0])):
        tr, te = loso_train_ids(model, d, fold)
        w = position_weights(y[tr], rounds[tr]); ids = tr[w > 0]; w = w[w > 0]
        tf, clf = fit([docs[i] for i in ids], y[ids], w)
        pred[te] = clf.decision_function(tf.transform([docs[i] for i in te]))
        fits.append(dict(fold=int(fold), held_scenario=sorted({rows[i]['scenario'] for i in te}), test=te, train=ids,
                         features=tf.get_feature_names_out(), coef=clf.coef_[0].copy(), intercept=float(clf.intercept_[0]), idf=tf.idf_.copy()))
    probe = np.tile(pred, (3, 1)); ids = np.arange(len(rows))
    auc = stream_metrics(rows, ids, y, probe, d['folds'], 1)[0]['within_round_auc']
    return dict(pred=pred, fits=fits, auc=auc, y=y)


def saved_scores(model, view):
    with gzip.open(SAVED[model]/'scores.csv.gz', 'rt') as f:
        return {r['row_id']: float(r[view]) for r in csv.DictReader(f) if r['seed'] == '42'}


def e12():
    d = dev('qwen'); rows = d['rows']; docs = d['texts']['latest_input']
    y = np.array([r['y_immediate'] for r in rows]); rounds = np.array([r['round'] for r in rows])
    w = position_weights(y, rounds); keep = w > 0
    tf, clf = fit([t for t, k in zip(docs, keep) if k], y[keep], w[keep])
    f = fresh_qwen(); fy = np.array([r['label'] for r in f['rows']])
    s = clf.decision_function(tf.transform(f['texts']['latest_input']))
    return dict(fit=dict(features=tf.get_feature_names_out(), coef=clf.coef_[0].copy(), intercept=float(clf.intercept_[0]), idf=tf.idf_.copy(), train=np.flatnonzero(keep)),
                pred=s, y=fy, auc=auroc(f['rows'], fy, s))


def gpt_alldev():
    """Reference only (not in the paper): gpt full-input fit on all dev calls, to compare with fold-averaged weights."""
    d = dev('gpt'); rows = d['rows']; docs = d['texts']['full_input']
    y = np.array([r['y_immediate'] for r in rows]); rounds = np.array([r['round'] for r in rows])
    w = position_weights(y, rounds); keep = w > 0
    tf, clf = fit([t for t, k in zip(docs, keep) if k], y[keep], w[keep])
    return dict(features=tf.get_feature_names_out(), coef=clf.coef_[0].copy(), intercept=float(clf.intercept_[0]), idf=tf.idf_.copy())


if __name__ == '__main__':
    out = {}; check = {}
    for model in ['gpt', 'qwen']:
        view = VIEW[model]; r = loso(model, view); rows = dev(model)['rows']
        saved = saved_scores(model, view); diff = max(abs(saved[row['row_id']] - p) for row, p in zip(rows, r['pred']))
        reported = json.loads((SAVED[model]/'metrics.json').read_text())['metrics'][view]['within_round_auc']
        check[f'{model}_dev_{view}'] = dict(reproduced_auc=r['auc'], saved_run_auc=reported, max_abs_score_diff_vs_saved=diff, calls=len(rows), leaking=int(r['y'].sum()))
        out[f'{model}_loso'] = r; print(model, check[f'{model}_dev_{view}'], flush=True)
    r = e12()
    saved = {row['row_id']: float(row['bow']) for row in csv.DictReader(gzip.open(ROOT/'runs/20261028-0228-e12-bow-test-auroc/bow_scores.csv.gz', 'rt'))}
    diff = max(abs(saved[row['row_id']] - p) for row, p in zip(fresh_qwen()['rows'], r['pred']))
    reported = json.loads((ROOT/'runs/20261028-0228-e12-bow-test-auroc/metrics.json').read_text())['bow']['within_round_auc']
    check['qwen_test_latest_input'] = dict(reproduced_auc=r['auc'], saved_run_auc=reported, max_abs_score_diff_vs_saved=diff, calls=len(r['y']), leaking=int(r['y'].sum()))
    print(check['qwen_test_latest_input']); out['qwen_e12'] = r
    out['gpt_alldev'] = gpt_alldev()
    (CACHE/'fits.pkl').write_bytes(pickle.dumps(out))
    (OUT/'reproduction.json').write_text(json.dumps(check, indent=1, default=float))
