"""Shared loaders: rebuild the exact rows, texts, folds and fits used for the paper's text-classifier numbers.

Imports the paper code (arxiv_text_workflow.source_data/texts_flags, metrics.position_weights,
reasoning_stream_report.stream_metrics, e12_cascade.load) so data, masking and views are identical.
"""
import json, pickle, sys, warnings
from pathlib import Path
import numpy as np
ROOT = Path('/Users/leo/precrime-agents')
sys.path[:0] = [str(ROOT/'experiments/E05-probes'), str(ROOT/'src'), str(ROOT/'experiments/E03-target-screen'), str(ROOT/'experiments/E12-cascade')]
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from arxiv_text_workflow import source_data, texts_flags
from metrics import position_weights
from reasoning_stream_report import stream_metrics
import e12_cascade
RESTORE = Path('/Users/leo/collusion-research')  # read-only; same files as the paper's restore root
# The paper runs used /Users/leo/collusion-research-followup as ROOT (see run meta.json 'command').
# Some run inputs (Qwen rows.json, plans.json, JEV scores) exist only there; code files are byte-identical.
FOLLOW = Path('/Users/leo/collusion-research-followup')
import arxiv_text_workflow as _atw
_atw.ROOT = FOLLOW
e12_cascade.ROOT = FOLLOW
e12_cascade.PROBE = FOLLOW/'runs/20261026-2212-e05-arxiv-qwen-fresh-percall'
e12_cascade.JEV = FOLLOW/'runs/20261026-1309-e06-percall-input-jev-qwen/probe_scores.csv'
e12_cascade.EPISODE_LABELS = FOLLOW/'runs/20261026-2155-e05-qwen-any-public-audit/rows.json'
OUT = Path(__file__).resolve().parent; CACHE = OUT/'cache'


def vectorizer(**kw):
    # Exactly the paper recipe (arxiv_text_workflow.py / e12_cascade.add_bow).
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True, **kw)


def classifier():
    return LogisticRegression(C=.1, solver='lbfgs', max_iter=2000, tol=1e-5, random_state=42)


def fit(docs, y, w, **kw):
    tf = vectorizer(**kw); x = tf.fit_transform(docs); clf = classifier()
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning); clf.fit(x, y, sample_weight=w)
    return tf, clf


def dev(model):
    """Development rows, both text views, LOSO folds (as arxiv_text_workflow.main)."""
    p = CACHE/f'dev_{model}.pkl'
    if p.exists(): return pickle.loads(p.read_bytes())
    rows, probe, folds, source = source_data(model, RESTORE)
    texts, flags, _ = texts_flags(rows, RESTORE, model)
    d = dict(rows=rows, probe=probe, folds=folds, texts=texts)
    p.write_bytes(pickle.dumps(d)); return d


def loso_train_ids(model, d, fold):
    rows, folds = d['rows'], d['folds']; families = np.array([r['family'] for r in rows])
    te = np.flatnonzero(folds[0] == fold); held = set(families[te])
    tr = np.array([i for i in range(len(rows)) if folds[0, i] != fold and families[i] not in held])
    if model == 'qwen':  # same override as arxiv_text_workflow.main
        saved = json.loads((FOLLOW/'runs/20261026-2201-e05-arxiv-qwen-any-public/rows.json').read_text())
        plans = [p for p in json.loads((FOLLOW/'runs/20261026-2201-e05-arxiv-qwen-any-public/plans.json').read_text()) if p['fold'] == fold]
        keys = {f"{saved[j]['episode']}::request_idx={saved[j]['request_idx']}" for j in plans[0]['split']['train']}
        tr = np.array([i for i, r in enumerate(rows) if r['row_id'] in keys])
    return tr, te


def fresh_qwen():
    """Qwen test-episode rows and latest-message texts (as e12_cascade.add_bow + bow_test_auroc)."""
    p = CACHE/'fresh_qwen.pkl'
    if p.exists(): return pickle.loads(p.read_bytes())
    import csv, gzip
    rows = e12_cascade.load()
    meta_ep = {f"{r['episode']}::request_idx={r['request_idx']}": r for r in json.loads(e12_cascade.EPISODE_LABELS.read_text())}
    fresh = [dict(meta_ep[r['row_id']], row_id=r['row_id']) for r in rows]
    texts = texts_flags(fresh, RESTORE, 'qwen')[0]
    with gzip.open(e12_cascade.PROBE/'probe_scores.csv.gz', 'rt') as f:
        meta = {r['row_id']: r for r in csv.DictReader(f)}
    for r in rows:
        m = meta[r['row_id']]; r.update(round=int(m['round']), scenario=m['scenario'], family=m['family'])
    d = dict(rows=rows, texts=texts)
    p.write_bytes(pickle.dumps(d)); return d


def auroc(rows, y, scores):
    """Paper metric: within scenario and round (stream_metrics), single score vector, no bootstrap."""
    ids = np.arange(len(rows)); s = np.asarray(scores, float)[None, :]
    return stream_metrics(rows, ids, y, s, np.zeros((1, len(rows)), int), 1)[0]['within_round_auc']
