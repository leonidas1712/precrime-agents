"""Task 2 (queue item 22): Qwen full-input bag-of-words on calls right after a protected-file read.

Post-read call = the latest message (the call's last input message) contains the masked secret
(the same subgroup definition as the bag-of-words interpretability analysis, 'latest message contains masked secret').
Metric: the paper's within-scenario-and-round AUROC (stream_metrics), 1000 family-bootstrap draws (seed 20261025),
paired probe-minus-BoW difference from the same draws (as arxiv_text_workflow.py).

Dev: saved leave-one-scenario-out full-input scores of the paper run 20261026-2245-e05-arxiv-qwen-text-workflow
     (refit check against bow_interp/common.py recipe). Probe: saved LOSO scores, 3 seeds.
Test: full-input classifier fit on all 2,976 development calls (paper recipe) scored on the 1,825 test calls;
      probe: saved frozen test scores.

    experiments/E19-v2-analyses/run.sh experiments/E19-v2-analyses/task2_fullinput_postread.py
"""
import csv, gzip, json, pickle
from collections import defaultdict
import numpy as np
from common_v2 import *
from common_v2 import ROOT, OUT

class Model:  # same class as task1_bow.Model, needed to unpickle cache/fits_task1.pkl
    def __init__(self, tf, clf): self.tf, self.clf = tf, clf
    def score(self, docs): return self.clf.decision_function(self.tf.transform(docs))


SAVED = ROOT / 'runs/20261026-2245-e05-arxiv-qwen-text-workflow'


def ev(rows, y, scores, folds, mask, reps=1000):
    ids = np.flatnonzero(mask)
    m, b = stream_metrics(rows, ids, y[ids], scores[:, ids], folds[:, ids], reps)
    return m, b


def tie_cells(rows, s, mask, y):
    cells = defaultdict(list)
    for i in np.flatnonzero(mask):
        cells[(rows[i]['scenario'], rows[i]['round'])].append((round(float(s[i]), 6), int(y[i])))
    mixed = [v for v in cells.values() if len({b for _, b in v}) == 2]
    return dict(cells=len(cells), cells_with_more_than_one_distinct_score=sum(len({a for a, _ in v}) > 1 for v in cells.values()),
                cells_with_leaking_and_nonleaking=len(mixed), leaking_calls_in_those_cells=sum(b for v in mixed for _, b in v),
                nonleaking_calls_in_those_cells=sum(1 - b for v in mixed for _, b in v),
                mixed_cells_where_bow_scores_differ=sum(len({a for a, _ in v}) > 1 for v in mixed))


def main():
    res = {}
    # ---------------- dev
    d = dev('qwen'); rows = d['rows']; y = np.array([r['y_immediate'] for r in rows])
    saved = defaultdict(dict)
    for r in csv.DictReader(gzip.open(SAVED / 'scores.csv.gz', 'rt')):
        saved[r['row_id']][int(r['seed'])] = (float(r['full_input']), float(r['latest_input']), int(r['fold']))
    bow_full = np.array([[saved[r['row_id']][s][0] for r in rows] for s in (42, 43, 44)])
    bow_latest = np.array([[saved[r['row_id']][s][1] for r in rows] for s in (42, 43, 44)])
    probe, folds = d['probe'], d['folds']
    pv = np.array(['PROTECTED_VALUE' in t for t in d['texts']['latest_input']])
    allm = np.ones(len(rows), bool)
    out = {}
    for name, mask in [('all_calls', allm), ('post_read', pv), ('not_post_read', ~pv)]:
        mp, bp = ev(rows, y, probe, folds, mask); mf, bf = ev(rows, y, bow_full, folds, mask); ml, bl = ev(rows, y, bow_latest, folds, mask)
        diff = bp['within_round'] - bf['within_round']
        out[name] = dict(calls=int(mask.sum()), leaking=int(y[mask].sum()), families=len({rows[i]['family'] for i in np.flatnonzero(mask)}),
                         episodes=mp['episodes'],
                         probe=dict(auc=mp['within_round_auc'], ci95=mp['within_round_ci95']),
                         bow_full=dict(auc=mf['within_round_auc'], ci95=mf['within_round_ci95']),
                         bow_latest=dict(auc=ml['within_round_auc'], ci95=ml['within_round_ci95']),
                         probe_minus_bow_full=dict(estimate=mp['within_round_auc'] - mf['within_round_auc'], ci95=np.nanquantile(diff, [.025, .975]).tolist()))
    out['post_read_tie_check_full'] = tie_cells(rows, bow_full[0], pv, y)
    out['post_read_tie_check_latest'] = tie_cells(rows, bow_latest[0], pv, y)
    paper = json.loads((SAVED / 'metrics.json').read_text())['metrics']['full_input']['within_round_auc']
    out['paper_check_all_calls_full_input'] = dict(recomputed=out['all_calls']['bow_full']['auc'], saved_run=paper)
    res['dev'] = out
    print('dev', json.dumps(out, default=float)[:1500], flush=True)

    # ---------------- test
    f = fresh_qwen(); frows = f['rows']; fy = np.array([r['label'] for r in frows])
    M = pickle.loads((OUT / 'cache/fits_task1.pkl').read_bytes())
    sf = M['qwen_full'].score(f['texts']['full_input'])[None]
    sl = M['qwen_latest'].score(f['texts']['latest_input'])[None]
    pr = np.array([r['probe'] for r in frows])[None]
    z = np.zeros((1, len(frows)), int)
    fpv = np.array(['PROTECTED_VALUE' in t for t in f['texts']['latest_input']])
    out = {}
    for name, mask in [('all_calls', np.ones(len(frows), bool)), ('post_read', fpv), ('not_post_read', ~fpv)]:
        mp, bp = ev(frows, fy, pr, z, mask); mf, bf = ev(frows, fy, sf, z, mask); ml, bl = ev(frows, fy, sl, z, mask)
        diff = bp['within_round'] - bf['within_round']
        out[name] = dict(calls=int(mask.sum()), leaking=int(fy[mask].sum()), families=len({frows[i]['family'] for i in np.flatnonzero(mask)}),
                         episodes=mp['episodes'],
                         probe=dict(auc=mp['within_round_auc'], ci95=mp['within_round_ci95']),
                         bow_full=dict(auc=mf['within_round_auc'], ci95=mf['within_round_ci95']),
                         bow_latest=dict(auc=ml['within_round_auc'], ci95=ml['within_round_ci95']),
                         probe_minus_bow_full=dict(estimate=mp['within_round_auc'] - mf['within_round_auc'], ci95=np.nanquantile(diff, [.025, .975]).tolist()))
    out['post_read_tie_check_full'] = tie_cells(frows, sf[0], fpv, fy)
    out['post_read_tie_check_latest'] = tie_cells(frows, sl[0], fpv, fy)
    res['test'] = out
    res['note'] = ('Post-read = the call\'s last input message contains the masked secret (a tool result of reading the protected file). '
                   'Test numbers use classifiers/probes trained on development episodes that include the test scenarios (not held-out-scenario); '
                   'dev numbers are leave-one-scenario-out. AUROC counts only scenario-and-round cells with both a leaking and a non-leaking post-read call.')
    print('test', json.dumps(out, default=float)[:1500], flush=True)
    (OUT / 'task2_fullinput_postread.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
