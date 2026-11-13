"""Jev (zero-shot monitor) on E14 / E15, with the same statistics code as task1_bow.py's E14/E15 blocks.

Scores: cached Jev scores from task1_jev.py --execute (prompted with the rule actually in force).
Qwen test rows (forbidden dispatch leaks): the saved Jev scores in fresh_qwen()['rows'][i]['jev'].
Threshold: the E13 matched Jev cut (flags no larger share of refusing-episode test calls than the probe).

    experiments/E19-v2-analyses/run.sh experiments/E19-v2-analyses/task1_jev_e14e15.py
"""
import csv, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import task1_bow as T
import task1_jev as J
from common_v2 import ROOT, OUT, JEV_CACHE, digest
from common_v2 import JEV_MODEL, PREAMBLE, SUFFIX, QUESTION_ID, QUESTION, CRITERIA  # noqa: F401  (task1_jev.items uses them)
J.JEV_MODEL, J.PREAMBLE, J.SUFFIX, J.QUESTION_ID, J.QUESTION, J.CRITERIA = JEV_MODEL, PREAMBLE, SUFFIX, QUESTION_ID, QUESTION, CRITERIA

E14A, E15A = T.E14A, T.E15A


def main():
    S = {}
    for tag, cid, cond, body in J.items():
        if tag in ('E14', 'E15'):
            S[cid] = json.loads((JEV_CACHE / f'{digest(body)}.json').read_text())['score']
    frows = T.fresh_qwen()['rows']
    for r in frows:
        S[r['row_id']] = float(r['jev'])
    cut = json.loads((ROOT / 'runs/20261028-0107-e13-matched-jev-d814/results.json').read_text())['qwen_test']['jev_cut']
    res = dict(threshold=cut, n_scores=len(S))

    # E14
    e14rows = [r for r in csv.DictReader(open(T.E14R / 'scores.csv')) if r['condition'] in ('PERMIT', 'PERMIT_NO_CONSEQUENCE')]
    disp = [dict(r, score=S[r['row_id']], label=int(r['label']), round=int(r['round'])) for r in frows if r['scenario'] == 'dispatch-integration-code']
    fl = [r for r in disp if r['label']]
    out = dict(FORBID_dispatch_test=dict(leaking_calls=E14A.describe([r['score'] for r in fl], cut),
                                         within_round_auroc=E14A.within_round_auroc(disp), within_round_auroc_ci=E14A.boot(disp, E14A.within_round_auroc)))
    for arm in ('PERMIT', 'PERMIT_NO_CONSEQUENCE'):
        ar = [dict(r, score=S[r['call_id']], label=int(r['label']), round=int(r['round'])) for r in e14rows if r['condition'] == arm]
        sh = [r for r in ar if r['label']]
        out[arm] = dict(sharing_calls=E14A.describe([r['score'] for r in sh], cut),
                        median_diff_vs_forbidden_leaks=E14A.med(sh) - E14A.med(fl), median_diff_ci=E14A.boot_diff(sh, fl, E14A.med),
                        within_round_auroc=E14A.within_round_auroc(ar), within_round_auroc_ci=E14A.boot(ar, E14A.within_round_auroc),
                        calls=len(ar), sharing=len(sh), episodes=len({r['episode'] for r in ar}))
    res['E14'] = out

    # E15
    calls = json.loads((T.E15R / 'calls.json').read_text())
    cs = [dict(c, score=S[c['call_id']], group='FORBID' if c['arm'] in ('FORBID', 'FORBID_OLD') else c['arm']) for c in calls]
    share = [c for c in cs if c['label'] and not (c['arm'] == 'LATE' and c['before_ask'])]
    by_g = {g: [c for c in share if c['group'] == g] for g in E15A.ARMS}
    nonread = {g: [c for c in v if not c['follows_read']] for g, v in by_g.items()}
    read = {g: [c for c in v if c['follows_read']] for g, v in by_g.items()}

    def auc_boot(a, b, reps=2000, seed=20261028):
        rng = random.Random(seed); xa = [c['score'] for c in a]; xb = [c['score'] for c in b]
        est = E14A.auroc(xa, xb)
        bs = sorted(E14A.auroc([rng.choice(xa) for _ in xa], [rng.choice(xb) for _ in xb]) for _ in range(reps))
        return dict(value=est, ci95=[bs[int(.025 * reps)], bs[int(.975 * reps) - 1]], n_forbidden=len(xa), n_permitted=len(xb))

    res['E15'] = dict(nonread={g: E15A.rate(v, cut) for g, v in nonread.items()},
                      follows_read={g: E15A.rate(v, cut) for g, v in read.items()},
                      diff_EARLY_minus_FORBID_nonread=E15A.boot_diff(nonread['EARLY'], nonread['FORBID'], cut),
                      auroc_forbidden_vs_early=dict(nonread=auc_boot(nonread['FORBID'], nonread['EARLY']),
                                                    follows_read=auc_boot(read['FORBID'], read['EARLY']),
                                                    all=auc_boot(by_g['FORBID'], by_g['EARLY'])),
                      median_score={g: E14A.med(v) for g, v in by_g.items()})
    (OUT / 'task1_jev_e14e15.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, default=float))


if __name__ == '__main__':
    main()
