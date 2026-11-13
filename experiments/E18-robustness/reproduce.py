"""E18 step 1: reproduce the published development ladder and the test "This call?" numbers from saved activations.

Refits from scratch (CPU, no GPU/API) with the original E05 code:
  gpt-oss dev  : lastprompt_percall.run_fit, matched 1,995-call cohort, 18 held-out-scenario (loso) fits per label
                 labels immediate, strict1, round_any, eventual_any
  Qwen dev     : arxiv_qwen_refit.fit, 2,976 dev calls, 18 loso fits per label; strict1 as arxiv_qwen_strict_next
  test         : freeze() loop of arxiv_frozen_percall / arxiv_gpt_fresh (3 seeds, dev-only), scored once
Every number is compared with the stored metrics; any mismatch fails the run.

    uv run --no-project --python 3.11 --with numpy==2.4.6 --with scikit-learn==1.8.0 --with scipy==1.17.1 --with threadpoolctl==3.7.0 --with torch --with matplotlib \
        python experiments/E18-robustness/reproduce.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (sets sys.path)
import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
from collusion.runs import start_run, write_json_atomic  # noqa: E402
import devfits  # noqa: E402

TOL = 1e-9


def check(key, got, failures):
    t = common.target(key)
    ok = (abs(got['within_round_auc'] - t['within_round_auc']) < TOL
          and np.allclose(got['within_round_ci95'], t['within_round_ci95'], atol=TOL, rtol=0)
          and got['rows'] == t['rows'] and got['positives'] == t['positives'])
    print('CHECK', key, 'OK' if ok else 'MISMATCH', got['within_round_auc'], t['within_round_auc'], flush=True)
    if not ok:
        failures.append(key)
    return dict(reproduced=bool(ok), target_auc=t['within_round_auc'], target_ci95=t['within_round_ci95'],
                target_rows=t['rows'], target_positives=t['positives'], **got)


def main():
    run = start_run('e18-reproduce', experiment='E18', tags=['cpu-only', 'reproduction'],
                    parents=['20261026-2201-e05-arxiv-qwen-any-public', '20261026-2223-e05-arxiv-qwen-strict-next',
                             '20261026-1304-e05-lastprompt-label-ladder', '20261026-1221-e05-lastprompt-percall-6013',
                             '20261026-2212-e05-arxiv-qwen-fresh-percall', '20261026-2222-e05-arxiv-gpt-fresh-percall'],
                    settings=dict(C=.1, workers=2, threads_per_worker=2, no_gpu=True, no_api=True, tolerance=TOL))
    try:
        out = run.dir; results = {}; failures = []
        qdir, gdir = out / 'qwen_dev', out / 'gpt_dev'
        qrows, qplans, qstrict = devfits.qwen_prepare(qdir)
        gplans = devfits.gpt_plans()
        jobs = [(devfits.gpt_job, (str(gdir), label, p, .1)) for label in ['immediate', 'strict1', 'round_any', 'eventual_any'] for p in gplans]
        jobs += [(devfits.qwen_job, (str(qdir), label, p, .1)) for label in ['immediate', 'round_near', 'eventual'] for p in qplans]
        jobs += [(devfits.qwen_job, (str(qdir), 'strict1', p, .1)) for p in qstrict]
        with ProcessPoolExecutor(max_workers=2) as pool:
            for i, f in enumerate(as_completed([pool.submit(fn, j) for fn, j in jobs]), 1):
                print('DONE', i, len(jobs), f.result(), flush=True)
        with threadpool_limits(limits=2):
            grows = devfits.common.gpt_dev()[0]
            for label, n, key in [('immediate', 1995, 'gpt_dev/immediate'), ('strict1', 1874, 'gpt_dev/strict1'),
                                  ('round_any', 1995, 'gpt_dev/round_any'), ('eventual_any', 1995, 'gpt_dev/eventual_any')]:
                results[key] = check(key, devfits.summarize(grows, gdir / 'matched' / label, label, n), failures)
            for label, n, key in [('immediate', 2976, 'qwen_dev/immediate'), ('round_near', 2976, 'qwen_dev/round_near'),
                                  ('eventual', 2976, 'qwen_dev/eventual'), ('strict1', 2910, 'qwen_dev/strict1')]:
                results[key] = check(key, devfits.summarize(qrows, qdir / label, label, n), failures)
            write_json_atomic(out / 'reproduction.json', results)
            # test "This call?" — refit the frozen models and compare with the frozen plans
            m, specs, _, _ = common.qwen_fresh(qrows, 'immediate')
            frozen = json.loads(common.Q_PLAN.read_text())['models']
            same = all(s['layer'] == f['layer'] and abs(s['train_mean'] - f['train_mean']) < 1e-9 for s, f in zip(specs, frozen))
            results['qwen_test/immediate'] = check('qwen_test/immediate', common.summary(m) | dict(frozen_models_identical=same,
                                                   layers=[s['layer'] for s in specs]), failures)
            if not same:
                failures.append('qwen_test/frozen_models')
            dev = common.gpt_dev_fresh_rows(); test = common.gpt_test_rows(); xt = common.gpt_test_features(test)
            m, specs, _, _ = common.gpt_fresh(dev, test, xt, 'immediate')
            frozen = json.loads(common.GPT_PLAN.read_text())['models']
            same = all(s['layer'] == f['layer'] and abs(s['train_mean'] - f['train_mean']) < 1e-9 for s, f in zip(specs, frozen))
            results['gpt_test/immediate'] = check('gpt_test/immediate', common.summary(m) | dict(frozen_models_identical=same,
                                                  layers=[s['layer'] for s in specs]), failures)
            if not same:
                failures.append('gpt_test/frozen_models')
        write_json_atomic(out / 'reproduction.json', results)
        run.result(reproduced=not failures, failures=failures,
                   **{k: round(v['within_round_auc'], 4) for k, v in results.items()})
        run.finish(status='finished' if not failures else 'failed', error=None if not failures else f'mismatch: {failures}')
        print('FAILURES', failures)
    except BaseException as e:
        run.finish('failed', str(e)); raise


if __name__ == '__main__':
    main()
