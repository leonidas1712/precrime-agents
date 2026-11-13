"""E18 (B): regularisation sensitivity of development "This call?" AUROC.

Held-out-scenario (loso) development fits, same saved folds, same inner layer-selection rule; only the
L2 strength C of the logistic regression changes (recipe: C=0.1). C=0.1 is refit and must reproduce
the published value. If lbfgs fails to converge at the recipe's max_iter=2000, the fit is retried at
20000 and listed under max_iter_raised.

    --part lastprompt : gpt-oss last-prompt (1,995 calls) and Qwen last-prompt (2,976 calls)
    --part reasoning  : gpt-oss reasoning token probe (reasoning_stream_fit.run_job, token-max score)

    uv run ... python experiments/E18-robustness/c_sweep.py --part lastprompt --repro runs/<id>-e18-reproduce
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
from collusion.runs import start_run, write_json_atomic  # noqa: E402
import devfits  # noqa: E402

CS = [0.01, 0.03, 0.1, 0.3, 1, 10]
REASONING_TARGET = common.OLD / '20261026-0526-e05-reasoning-stream/metrics.json'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--part', choices=['lastprompt', 'reasoning'], required=True)
    p.add_argument('--repro', type=Path, required=True); p.add_argument('--workers', type=int, default=2); a = p.parse_args()
    rep = json.loads((a.repro / 'reproduction.json').read_text())
    # Accept refits within 0.002 of the published AUROC (6 dev refits differ by <=0.0015, likely uncommitted code
    # at the original runs; every 2-decimal paper value is unchanged). See RESULTS.md section 6.
    assert all(abs(v['within_round_auc'] - v['target_auc']) <= 0.002 for v in rep.values()), 'Step 1 did not reproduce within 0.002; stop'
    run = start_run(f'e18-c-sweep-{a.part}', experiment='E18', tags=['cpu-only', 'sensitivity'], parents=[a.repro.name],
                    settings=dict(C=CS, part=a.part, split='loso (held-out scenarios)', workers=a.workers, threads_per_worker=2,
                                  no_gpu=True, no_api=True))
    try:
        out = run.dir; results = {}
        if a.part == 'lastprompt':
            qrows, qplans, _ = devfits.qwen_prepare(out / 'qwen_rows')
            gplans = devfits.gpt_plans()
            jobs = [(devfits.gpt_job, (str(out / f'gpt_C{C}'), 'immediate', pl, C)) for C in CS for pl in gplans]
            jobs += [(devfits.qwen_job, (str(out / f'qwen_C{C}'), 'immediate', pl, C)) for C in CS for pl in qplans]
            for C in CS:  # arxiv_qwen_refit.fit reads rows.json / analysis_plan.json from its output folder
                d = out / f'qwen_C{C}'; d.mkdir(parents=True, exist_ok=True)
                for f in ['rows.json', 'analysis_plan.json']:
                    (d / f).symlink_to(out / 'qwen_rows' / f)
        else:
            from reasoning_stream_labels import add_horizon_labels
            rplans = devfits.gpt_plans()
            jobs = [(devfits.reasoning_job, (str(out / f'reasoning_C{C}'), pl, C)) for C in [0.1] + [c for c in CS if c != 0.1] for pl in rplans]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for i, f in enumerate(as_completed([pool.submit(fn, j) for fn, j in jobs]), 1):
                print('DONE', i, len(jobs), f.result(), flush=True)
        with threadpool_limits(limits=2):
            if a.part == 'lastprompt':
                grows = common.gpt_dev()[0]
                for C in CS:
                    results[f'gpt_lastprompt/C={C}'] = devfits.summarize(grows, out / f'gpt_C{C}' / 'matched' / 'immediate', 'immediate', 1995)
                    results[f'qwen_lastprompt/C={C}'] = devfits.summarize(qrows, out / f'qwen_C{C}' / 'immediate', 'immediate', 2976)
                    write_json_atomic(out / 'metrics.json', results)
                checks = dict(gpt=abs(results['gpt_lastprompt/C=0.1']['within_round_auc'] - common.target('gpt_dev/immediate')['within_round_auc']) < 1e-9,
                              qwen=abs(results['qwen_lastprompt/C=0.1']['within_round_auc'] - common.target('qwen_dev/immediate')['within_round_auc']) < 1e-9)
            else:
                rows = add_horizon_labels(json.loads((common.GPT_STREAM_DATA / 'replies.json').read_text()))
                for C in CS:
                    results[f'gpt_reasoning/C={C}'] = devfits.summarize(rows, out / f'reasoning_C{C}' / 'immediate' / 'all', 'immediate', 1995, score_key='max_scores')
                    write_json_atomic(out / 'metrics.json', results)
                t = json.loads(REASONING_TARGET.read_text())['immediate/all/loso/extended/token_max']
                checks = dict(gpt_reasoning=abs(results['gpt_reasoning/C=0.1']['within_round_auc'] - t['within_round_auc']) < 1e-9)
        results['C0.1_reproduces_published'] = checks
        write_json_atomic(out / 'metrics.json', results)
        run.result(c01_reproduced=all(checks.values()),
                   **{k: round(v['within_round_auc'], 4) for k, v in results.items() if isinstance(v, dict) and 'within_round_auc' in v})
        run.finish(status='finished' if all(checks.values()) else 'failed', error=None if all(checks.values()) else 'C=0.1 mismatch')
    except BaseException as e:
        run.finish('failed', str(e)); raise


if __name__ == '__main__':
    main()
