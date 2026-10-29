"""Register text monitors that read the current call's reasoning (gpt-oss-20b development calls).

Jev and GPT-5-nano read cumulative prefixes of the call's own reasoning (secret value masked) and
answer whether the agent will write the value in this reply. Same 1,995 calls and 121 leaking calls
as the development probes; AUROC within scenario x round with family bootstrap. The monitors are not
trained, so the held-out-scenario ("loso") and pooled splits give the same AUROC; we read "loso"
to match the development-episode probe results.
"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = [('gptossReasonJevAuroc', 'runs/20261026-0740-e06-reasoning-prefix-jev', 'Jev'),
        ('gptossReasonNanoAuroc', 'runs/20261026-0756-e06-reasoning-prefix-nano', 'GPT-5-nano')]
PATH = ['immediate/all/loso', 'within_round_auc']
CI = ['immediate/all/loso', 'within_round_ci95']


def main():
    reg = json.loads((HERE/'numbers.yaml').read_text())
    for macro, run, name in RUNS:
        d = json.loads((ROOT/run/'metrics.json').read_text())
        block = d[PATH[0]]
        assert block['rows'] == 1995 and block['positives'] == 121, (run, block['rows'], block['positives'])
        assert block['within_round_auc'] == d['immediate/all/pooled']['within_round_auc']
        reg['numbers'][macro] = dict(
            value=block['within_round_auc'], ci=block['within_round_ci95'], macro=macro,
            dataset='GPT-OSS development episodes (1,995 calls, 121 leaking)',
            evaluation=f'{name} reads prefixes of the current call reasoning (secret masked); held-out scenarios split (identical to pooled: no training); within scenario x round AUROC',
            status='exploratory', run=run, format='decimal', decimals=2,
            source=dict(file=run+'/metrics.json', value_path=PATH, ci_path=CI))
    (HERE/'numbers.yaml').write_text(json.dumps(reg, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__': main()
