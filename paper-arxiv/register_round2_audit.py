"""Register round-two count checks and apply approved table labels."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RUN='runs/20261027-0242-e05-arxiv-round2-audit'
HISTORY='A first replay did not reproduce the saved scores and was discarded; the reported replay holds the saved reply fixed and reproduces them.'


def main():
    reg=json.loads((HERE/'numbers.yaml').read_text());data=json.loads((ROOT/RUN/'audit.json').read_text())
    def add(macro,path):
        value=data
        for k in path:value=value[k]
        reg['numbers'][macro]=dict(value=value,macro=macro,dataset='Qwen scored development/fresh calls',evaluation='Saved source-file-writeback and round-label audit',status='verified',run=RUN,format='integer',source=dict(file=RUN+'/audit.json',value_path=path))
    for split,prefix in [('all','qwen'),('development','qwenDev'),('fresh','qwenFresh')]:
        for field,suffix in [('calls','Calls'),('episodes','Episodes')]:add(prefix+'SourceWriteback'+suffix,['writebacks',split,field])
    add('qwenRoundLegacyCalls',['round_table_reconciliation','rows'])
    add('qwenRoundLegacyPositives',['round_table_reconciliation','positives'])
    reg['numbers']['qwenOriginalNeartermProbe']['evaluation']='Held-out scenarios; eventual-label-trained probe evaluated against A_hazard without refit; same 2910 calls/168 positives as near-term-refit comparison'
    reg['numbers']['qwenRoundALosoSelected']['evaluation']='Held-out scenarios; A_hazard-trained and inner-layer-selected probe; same 2910 calls/168 positives as eventual-trained no-refit comparison'
    reg['numbers']['qwen_no_source_writeback_auroc']['dataset']='Qwen development episodes; one development source-writeback episode excluded (the other affected episode is fresh)'
    for row in reg['tables']['round_level']['rows']:
        if row[:3]==['Qwen development','Near-term','Probe']:row[2]='Probe (no refit)'
    for row in reg['tables']['shortcuts']['rows']:
        if row[1]=='Input probe':row[1]='Last-prompt probe'
    for row in reg['tables']['robust_call']['rows']:
        for i,cell in enumerate(row):
            if cell=='Exclude source-writeback episode':row[i]='Exclude development writeback episode'
    for row in reg['tables']['history']['rows']:
        if isinstance(row[0],str) and row[0].startswith('E08 Qwen'):
            row[0]='E08 Qwen replay';row[-1]=HISTORY
    (HERE/'numbers.yaml').write_text(json.dumps(reg,indent=2)+'\n')


if __name__=='__main__':main()
