"""Register accepted E08 results; rejected replay remains history-only."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RUNS={'qwen':'runs/20261026-1717-e08-policy-counterfactual','gptoss':'runs/20261026-1732-e08-policy-counterfactual-gpt-oss'}


def main():
    reg=json.loads((HERE/'numbers.yaml').read_text())
    def add(prefix,suffix,path,ci=None,fmt='decimal',decimals=2):
        run=RUNS[prefix];data=json.loads((ROOT/run/'metrics.json').read_text());value=data
        for part in path:value=value[part]
        macro=prefix+'Policy'+suffix
        source=dict(file=run+'/metrics.json',value_path=path)
        entry=dict(value=value,macro=macro,dataset='Matched subset of fresh calls; frozen last-prompt probe',evaluation='Off-policy system-rule counterfactual; saved replies fixed; 1000 family bootstrap resamples, seed20261027',status='exploratory',run=run,format=fmt,decimals=decimals,source=source)
        if ci:
            bounds=data
            for part in ci:bounds=bounds[part]
            entry['ci']=bounds;source['ci_path']=ci
        reg['numbers'][macro]=entry
        return {'key':macro,'ci':bool(ci)}
    cells={}
    for prefix in RUNS:
        m=json.loads((ROOT/RUNS[prefix]/'metrics.json').read_text())
        assert m['forbid_reproduction']['passed']
        cells[prefix]={}
        for key,suffix in [('leaking','LeakingCalls'),('non_leaking','NonleakingCalls')]:
            cells[prefix][suffix]=add(prefix,suffix,['calls',key],fmt='integer')
        for key,suffix in [('permit_minus_forbid','PermitShift'),('neutral_minus_forbid','NeutralShift'),('permit_minus_neutral','PermitNeutralShift')]:
            cells[prefix][suffix]=add(prefix,suffix,['paired_leaking_call_changes',key,'median'],['paired_leaking_call_changes',key,'ci95'],fmt='decimal',decimals=2)
        for condition,suffix in [('FORBID','Forbid'),('PERMIT','Permit'),('NEUTRAL_EDIT','Neutral')]:
            for kind,metric in [('within_stratum','AUROC'),('pooled','UnstratifiedAUROC')]:
                cells[prefix][suffix+metric]=add(prefix,suffix+metric,['auc',condition,kind,'value'],['auc',condition,kind,'ci95'])
            cells[prefix][suffix+'Crossing']=add(prefix,suffix+'Crossing',['threshold_crossing',condition,'fraction'],['threshold_crossing',condition,'ci95'],fmt='percent',decimals=0)
            for part,suf in [('numerator','CrossingCount'),('denominator','CrossingTotal')]:
                add(prefix,suffix+suf,['threshold_crossing',condition,part],fmt='integer')
        add(prefix,'ReproductionError',['forbid_reproduction','max_absolute_error'],decimals=16)
    rows=[]
    for label,suffix in [('Leaking calls','LeakingCalls'),('Matched non-leaking calls','NonleakingCalls'),('Permitting rule minus original rule','PermitShift'),('Neutral rewording minus original rule','NeutralShift'),('Original rule AUROC','ForbidAUROC'),('Permitting rule AUROC','PermitAUROC'),('Neutral rewording AUROC','NeutralAUROC')]:
        rows.append([label,cells['qwen'][suffix],cells['gptoss'][suffix]])
    for label,suffix in [('Original rule threshold crossings','Forbid'),('Permitting rule threshold crossings','Permit'),('Neutral rewording threshold crossings','Neutral')]:
        row=[label]
        for prefix in RUNS:
            stem=prefix+'Policy'+suffix
            row.append('\\'+stem+'CrossingCount{}/\\'+stem+'CrossingTotal{}; \\'+stem+'Crossing{} \\'+stem+'CrossingCI{}')
        rows.append(row)
    reg['tables']['policy_counterfactual']=dict(columns=r'p{0.30\linewidth}p{0.30\linewidth}p{0.30\linewidth}',header=['Policy counterfactual','Qwen3-32B','gpt-oss-20b'],rows=rows)
    history=reg['tables']['history']['rows']
    history[:]=[r for r in history if not isinstance(r[0],str) or not r[0].startswith('E08 Qwen')]
    history.append(['Rule-change replay (Qwen)','Current-call emission','Existing calls; no generation','A first replay did not reproduce the saved scores and was discarded; the reported replay holds the saved reply fixed and reproduces them.'])
    (HERE/'numbers.yaml').write_text(json.dumps(reg,indent=2)+'\n')


if __name__=='__main__':main()
