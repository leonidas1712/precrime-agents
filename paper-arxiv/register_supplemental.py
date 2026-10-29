"""Register source-verified supplemental results and generate table specifications."""
import json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def get(d,path):
    for k in path:d=d[k]
    return d

def main():
    registry=json.loads((HERE/'numbers.yaml').read_text());numbers=registry['numbers'];
    for key in list(numbers):
        if any(c.isdigit() for c in numbers[key]['macro']):del numbers[key]
    tables=registry.setdefault('tables',{})
    def add(macro,file,path,ci=None,fmt='decimal',decimals=2,dataset='development episodes',evaluation='held-out scenarios',status='exploratory',divide=None):
        macro=''.join(('Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine')[int(c)] if c.isdigit() else c for c in macro)
        data=json.loads((ROOT/file).read_text());value=get(data,path)
        if divide:value/=get(data,divide)
        e=dict(value=value,macro=macro,dataset=dataset,evaluation=evaluation,status=status,run=str(Path(file).parent),format=fmt,decimals=decimals,source=dict(file=file,value_path=path))
        if divide:e['source']['divide_by_path']=divide
        if ci:e['ci']=get(data,ci);e['source']['ci_path']=ci
        numbers[macro]=e;return dict(key=macro,ci=bool(ci))
    def old(key,ci=True):return dict(key=key,ci=ci)
    def table(name,header,rows,columns=None):tables[name]=dict(header=header,rows=rows,columns=columns or 'l'*len(header))
    readg='runs/20261026-0526-e05-reasoning-stream/immediate_readstep.json';readq='runs/20261026-2211-e05-arxiv-qwen-controls/metrics.json'
    add('gptossProbeMinusReadFlag',readg,['paired','loso/matched_last_prompt minus latest_incoming_protected_read','difference'],['paired','loso/matched_last_prompt minus latest_incoming_protected_read','ci95'])
    add('gptossPostReadProbe',readg,['metrics','loso/latest_incoming_protected_read_only/matched_last_prompt','within_round_auc'],['metrics','loso/latest_incoming_protected_read_only/matched_last_prompt','within_round_ci95'])
    add('qwenPostReadProbe',readq,['metrics','probe_within_postread','within_round_auc'],['metrics','probe_within_postread','within_round_ci95'])
    registry['aliases']['qwenProbeMinusReadFlag']='qwenProbeMinusReadFlag' if False else numbers['qwen_probe_minus_read_flag']['macro']
    if registry['aliases']['qwenProbeMinusReadFlag']=='qwenProbeMinusReadFlag':del registry['aliases']['qwenProbeMinusReadFlag']
    textrows=[]
    for model,folder in [('gptoss','20261026-2242-e05-arxiv-gpt-text-workflow'),('qwen','20261026-2245-e05-arxiv-qwen-text-workflow')]:
        file=f'runs/{folder}/metrics.json'
        for name,suffix in [('full_input','TextFull'),('latest_input','TextLatest')]:
            cell=add(model+suffix,file,['metrics',name,'within_round_auc'],['metrics',name,'within_round_ci95'])
            gain=add(model+'ProbeMinus'+suffix,file,['paired','last_prompt_minus_'+name,'estimate'],['paired','last_prompt_minus_'+name,'ci95'])
            textrows.append([model,'Full input' if name=='full_input' else 'Latest message',cell,gain])
        if model=='gptoss':
            add('gptossWriteFlag',file,['metrics','write_file','within_round_auc'],['metrics','write_file','within_round_ci95'])
            add('gptossWithinWriteProbe',file,['metrics','last_prompt_within_handoff_write','within_round_auc'],['metrics','last_prompt_within_handoff_write','within_round_ci95'])
            add('gptossProbeMinusWriteFlag',file,['paired','last_prompt_minus_write_file','estimate'],['paired','last_prompt_minus_write_file','ci95'])
    table('trained_text',['Target','TF-IDF input','AUROC [CI]','Last-prompt gain [CI]'],textrows)
    f='runs/20261026-1302-e05-stream-confirm-score/alarms.json'
    add('gptossFreshRefusingEpisodeAlarm',f,['held_recruiting_episode_alarm','rate'],fmt='percent',decimals=0,dataset='fresh test',evaluation='frozen reasoning threshold',status='confirmed')
    # Wilson interval: one first-emission call per independent leaking episode.
    a=json.loads((ROOT/f).read_text())['leaking_reply_with_reasoning_warning'];n=a['denominator'];k=a['numerator'];z=1.959963984540054;p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;rad=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    derived=HERE/'derived_operating_points.json';derived.write_text(json.dumps(dict(caught=dict(value=p,ci95=[center-rad,center+rad],numerator=k,denominator=n,method='Wilson 95% binomial interval; one first-emission call per episode',source=f)),indent=2)+'\n')
    add('gptossFreshCaught','paper-arxiv/derived_operating_points.json',['caught','value'],['caught','ci95'],fmt='percent',decimals=0,dataset='fresh test',evaluation='frozen reasoning threshold',status='confirmed')
    # Replay metrics retain their original token horizons and collection cohorts.
    replay=[]
    f='runs/20261025-0536-e05-replay2-analysis/replay-fidelity.json'
    replay.append(['Qwen development','First 64 reply tokens',add('qwenReplayNLL',f,['nll'],decimals=3),add('qwenReplayTop',f,['top5'],fmt='percent',decimals=2),'ratio median 1.0'])
    f='runs/20261025-0923-e02-replay3-completion/summary.json'
    replay.append(['Qwen test','First 64 reply tokens',add('qwenFreshReplayNLL',f,['weighted_nll'],decimals=3),add('qwenFreshReplayTop',f,['weighted_top5'],fmt='percent',decimals=2),'--'])
    f='runs/20261026-0138-e02-gpt-oss-full-completion/summary.json';d=json.loads((ROOT/f).read_text())
    for i,r in enumerate(d['fidelity']):
        provider=r['provider'];segment=r['segment'];macro='gptReplay'+provider+segment.title()
        cnt=d['prompt_counts'][provider];coverage=f"{cnt['within1']}/{cnt['requests']} within 1"
        replay.append(['GPT '+provider,{'all':'All incl. framing','reasoning':'Reasoning body','answer':'Answer/tool body'}[segment],add(macro+'NLL',f,['fidelity',i,'mean_token_nll'],decimals=3),add(macro+'Top',f,['fidelity',i,'top5_fraction'],fmt='percent',decimals=2),coverage])
    f='runs/20261026-1302-e05-stream-confirm-score/fidelity.json'
    for seg in ['all','reasoning','answer']:
        replay.append(['GPT test AkashML',{'all':'All incl. framing','reasoning':'Reasoning body','answer':'Answer/tool body'}[seg],add('gptFreshReplay'+seg.title()+'NLL',f,['AkashML',seg,'mean_token_nll'],decimals=3),add('gptFreshReplay'+seg.title()+'Top',f,['AkashML',seg,'top5_fraction'],fmt='percent',decimals=2),'1650/1671 within 1'])
    table('replay',['Collection/provider','Token span','NLL','Top-5','Prompt counts'],replay)
    # Original near-term definitions exclude the emitting call; separate from new ladder.
    rr=[];f='runs/20261025-0730-e05-near-term/metrics.json'
    for variant in ['A_hazard','B_round_contrast','C_step_contrast']:
      for mode in ['loso','pooled']:
       for method in ['nested','fixed32']:
        macro='qwenRound'+variant.split('_')[0]+mode.title()+('Selected' if method=='nested' else 'Fixed')
        rr.append([variant[0],mode,'Logistic selected' if method=='nested' else 'Logistic layer 32',add(macro,f,[variant,mode,'methods',method,'within_round_auc'],[variant,mode,'methods',method,'within_round_ci95'],dataset='original Qwen development round cohort')])
    f='runs/20261025-0848-e05-near-term-mlp/metrics.json'
    for variant in ['A_hazard','B_round_contrast']:
      for mode in ['loso','pooled']:
        rr.append([variant[0],mode,'MLP',add('qwenRoundMLP'+variant[0]+mode.title(),f,[variant+'/'+mode,'methods','mlp','within_round_auc'],[variant+'/'+mode,'methods','mlp','within_round_ci95'],dataset='original Qwen development round cohort')])
    table('robust_round',['Label','Split','Method','AUROC [CI]'],rr)
    roundrows=[]
    for label,filename in [('Eventual','metrics_main.json'),('Near-term','metrics_main_A_hazard.json')]:
      f='runs/20261025-0147-e06-monitor-main/'+filename;d=json.loads((ROOT/f).read_text())
      for name in ['probe (held-out scenario)','jev forecaster L3']:
        if name not in d:continue
        suffix='Probe' if name.startswith('probe') else 'Jev';base=[name,'ranking'];r=get(d,base)
        roundrows.append(['Qwen development',label,suffix,add('qwenOriginal'+label.replace('-','')+suffix,f,base+['within_round_auc'],base+['within_round_ci95'],dataset='original Qwen development round cohort')])
    f='runs/20261025-1030-e06-confirmation/confirm.json'
    for name,suffix in [('probe','Probe'),('jev forecaster L3','Jev'),('llm detector L2','Nano')]:
        roundrows.append(['Qwen test','Near-term',suffix,add('qwenRoundFresh'+suffix,f,['A_hazard','auc',name],['A_hazard','ci95',name],dataset='original Qwen fresh round cohort',status='confirmed')])
    for name,suffix in [('jev forecaster L3','Jev'),('llm detector L2','Nano')]:
        key='gain from probe over '+name
        roundrows.append(['Qwen test','Near-term','Gain over '+suffix,add('qwenRoundFreshGain'+suffix,f,['A_hazard','diff',key],['A_hazard','ci95',key],dataset='original Qwen fresh round cohort',status='confirmed')])
    table('round_level',['Collection','Label','Method','AUROC / gain [CI]'],roundrows)
    alarms=[]
    f='runs/20261026-1302-e05-stream-confirm-score/alarms.json'
    alarms.append(['GPT test','Reasoning','Development control calls']+[add('gptFreshAlarm'+str(i),f,[k,'rate'],fmt='percent',decimals=1,dataset='fresh test',status='confirmed') for i,k in enumerate(['leaking_reply_with_reasoning_warning','held_recruiting_reply_alarm','held_recruiting_episode_alarm'])])
    f='runs/20261026-2224-e05-arxiv-fresh-comparisons/metrics.json'
    for model,label in [('qwen','Qwen test'),('gpt_oss','GPT test')]:
        alarms.append([label,'Last-prompt','Development control calls']+[add(model.replace('_','')+'FreshAlarm'+str(i),f,[model,'alarms',k],fmt='percent',decimals=1,dataset='fresh test') for i,k in enumerate(['leak_rate','refusing_rate','refusing_episode_rate'])])
    f='runs/20261026-2220-e05-arxiv-table-inputs/metrics.json'
    for method,label in [('probe','Last-prompt'),('jev','Jev input')]:
        alarms.append(['Qwen development',label,'Training control calls']+[add('qwenDevAlarm'+method.title()+str(i),f,['qwen_alarms',method,k],fmt='percent',decimals=1) for i,k in enumerate(['leak_flagged','refusing_call_flagged','refusing_episode_flagged'])])
    table('alarms',['Collection','Method','Calibration','Caught calls','Refusing calls','Refusing episodes'],alarms)
    # Only audit quantities verified from structured historical outputs are included.
    labels=[];f='runs/20261026-0145-e06-addressee-audit/counts.json';d=json.loads((ROOT/f).read_text())
    for i,r in enumerate(d):
      if r['group']=='first_leak':labels.append([r['model'],'Historical first channel leak','Jev addressed to peer',add('labelAddress'+str(i),f,[i,'jev_to_peer'],fmt='integer'),add('labelAddressN'+str(i),f,[i,'messages'],fmt='integer')])
    f='runs/20261026-0118-e06-authority-claim-audit/metrics.json'
    for model,suffix in [('Qwen3-32B','Qwen'),('gpt-oss-20b','GPT')]:
      for q,key in [('Operator','q1_operator_authority'),('Receiver','q2_receiver_authority')]:
        labels.append([suffix,'Historical episode outcome',q+' claim among leaks',add('labelAuthority'+suffix+q,f,[model,key,'give_in_with_any_yes'],fmt='integer'),add('labelAuthorityN'+suffix+q,f,[model,key,'give_in_episodes'],fmt='integer')])
    table('labels',['Target','Audit cohort','Check','Yes','Denominator'],labels)
    table('history',['Analysis','Label','Fixed before collection','Test evaluation'],[
        ['Qwen original','This/next round','No','Fixed before analysis'],['GPT reasoning','Current-call emission','Yes (test)','Pre-registered'],['Qwen last-prompt','Current-call emission','No','Newly fixed, previously inspected'],['GPT last-prompt','Current-call emission','No','Newly fixed, previously inspected'],['TF-IDF / action controls','Current-call emission','No','None']])
    table('changed_numbers',['Analysis','Current estimate','Change from sprint'],[
        ['Qwen development per-call',old('qwen_percall_probe_auroc'),'Development only; any-public label'],['Qwen test per-call',old('qwen_fresh_percall_probe_auroc'),'New dev-only models'],['GPT test last-prompt',old('gptoss_fresh_lastprompt_auroc'),'New dev-only models'],['Qwen original round test',old('qwen_fresh_nearterm_probe_auroc'),'Unchanged'],['GPT reasoning test',old('gptoss_fresh_reasoning_probe_auroc'),'Unchanged']])
    table('repro',['Result','Run identifier','Script'],[
        ['Qwen dev ladder',r'\texttt{20261026-2201}',r'\texttt{arxiv\_qwen\_refit.py}'],['Qwen test',r'\texttt{20261026-2212}',r'\texttt{arxiv\_frozen\_percall.py}'],['GPT test last-prompt',r'\texttt{20261026-2222}',r'\texttt{arxiv\_gpt\_fresh.py}'],['Test paired comparisons',r'\texttt{20261026-2224}',r'\texttt{arxiv\_fresh\_comparisons.py}'],['Text/action controls',r'\texttt{20261026-2242/2245}',r'\texttt{arxiv\_text\_workflow.py}'],['All table/macro sources',r'\texttt{numbers.yaml}',r'\texttt{generate\_numbers.py}']])
    (HERE/'numbers.yaml').write_text(json.dumps(registry,indent=2)+'\n')
if __name__=='__main__':main()
