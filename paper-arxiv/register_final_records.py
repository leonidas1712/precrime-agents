"""Register final existing-record coverage and settings summaries."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def main():
    p=HERE/'numbers.yaml';r=json.loads(p.read_text());file='runs/20261026-2330-e05-arxiv-final-records/summary.json';d=json.loads((ROOT/file).read_text())
    def add(macro,path,source=file,fmt='integer',decimals=0):
        v=json.loads((ROOT/source).read_text())
        for k in path:v=v[k]
        r['numbers'][macro]=dict(value=v,macro=macro,dataset='existing saved records',evaluation='descriptive audit; no fitting',status='verified',run=str(Path(source).parent),format=fmt,decimals=decimals,source=dict(file=source,value_path=path));return {'key':macro}
    rows=[]
    names={'eventual':'Eventual','eventual_any':'Eventual','round_near':'This/next round','round_any':'This/next round','immediate':'This call','strict_next':'One call earlier','strict1':'One call earlier','original_round_A':'Original round label'}
    for i,(key,counts) in enumerate(d['group_coverage'].items()):
        model,cohort,label=key.split('/');prefix='coverage'+chr(65+i)
        row=[model+' '+cohort,names[label]]
        for field in ['calls','retained_calls','positives','retained_positives','groups','retained_groups']:
            row.append(add(prefix+''.join(s.title() for s in field.split('_')),['group_coverage',key,field]))
        rows.append(row)
    r['tables']['group_coverage']=dict(columns='llrrrrrr',header=['Dataset','Label','Calls','Kept','Positive','Kept positive','Groups','Kept groups'],rows=rows)
    alarm=['Qwen test','Jev input','Saved CV controls']
    for suffix,key in [('Caught','caught_rate'),('Refusing','refusing_rate'),('Episodes','refusing_episode_rate')]:alarm.append(add('qwenFreshJevAudit'+suffix,['qwen_fresh_jev_alarm_rates',key],fmt='percent',decimals=1))
    r['tables']['alarms']['rows']=[row for row in r['tables']['alarms']['rows'] if not(row[0]=='Qwen test' and row[1]=='Jev input')]+[alarm]
    qf='paper-arxiv/frozen/qwen_percall_plan.json';gf='paper-arxiv/frozen/gpt_percall_plan.json';rf='runs/20261026-1139-e05-stream-confirm/frozen_plan.json'
    q=json.loads((ROOT/qf).read_text());g=json.loads((ROOT/gf).read_text());assert q['recipe']['candidates']==list(range(0,64,4))+[63]
    qpeer=d['peer_sampling']['qwen'];gpeer=d['peer_sampling']['gpt'];assert len(qpeer)==len(gpeer)==1
    qt=qpeer[0]['settings'];gt=gpeer[0]['settings']
    settings=[['Target temperature','0.7','0.7'],['Target token limit','1536','4096'],['Target reasoning','Disabled','Low'],['Target provider','DeepInfra (fp8)','AkashML / DekaLLM'],
      ['Peer model',r'\texttt{deepseek-v4-flash}',r'\texttt{deepseek-v4-pro-0813}'],
      ['Peer temperature',add('qwenPeerTemperature',['peer_sampling','qwen',0,'settings','temperature'],fmt='decimal',decimals=1),add('gptPeerTemperature',['peer_sampling','gpt',0,'settings','temperature'],fmt='decimal',decimals=1)],
      ['Peer token limit',add('qwenPeerTokenLimit',['peer_sampling','qwen',0,'settings','max_tokens']),add('gptPeerTokenLimit',['peer_sampling','gpt',0,'settings','max_tokens'])],
      ['Peer reasoning','Disabled','Medium'],['Probe seeds','42, 43, 44','42, 43, 44'],
      ['Candidate layers (zero-based)',', '.join(map(str,q['recipe']['candidates'])),', '.join(map(str,g['recipe']['candidates']))],
      ['Layers (seed order)',', '.join(str(x['layer']) for x in q['models']),', '.join(str(x['layer']) for x in g['models'])],
      ['Last-prompt threshold',add('qwenFrozenThreshold',['threshold'],qf,fmt='decimal',decimals=8),add('gptFrozenThreshold',['threshold'],gf,fmt='decimal',decimals=8)],
      ['Reasoning threshold','--',add('gptReasoningFrozenThreshold',['alarm','threshold'],rf,fmt='decimal',decimals=8)],
      ['Alarm rule','Strict $>$; nominal 5\\% control calls','Strict $>$; nominal 5\\% control calls'],
      ['Family rule','All calls grouped by family ID','All calls grouped by family ID'],
      ['Scenario holdout','Remove shared test-family IDs from training','Remove shared test-family IDs from training'],
      ['Development families','133','238']]
    assert qt['temperature']==gt['temperature']==.7 and qt['max_tokens']==1536 and gt['max_tokens']==4096
    r['tables']['settings']=dict(columns=r'p{0.24\linewidth}p{0.32\linewidth}p{0.32\linewidth}',header=['Setting','Qwen3-32B','gpt-oss-20b'],rows=settings)
    for macro in ['qwenProbeMinusTextFull','qwenProbeMinusTextLatest','gptossProbeMinusTextFull','gptossProbeMinusTextLatest']:
        r['numbers'][macro]['format']='signed';r['numbers'][macro]['decimals']=3
    p.write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
