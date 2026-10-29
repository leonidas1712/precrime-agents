"""Generate source-backed coverage, detector settings, and scenario-rule tables."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RUN='runs/20261027-0104-e05-arxiv-repro-details'


def escape(text):
    return ''.join({'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_\allowbreak{}','{':r'\{','}':r'\}'}.get(c,c) for c in str(text))


def main():
    reg=json.loads((HERE/'numbers.yaml').read_text())
    counts=json.loads((ROOT/RUN/'counts.json').read_text())
    settings=json.loads((ROOT/RUN/'settings.json').read_text())
    scenarios=json.loads((ROOT/RUN/'scenarios.json').read_text())
    def add(macro,path):
        val=counts
        for k in path:val=val[k]
        reg['numbers'][macro]=dict(value=val,macro=macro,dataset='Matched development/fresh call cohorts',evaluation='Saved metadata recount',status='verified',run=RUN,format='integer',source=dict(file=RUN+'/counts.json',value_path=path))
        return {'key':macro}
    cover=[]
    for split,title in [('development','Development'),('fresh','Fresh')]:
        prefix='gptossCoverage'+title
        cover.append([{'Fresh':'Test'}.get(title,title)]+[add(prefix+macro,['gptoss',split,key]) for macro,key in [('Episodes','episodes'),('EmittingEpisodes','emitting_episodes'),('MatchedCalls','matched_calls'),('EmittingCalls','matched_emitting_calls'),('EmptyEmissions','empty_reasoning_emission_calls'),('RetainedEarlierNegatives','retained_earlier_calls_in_excluded_emission_episodes')]])
    reg['tables']['reasoning_coverage']=dict(columns='lrrrrrr',header=['Set','Episodes','Emitting eps.','Matched calls','Emitting calls','Empty emissions','Earlier negatives'],rows=cover)
    qrows=[]
    for key,label,macro in [('recruiting_calls','All recruiting calls','RecruitingCalls'),('emitting_calls','Emitting recruiting calls','EmittingCalls'),('earlier_nonemitting_calls_in_emitting_episodes','Earlier nonemitting calls in emitting episodes','EarlierNonemittingCalls'),('refusing_episode_calls','Calls in refusing recruiting episodes','RefusingCalls'),('control_calls','Additional control calls','ControlCalls'),('recruiting_plus_control_calls','Recruiting plus control calls','AllCalls')]:
        qrows.append([label,add('qwenFreshAccounting'+macro,['qwen_fresh',key])])
    reg['tables']['qwen_call_accounting']=dict(columns='lr',header=['Qwen test population','Calls'],rows=qrows)
    tf=settings['tfidf'];lr=settings['text_logistic'];mlp=settings['mlp'];inc=settings['incoming']
    detector_rows=[
      ['TF-IDF features',f"Word {tf['ngram_range'][0]}--{tf['ngram_range'][1]}-grams; minimum document frequency {tf['min_df']}; at most {tf['max_features']:,} features; sublinear term frequency; L2 normalization. Vocabulary and IDF fitted on training calls only."],
      ['Text logistic regression',f"C={lr['C']}; {lr['solver']}; {lr['max_iter']} iterations; tolerance {lr['tol']}; round-balanced weights; fixed hyperparameters; exact saved LOSO families."],
      ['Text input views','Full: saved messages and tools, with provider-matched previous reasoning retention. Latest: final saved request message only. Exact protected values masked in both; current reply excluded.'],
      ['MLP (round-level appendix)',f"One hidden layer, {mlp['hidden_units']} ReLU units; train-only StandardScaler; weighted binary cross-entropy; AdamW, learning rate {mlp['learning_rate']}, weight decay {mlp['weight_decay']}; {mlp['epochs']} epochs, batch {mlp['batch_size']}. Layer selected by inner validation AUROC; fixed epoch count."],
      ['Incoming-only features',f"GPT-OSS layer {inc['layer']}, fixed before fitting; seed {inc['seed']}; six LOSO rotations. All newly appended user/tool messages since the previous target request, including Harmony framing; no current reply tokens."],
      ['Incoming-only training',f"Logistic C={inc['C']}; reply-balanced scaling and round-balanced reply loss. At most {inc['training_tokens_per_reply_max']} evenly spaced training tokens per call, endpoints included. SWiM window 16 on the sampled sequence plus softmax-weighted loss, temperature 1; no layer selection."],
      ['Incoming-only scoring','All incoming positions; exponential moving average with alpha=2/17, then maximum per call. Cutoff: strict greater-than, nominal 5 percent of calibration control-call maxima. Different recipe and coverage from the last-prompt comparison.'],
    ]
    reg['tables']['detector_settings']=dict(columns=r'p{0.22\linewidth}p{0.70\linewidth}',header=['Detector setting','Saved implementation'],rows=detector_rows)
    reg['tables']['split_settings']=dict(columns=r'p{0.22\linewidth}p{0.70\linewidth}',header=['Split setting','Implementation'],rows=[
      ['Family','Episode-generation family ID. Related conditions/tactics reuse the same family ID and deterministic files/value within a scenario; IDs reused across scenarios remain one group. All calls in an episode inherit that ID.'],
      ['Mixed-scenario CV','Five family-grouped outer folds, seeds 42/43/44. Inner validation is one of four family-grouped folds (approximately 25 percent); training and validation families are disjoint.'],
      ['LOSO layer selection','Outer test is one scenario; remove its family IDs from training even when shared across scenarios. Inner validation holds out one training scenario and purges its family IDs. Choose the highest within-scenario/round AUROC; ties use the earliest candidate layer.'],
      ['Final last-prompt models','Inner selection uses a 25 percent development-family holdout, separately for seeds 42/43/44. Refit chosen layers on development data; test scores average logits standardized by each model training mean and standard deviation.'],
      ['Reasoning model','One seed-42 model; 25 percent development-family inner holdout, layer selected by validation MAX-token AUROC. Refit on development calls; no refit on test episodes.'],
    ])
    scenario_rows=[]
    for row in scenarios:
        value=escape(row['secret_key'])+r'\newline '+escape(row['secret_prefix'])+f" + {row['hex_characters']} hexadecimal characters"
        scenario_rows.append([escape(row['id'])+r'\newline '+value,escape(row['rule_text'])])
    reg['tables']['scenario_rules']=dict(columns=r'p{0.27\linewidth}p{0.65\linewidth}',header=['Scenario / synthetic value','Exact operator policy (consequence condition)'],rows=scenario_rows)
    (HERE/'numbers.yaml').write_text(json.dumps(reg,indent=2)+'\n')


if __name__=='__main__':main()
