"""Predeclared engineering review of actual full-corpus token supply.

Category targets stay50/30/10/10. Wiki is bounded to0.5% and3 exposures;
remaining Korean web weights equalize exposure to actual train-token supply.
Other exposure limits are engineering guards, not language-quality guarantees.
"""
import math

SOURCES={'wiki_ko':'ko','fineweb2_ko':'ko','hplt_ko':'ko','fineweb_edu':'en','github_code_v1':'code','owm':'math'}
POLICY=dict(version=1,main_tokens=119537664000,rehearsal_tokens=10485760000,wiki_weight_max=.005,
            category_targets=dict(ko=.5,en=.3,code=.1,math=.1),minimum_ko_en_holdout_tokens=50000000,
            max_main_exposures=dict(wiki_ko=3,fineweb2_ko=3,hplt_ko=3,fineweb_edu=2,github_code_v1=4,owm=2),
            scope='Full actual train-token availability review. Wiki prior5% is reduced when it would heavily repeat the small corpus. No source quality or official benchmark guarantee.')


def choose_mix(counts,categories,policy=POLICY):
    if categories!=SOURCES or set(counts['source_counts'])!=set(SOURCES):raise ValueError('Unexpected full source/category coverage')
    available={};holds={c:0 for c in policy['category_targets']}
    for source,record in counts['source_counts'].items():
        for key in ['tokens','holdout_tokens','train_docs','holdout_docs']:
            if type(record[key]) is not int or record[key]<1:raise ValueError('Actual source token/document counts invalid')
        if record['tokens']<max(4097,record['train_docs']) or record['holdout_tokens']<record['holdout_docs']:
            raise ValueError('Actual token supply cannot cover source documents')
        available[source]=record['tokens'];holds[categories[source]]+=record['holdout_tokens']
    if holds!=counts['category_holdout_tokens'] or min(holds['ko'],holds['en'])<policy['minimum_ko_en_holdout_tokens']:
        raise ValueError('Actual category holdout target/count differs')
    wiki=min(policy['wiki_weight_max'],policy['max_main_exposures']['wiki_ko']*available['wiki_ko']/policy['main_tokens'])
    korean=policy['category_targets']['ko']-wiki;web=available['fineweb2_ko']+available['hplt_ko']
    weights=dict(wiki_ko=wiki,fineweb2_ko=korean*available['fineweb2_ko']/web,hplt_ko=korean*available['hplt_ko']/web,
                 fineweb_edu=.3,github_code_v1=.1,owm=.1)
    exposures={str(target):{s:target*w/available[s] for s,w in weights.items()} for target in [policy['rehearsal_tokens'],policy['main_tokens']]}
    for source,exposure in exposures[str(policy['main_tokens'])].items():
        if not math.isfinite(exposure) or exposure>policy['max_main_exposures'][source]+1e-10:raise ValueError('Source repetition requires review:'+source)
    if not math.isclose(sum(weights.values()),1,abs_tol=1e-10):raise ValueError('Selected mix weights differ')
    return dict(weights=weights,expected_exposures=exposures,actual_train_tokens=available,category_holdout_tokens=holds,
                prior_wiki_5pct_exposures=policy['main_tokens']*.05/available['wiki_ko'],
                reason='Actual full train-token counts used. Category targets ko/en/code/math50/30/10/10. Wiki capped at0.5% and3 main-target exposures rather than copying the old5% proposal; other Korean weights proportional to actual web train tokens. Exposure and50M ko/en holdout guards passed. Sampler is replacement sampling; exposures are expectations, not strict epochs.')
