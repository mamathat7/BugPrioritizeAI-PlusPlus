"""Bug-level paired testing with seed averaging, Holm adjustment and project bootstrap."""
from __future__ import annotations
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from ..utils import write_csv,write_json


METRICS=('APFD','APFDc','MRR','Recall@5','Recall@10','Top@5','Top@10','NDCG@5','NDCG@10')


def holm(pvalues):
    ps=np.asarray(pvalues,dtype=float)
    n=len(ps);result=np.ones(n)
    order=np.argsort(ps);running=0.
    for rank,i in enumerate(order):
        running=max(running,(n-rank)*ps[i])
        result[i]=min(1.,running)
    return result.tolist()


def paired_rank_biserial(diffs):
    from scipy.stats import rankdata
    d=np.asarray(diffs,dtype=float)
    nz=d[abs(d)>1e-12]
    if not len(nz):return 0.
    ranks=rankdata(abs(nz));pos=sum(ranks[nz>0]);neg=sum(ranks[nz<0]);total=pos+neg
    return float((pos-neg)/total)


def project_cluster_ci(differences,project_ids,resamples=10000,seed=11):
    rng=np.random.default_rng(seed)
    groups=defaultdict(list)
    for diff,project in zip(differences,project_ids):groups[project].append(float(diff))
    keys=list(groups)
    if not keys:raise ValueError('No projects')
    arrays=[np.asarray(groups[k]) for k in keys]
    draws=[]
    for _ in range(resamples):
        which=rng.integers(0,len(arrays),len(arrays))
        draw=np.concatenate([arrays[k] for k in which])
        draws.append(float(np.mean(draw)))
    return tuple(float(x) for x in np.quantile(draws,[.025,.975]))


def read_runs(folder):
    """Find summary.json + corresponding per_bug_metrics.csv recursively."""
    found=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for path in Path(folder).rglob('summary.json'):
        per=path.parent/'per_bug_metrics.csv'
        if not per.exists():continue
        meta=json.loads(path.read_text())
        method=meta.get('variant',meta.get('method',path.parent.name))
        seed=str(meta.get('seed','unknown'))
        with per.open(newline='',encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if str(r.get('valid','True')).lower()=='false':continue
                found[method][r['uid']][seed].append(r)
    return found


def _aggregate(records):
    """First combine folds/repeated rows within seed; then average five seeds per bug."""
    ret={}
    for method,by_uid in records.items():
        ret[method]={}
        for uid,by_seed in by_uid.items():
            seedavg={s:{m:float(np.mean([float(r[m]) for r in rows if r.get(m) not in ('',None)]))
                         for m in METRICS if all(r.get(m) not in ('',None) for r in rows)}
                     for s,rows in by_seed.items()}
            keys=set.intersection(*(set(z) for z in seedavg.values()))
            ret[method][uid]={'project':next(iter(by_seed.values()))[0]['project'],
                              'dataset':next(iter(by_seed.values()))[0]['dataset'],
                              'seed_count':len(seedavg),
                              **{m:float(np.mean([v[m] for v in seedavg.values()])) for m in keys}}
    return ret


def run_stats(input_dir,reference='full',out='results/stats',resamples=10000,seed=11):
    from scipy.stats import wilcoxon
    result=_aggregate(read_runs(input_dir))
    if reference not in result:raise ValueError(f'No reference method {reference!r}; found {list(result)}')
    rows=[];pvals=[]
    for comparator,data in sorted(result.items()):
        if comparator==reference:continue
        uids=sorted(set(result[reference]) & set(data))
        if len(uids)<2:continue
        for metric in METRICS:
            common=[u for u in uids if metric in result[reference][u] and metric in data[u]]
            if len(common)<2:continue
            diff=np.asarray([result[reference][u][metric]-data[u][metric] for u in common])
            projects=[result[reference][u]['dataset']+'::'+result[reference][u]['project'] for u in common]
            if not np.any(abs(diff)>1e-12):p=1.
            else:
                try:p=float(wilcoxon(diff,alternative='two-sided',zero_method='wilcox').pvalue)
                except ValueError:p=1.
            low,high=project_cluster_ci(diff,projects,resamples,seed)
            rows.append({'reference':reference,'comparison':comparator,'metric':metric,
                         'paired_bugs':len(common),'projects':len(set(projects)),
                         'seed_count_reference_min':min(result[reference][u]['seed_count'] for u in common),
                         'seed_count_comparator_min':min(data[u]['seed_count'] for u in common),
                         'mean_paired_gain':float(np.mean(diff)),
                         'CI95_low':low,'CI95_high':high,
                         'wilcoxon_p':p,'rank_biserial':paired_rank_biserial(diff)})
            pvals.append(p)
    if not rows: raise ValueError('No paired methods with >=2 matched bugs; provide multiple result directories')
    corrections=holm(pvals)
    for r,p in zip(rows,corrections):r['holm_p']=p;r['significant_0_05']=bool(p<.05)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'paired_stats.csv',rows)
    overview={'reference':reference,'comparisons':sorted({r['comparison'] for r in rows}),
              'tests':len(rows),'resamples':resamples,'statistics_unit':'bug averaged over seeds',
              'correction_scope':'all comparators and all reported metrics'}
    write_json(out/'statistical_summary.json',overview)
    return overview
