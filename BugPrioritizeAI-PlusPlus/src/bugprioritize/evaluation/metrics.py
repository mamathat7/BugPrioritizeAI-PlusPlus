"""One-fault-per-bug APFD/APFDc and ranking metrics. No outcome-dependent candidate selection."""
from __future__ import annotations
import math
import numpy as np


def order_rows(rows):
    # Deterministic paper tie break: score desc, cost asc, history freq desc, harness order.
    return sorted(rows, key=lambda r:(-float(r['score']),
                      float(r.get('cost_raw') or 1.),
                      -float(r.get('failure_frequency') or 0.),
                      int(r.get('order',0)),str(r.get('test_id',''))))


def metrics(ranking, ks=(5,10)):
    rows=order_rows(ranking)
    n=len(rows)
    if n==0: raise ValueError('Empty candidate suite')
    positive=[i for i,r in enumerate(rows) if int(r['label'])==1]
    if not positive:
        return {'valid':False, 'n':n,'positives':0}
    first=positive[0]; r=first+1
    costs=[max(1e-6,float(x.get('observed_cost') or x.get('cost_raw') or 1.)) for x in rows]
    total=sum(costs)
    # A single fault can be revealed by multiple tests; use the first positive.
    apfd=1.0-r/n+1.0/(2*n)
    apfdc=1.0-(sum(costs[:first])+0.5*costs[first])/total
    out={'valid':True,'n':n,'positives':len(positive),'first_rank':r,
         'APFD':apfd,'APFDc':apfdc,'MRR':1./r}
    y=[int(x['label']) for x in rows]
    for k in ks:
        top=min(n,k)
        out[f'Recall@{k}']=sum(y[:top])/len(positive)
        out[f'Top@{k}']=float(any(y[:top]))
        dcg=sum(y[i]/math.log2(i+2) for i in range(top))
        idcg=sum(1/math.log2(i+2) for i in range(min(len(positive),top)))
        out[f'NDCG@{k}']=dcg/idcg if idcg else 0.
    return out


def mean_metrics(per_bug):
    rows=[r for r in per_bug if r.get('valid')]
    if not rows: raise ValueError('No bugs with at least one confirmed positive test')
    keys=[k for k in rows[0] if k in ('APFD','APFDc','MRR') or k.startswith(('Recall@','Top@','NDCG@'))]
    return {k:float(np.mean([x[k] for x in rows])) for k in keys} | {
        'bugs_evaluated':len(rows),'bugs_without_positive':len(per_bug)-len(rows)}


def mrr_topk_bound(per_bug):
    mean=mean_metrics(per_bug)
    # First positive at 1..5, 6..10, or >10; upper bounds 1,1/6,1/11.
    upper=mean['Top@5']+(mean['Top@10']-mean['Top@5'])/6.+(1-mean['Top@10'])/11.
    if mean['MRR']>upper+1e-8:
        raise AssertionError(f'MRR-TopK inconsistency: {mean["MRR"]} > {upper}')
    return upper
