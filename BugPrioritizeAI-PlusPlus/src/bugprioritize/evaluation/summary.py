from __future__ import annotations
import csv
import json
import numpy as np
from pathlib import Path
from ..utils import write_json,write_csv
from .metrics import metrics,mean_metrics,mrr_topk_bound


def evaluate_saved_rankings(rankings,out):
    paths=sorted(Path(rankings).rglob('*.csv'))
    if not paths:raise ValueError('No ranking CSV files found')
    per=[]
    for path in paths:
        with path.open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        if not rows or int(rows[0].get('label','-1'))<0:
            raise ValueError(f'Ranking {path} has no verified labels; cannot evaluate inference-only files')
        met=metrics(rows)
        random_perms=Path(rankings).parent/'random_permutations'/f'{path.stem}.json'
        if random_perms.exists():
            record=json.loads(random_perms.read_text())
            by_id={r['test_id']:r for r in rows}
            recomputed=[]
            for order in record['orders']:
                # Preserve order via unique descending synthetic scores; the same metrics routine is used.
                ranked=[{**by_id[test_id],'score':len(order)-i} for i,test_id in enumerate(order)]
                recomputed.append(metrics(ranked))
            met={k:float(np.mean([r[k] for r in recomputed])) for k in recomputed[0]
                 if k in ('APFD','APFDc','MRR') or k.startswith(('Top@','NDCG@','Recall@'))}
            met.update({'valid':True,'n':len(rows),'positives':sum(int(r['label']) for r in rows)})
        per.append({'uid':rows[0]['uid'],'project':rows[0]['project'],'dataset':rows[0]['dataset'],**met})
    summary=mean_metrics(per);mrr_topk_bound(per)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'per_bug_metrics.csv',per)
    write_json(out/'summary.json',summary)
    return summary
