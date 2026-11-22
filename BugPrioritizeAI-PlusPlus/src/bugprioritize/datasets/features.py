"""Train-fold-only numerical transforms; never use observed target runtime at inference."""
from __future__ import annotations
import math
import numpy as np
from ..utils import write_json


class NumericTransform:
    def __init__(self, median_cost=1., max_cost=1., h_mean=None, h_sd=None):
        self.median_cost=float(median_cost)
        self.max_cost=float(max_cost)
        self.h_mean=list(h_mean if h_mean is not None else [0]*6)
        self.h_sd=list(h_sd if h_sd is not None else [1]*6)

    @classmethod
    def fit(cls, bugs):
        rows=[t for b in bugs for t in b['tests']]
        if not rows: raise ValueError('Cannot fit preprocessing on an empty train split')
        costs=[float(t['cost_estimate']) for t in rows if t.get('cost_estimate') is not None]
        median=float(np.median(costs)) if costs else 1.
        maximum=float(max(costs)) if costs else 1.
        hs=np.asarray([t.get('history_features',[0.]*6) for t in rows],dtype=np.float64)
        return cls(median,maximum,hs.mean(axis=0),np.maximum(hs.std(axis=0),1e-5))

    def transform(self, bug):
        rows=[]
        # Equation (24): suite-level min-max based on pre-decision estimates ONLY.
        estimates=[float(t['cost_estimate']) if t.get('cost_estimate') is not None and not t.get('cost_missing',False)
                   else self.median_cost for t in bug['tests']]
        lo=min(estimates);hi=max(estimates);span=max(hi-lo,1e-8)
        for t in bug['tests']:
            h=np.asarray(t.get('history_features',[0.]*6),dtype=np.float32)
            hist=((h-np.asarray(self.h_mean,dtype=np.float32))/np.asarray(self.h_sd,dtype=np.float32)).tolist()
            missing=t.get('cost_missing',t.get('cost_estimate') is None)
            cost=float(t['cost_estimate']) if not missing and t.get('cost_estimate') is not None else self.median_cost
            # No observed cost or test verdict enters this normalisation.
            cost_norm=max(0.,min((cost-lo)/span,1.))
            rows.append({'uid':bug['uid'],'dataset':bug['dataset'],'project':bug['project'],
                         'timestamp':bug['timestamp'],'timestamp_numeric':bug.get('timestamp_numeric',0.),
                         'memory_timestamp':bug.get('fix_timestamp',bug['timestamp']),
                         'fix_sha':bug.get('fix_sha',''),
                         'scenario':bug.get('scenario','rkd'),
                         'bug_text':bug.get('bug_text',''),'diff':bug.get('diff',{}),
                         'test_id':t['test_id'], 'test_name':t.get('name',t['test_id']),
                         'test_text':f"{t.get('name','')} {t.get('class','')} {t.get('path','')}",
                         'test_path':t.get('path',''),
                         'history':hist, 'history_sequence':t.get('history_sequence',[]), 'raw_history':h.tolist(), 'history_mask':float(t.get('history_mask',0)),
                         'cost_norm':cost_norm,'cost_raw':cost,'cost_missing':float(missing),
                         'observed_cost':t.get('observed_execution_time'),
                         'label':int(t.get('label',-1)),'order':int(t.get('order',0)),
                         'source':bug.get('source','unreported')})
        return rows

    def serialize(self):
        return {'median_cost':self.median_cost,'max_cost':self.max_cost,
                'h_mean':self.h_mean,'h_sd':self.h_sd}

    @classmethod
    def deserialize(cls,data):
        return cls(**data)
