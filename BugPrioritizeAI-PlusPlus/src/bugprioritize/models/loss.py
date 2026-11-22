"""Classification, log-priority pairwise hinge, cost and explicit L2 regularisation."""
from __future__ import annotations
from collections import defaultdict
import torch
from torch.nn import functional as F


def hybrid_loss(model, output, rows, ranking_weight=1., cost_weight=.5,
                l2_weight=1e-5, positive_weight=5., margin=1.):
    device=output['logits'].device
    y=torch.tensor([r['label'] for r in rows],dtype=torch.float32,device=device)
    costs=torch.tensor([r['cost_norm']*(1-r['cost_missing']) for r in rows],device=device)
    # Per-bug averages prevent large candidate suites dominating training.
    groups=defaultdict(list)
    for i,r in enumerate(rows): groups[r['uid']].append(i)
    cls_terms=[]; cost_terms=[]; rank_terms=[]
    for indices in groups.values():
        ids=torch.tensor(indices,device=device)
        cls_terms.append(F.binary_cross_entropy_with_logits(output['logits'][ids],y[ids],
                         pos_weight=torch.tensor(positive_weight,device=device)))
        cost_terms.append(((1-y[ids])*output['probability'][ids]*costs[ids]).mean())
        positive=ids[y[ids]>0.5]; negative=ids[y[ids]<0.5]
        if len(positive) and len(negative):
            d=output['log_priority'][positive][:,None]-output['log_priority'][negative][None,:]
            rank_terms.append(F.relu(margin-d).mean())
    cls=torch.stack(cls_terms).mean()
    cost=torch.stack(cost_terms).mean()
    rank=torch.stack(rank_terms).mean() if rank_terms else cls*0.
    if 'cost_all' in getattr(model,'ablate',set()) or model.variant=='bugprioritize':
        cost_weight=0.
    l2=sum((p**2).sum() for n,p in model.named_parameters() if p.requires_grad and p.ndim>1)
    total=cls+ranking_weight*rank+cost_weight*cost+l2_weight*l2
    return total,{'classification':float(cls.detach()),'ranking':float(rank.detach()),
                  'cost':float(cost.detach()),'l2':float(l2.detach()),'total':float(total.detach())}
