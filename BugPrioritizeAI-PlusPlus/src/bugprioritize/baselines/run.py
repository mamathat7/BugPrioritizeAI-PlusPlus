
from __future__ import annotations
from collections import Counter
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..utils import tokenize_identifiers, stable_hash, write_csv, write_json, seed_all
from ..train import load_split_data,write_ranking
from ..datasets.features import NumericTransform
from ..evaluation.metrics import metrics,mean_metrics,order_rows,mrr_topk_bound


def lexical_features(row):
    b=tokenize_identifiers(row['bug_text']);d=tokenize_identifiers(' '.join(row['diff'].values()))
    t=tokenize_identifiers(row['test_text']);p=tokenize_identifiers(row['test_path'])
    sets=[set(x) for x in (b,d,t,p)];sb,sd,st,sp=sets
    intersection=lambda a,z:len(a&z)
    jaccard=lambda a,z:len(a&z)/max(1,len(a|z))
    def overlap(a,z):return len(a&z)/max(1,len(a))
    feats=[jaccard(sb,st),jaccard(sd,st),jaccard(sb,sp),jaccard(sd,sp),
           overlap(sb,st),overlap(sd,st),overlap(st,sb),overlap(st,sd),
           intersection(sb,st),intersection(sd,st),intersection(sb,sp),intersection(sd,sp),
           len(sb),len(sd),len(st),len(sp)]
    return [float(x) for x in feats]


def handcrafted(row):
    # Exactly 42 numeric features: history(6), cost(2), lexical(16), extras(18).
    h=list(row['raw_history']); c=[row['cost_norm'],row['cost_missing']]
    lex=lexical_features(row)
    tokens=tokenize_identifiers(row['test_text']);bf=tokenize_identifiers(row['bug_text'])
    df=tokenize_identifiers(' '.join(row['diff'].values()))
    file=row['test_path']
    ext=[len(row['bug_text']),len(df),len(tokens),len(file),
         len(set(tokens)),len(set(df)),len(set(bf)),
         math.log1p(len(row['bug_text'])), math.log1p(len(df)),math.log1p(len(tokens)),
         float(file.endswith('.py')),float(file.endswith('.java')),float('test' in file.lower()),
         float('error' in tokens),float('exception' in tokens),
         float('boundary' in tokens),float('bug' in tokens),float('fix' in tokens)]
    assert len(h+c+lex+ext)==42
    return np.asarray(h+c+lex+ext,dtype=np.float32)


def bm25_scores(rows,k1=1.2,b=.75):
    docs=[tokenize_identifiers(r['test_text']) for r in rows]
    df=Counter(w for doc in docs for w in set(doc));N=len(docs)
    avg=max(1e-6,np.mean([len(doc) for doc in docs]))
    query=tokenize_identifiers(rows[0]['bug_text']+' '+' '.join(rows[0]['diff'].values()))
    unique_query=Counter(query)
    out=[]
    for doc in docs:
        f=Counter(doc);score=0.
        for word,qfreq in unique_query.items():
            if not f[word]:continue
            idf=math.log(1+(N-df[word]+.5)/(df[word]+.5))
            score+=idf*f[word]*(k1+1)/(f[word]+k1*(1-b+b*len(doc)/avg))*min(qfreq,3)
        out.append(score)
    return out


def _features(groups):
    feats=[];labs=[];lengths=[]
    for group in groups:
        feats.extend(handcrafted(x) for x in group)
        labs.extend(int(r['label']) for r in group)
        lengths.append(len(group))
    return np.stack(feats),np.array(labs,dtype=int),lengths


class GRUComparator(nn.Module):
    def __init__(self, input_dim=42, hidden=32):
        super().__init__()
        self.gru=nn.GRU(input_size=1,hidden_size=hidden,batch_first=True)
        self.mlp=nn.Sequential(nn.Linear(input_dim+hidden,64),nn.ReLU(),nn.Linear(64,1))

    def forward(self,x,seq):
        _,h=self.gru(seq)
        return self.mlp(torch.cat([x,h[-1]],dim=1)).squeeze(-1)


def verdict_matrix(groups):
    arr=[]
    for g in groups:
        for r in g:
            seq=list(r.get('history_sequence',[]))[-50:]
            arr.append([0.]*(50-len(seq))+list(map(float,seq)))
    return torch.tensor(arr,dtype=torch.float32).unsqueeze(-1)


def _fit_gru(groups,epochs,seed):
    seed_all(seed);model=GRUComparator();optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    for epoch in range(epochs):
        for g in groups:
            x=torch.tensor(np.stack([handcrafted(r) for r in g]))
            s=verdict_matrix([g]);y=torch.tensor([r['label'] for r in g],dtype=torch.float32)
            optimizer.zero_grad()
            logits=model(x,s)
            loss=F.binary_cross_entropy_with_logits(logits,y,pos_weight=torch.tensor(5.))
            loss.backward();optimizer.step()
    model.eval();return model


def _fit_retecs(groups,epochs,seed):
    """RETECS-style policy over test features, trained with a cost-aware first-failure reward."""
    seed_all(seed);model=nn.Sequential(nn.Linear(42,64),nn.Tanh(),nn.Linear(64,1))
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    baseline=0.
    rng=random.Random(seed)
    for epoch in range(epochs):
        shuffled=list(groups);rng.shuffle(shuffled)
        for group in shuffled:
            features=torch.tensor(np.stack([handcrafted(r) for r in group]))
            scores=model(features).squeeze(-1)
            remain=list(range(len(group))); logps=[];order=[]
            while remain:
                logit=scores[remain]
                dis=torch.distributions.Categorical(logits=logit)
                j=dis.sample();logps.append(dis.log_prob(j));chosen=remain.pop(int(j));order.append(chosen)
            cost=[float(group[j]['cost_raw']) for j in order]
            first=next((i for i,j in enumerate(order) if group[j]['label']==1),len(order)-1)
            total=max(1e-6,sum(cost))
            reward=1.-(sum(cost[:first])+.5*cost[first])/total
            advantage=reward-baseline; baseline=.95*baseline+.05*reward
            # Reward-weighted policy-gradient loss; detach reward, not policy.
            loss=-float(advantage)*torch.stack(logps).sum()/len(logps)
            optimizer.zero_grad();loss.backward();optimizer.step()
    model.eval();return model


def _fit_lambdamart(groups,seed,trials=0,val_groups=None,prepared=None):
    """TPE tuning is performed exclusively on held-out *validation* bugs."""
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError('Install with: pip install -e .[ranking]') from e
    if prepared is None:
        X,y,g=_features(groups)
        va=[np.stack([handcrafted(r) for r in group]) for group in (val_groups or [])]
    else:
        X,y,g,va=prepared
    kwargs=dict(n_estimators=100,learning_rate=.05,num_leaves=15,min_child_samples=5,
                reg_lambda=1.,random_state=seed,verbosity=-1)
    if trials:
        try:import optuna
        except ImportError as e:raise ImportError('Optuna required: pip install -e .[ranking]') from e
        if not val_groups:raise ValueError('Hyperparameter tuning needs held-out validation bugs')
        def objective(trial):
            cfg={**kwargs, 'num_leaves':trial.suggest_int('num_leaves',7,40),
                 'learning_rate':trial.suggest_float('learning_rate',.015,.2,log=True),
                 'n_estimators':trial.suggest_int('n_estimators',40,150),
                 'min_child_samples':trial.suggest_int('min_child_samples',3,20)}
            mod=lgb.LGBMRanker(**cfg);mod.fit(X,y,group=g)
            per=[]
            for rows,feat in zip(val_groups,va):
                if any(r['label'] for r in rows):
                    scored=[{**r,'score':float(z)} for r,z in zip(rows,mod.predict(feat))]
                    per.append(metrics(scored)['APFDc'])
            return float(np.mean(per)) if per else -1.
        study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective,n_trials=trials,show_progress_bar=False)
        kwargs.update(study.best_params)
    model=lgb.LGBMRanker(**kwargs);model.fit(X,y,group=g)
    return model


def _make_mm_encoder(model, train_groups, groups, split_protocol, hidden, topk):
    """Shared, previously trained encoder + training-only memory + train-only PCA.

    Returns held-out PCA features; no target labels are used. This comparator requires
    a full-model checkpoint trained with the identical split and training source.
    """
    from ..models.memory import RetrievalMemory
    from sklearn.decomposition import PCA
    memory=RetrievalMemory.build(model,train_groups,k=topk)
    model.eval()
    def projected(all_groups):
        raw=[]
        for group in all_groups:
            vecs=[]
            with torch.no_grad():
                for start in range(0,len(group),64):
                    batch=group[start:start+64]
                    c=model.encode_components(batch)
                    fetch,keys,present=memory.retrieve(c['key'],batch,split_protocol)
                    b,n,dim=fetch.shape
                    hist=model.memory_proj(fetch.reshape(-1,dim)).reshape(b,n,-1)
                    scores=(c['key'].unsqueeze(1)*keys).sum(-1).masked_fill(~present,-1e4)
                    w=scores.softmax(1)*present.float()
                    w=w/w.sum(1,keepdim=True).clamp(min=1e-8)
                    retrieved=(w.unsqueeze(-1)*hist).sum(1)
                    modalities=[c[k] for k in ('bug','diff','test','history','cost')]+[retrieved]
                    masks=c['availability'].clone();masks[:,5]=present.any(1).float()
                    blocks=[(z*masks[:,j,None]).cpu().numpy() for j,z in enumerate(modalities)]
                    vecs.extend([np.concatenate([np.stack(blocks,axis=1)[j].reshape(-1),
                                                     masks[j].cpu().numpy()]) for j in range(b)])
            raw.append(np.stack(vecs))
        return raw
    tr=projected(train_groups)
    other={kind:projected(val) for kind,val in groups.items()}
    all_train=np.concatenate(tr)
    pcas=[]
    for i in range(6):
        block=all_train[:,i*hidden:(i+1)*hidden]
        ncomp=min(64,hidden,len(block))
        pca=PCA(n_components=ncomp,random_state=11).fit(block)
        pcas.append(pca)
    def apply(raw):
        return [np.concatenate([*[pca.transform(a[:,j*hidden:(j+1)*hidden])
                                    for j,pca in enumerate(pcas)],a[:,-6:]],axis=1).astype(np.float32)
                for a in raw]
    return apply(tr),{k:apply(v) for k,v in other.items()}


def _write_baseline_rows(out_dir,bug,rows,score,method,rep=0):
    prediction=[]
    for r,s in zip(rows,score):
        prediction.append({**{k:r[k] for k in ('uid','dataset','project','test_id','order',
                         'label','observed_cost','cost_raw','cost_norm','source')},
                         'score':float(s),'probability':float('nan'),
                         'failure_frequency':r['raw_history'][0],
                         'modality_attention':'[]','retrieval_available':False})
    rank=order_rows(prediction)
    if rep==0:
        write_ranking(Path(out_dir)/'rankings'/f'{stable_hash(bug["uid"]):016x}.csv',rank)
    return rank


def run_baseline(data_file,split_file,out_dir,method='bm25',seed=11,epochs=10,max_bugs=0,trials=0,encoder_checkpoint=None):
    from ..evaluation.metrics import mean_metrics
    seed_all(seed);split,parts=load_split_data(data_file,split_file,max_bugs)
    trans=NumericTransform.fit(parts['train'])
    groups={k:[trans.transform(b) for b in v] for k,v in parts.items()}
    model=None
    mm_test=None
    if method=='mm_lambdamart':
        from ..train import load_checkpoint
        from ..utils import sha256_file
        if not encoder_checkpoint:
            raise ValueError('mm_lambdamart requires --encoder-checkpoint of full model trained on this same split')
        base,trained_transform,chk=load_checkpoint(encoder_checkpoint)
        if chk['split_sha256']!=sha256_file(split_file) or chk['data_sha256']!=sha256_file(data_file):
            raise ValueError('Multimodal encoder checkpoint split/data does not match')
        trans=trained_transform
        groups={k:[trans.transform(b) for b in v] for k,v in parts.items()}
        mm_train,mm_groups=_make_mm_encoder(base,groups['train'],{'val':groups['val'],'test':groups['test']},
                                          split['protocol'],base.hidden,chk['options'].get('topk',10))
        xx=np.concatenate(mm_train);yy=np.array([r['label'] for g in groups['train'] for r in g])
        gg=[len(g) for g in groups['train']]
        model=_fit_lambdamart(groups['train'],seed,trials,groups['val'],
                              prepared=(xx,yy,gg,mm_groups['val']))
        mm_test=mm_groups['test']
    elif method=='lambdamart':
        model=_fit_lambdamart(groups['train'],seed,trials,groups['val'])
    elif method=='gru':model=_fit_gru(groups['train'],epochs,seed)
    elif method=='retecs':model=_fit_retecs(groups['train'],epochs,seed)
    elif trials:raise ValueError('Only learning based methods support trials')
    per=[];out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    for bug_index,(bug,rows) in enumerate(zip(parts['test'],groups['test'])):
        if method=='random':
            replicates=[];permutations=[]
            for repetition in range(100):
                rng=random.Random(stable_hash(f'{seed}:{bug["uid"]}:{repetition}'))
                idx=list(range(len(rows)));rng.shuffle(idx)
                score=[0.]*len(rows)
                for i,j in enumerate(idx):score[j]=float(len(rows)-i)
                ranked=_write_baseline_rows(out_dir,bug,rows,score,method,rep=repetition)
                replicates.append(metrics(ranked));permutations.append([r['test_id'] for r in ranked])
            met={k:float(np.mean([r[k] for r in replicates])) for k in replicates[0]
                 if k in ('APFD','APFDc','MRR') or k.startswith(('Top@','NDCG@','Recall@'))}
            met.update({'valid':True,'n':len(rows),'positives':sum(r['label'] for r in rows)})
            path=out_dir/'random_permutations';path.mkdir(exist_ok=True)
            write_json(path/f'{stable_hash(bug["uid"]):016x}.json',{'uid':bug['uid'],'orders':permutations})
        else:
            if method=='original':scores=[-float(r['order']) for r in rows]
            elif method=='history':scores=[float(r['raw_history'][1]) for r in rows]
            elif method=='cost':scores=[-float(r['cost_raw']) for r in rows]
            elif method=='bm25':scores=bm25_scores(rows)
            elif method=='lambdamart':
                scores=model.predict(np.stack([handcrafted(r) for r in rows]))
            elif method=='mm_lambdamart':
                scores=model.predict(mm_test[bug_index])
            elif method=='gru':
                with torch.no_grad():scores=model(torch.tensor(np.stack([handcrafted(r) for r in rows])),
                                                 verdict_matrix([rows])).numpy()
            elif method=='retecs':
                with torch.no_grad():scores=model(torch.tensor(np.stack([handcrafted(r) for r in rows]))).squeeze(-1).numpy()
            else:raise ValueError(method)
            ranked=_write_baseline_rows(out_dir,bug,rows,scores,method)
            met=metrics(ranked)
        per.append({'uid':bug['uid'],'project':bug['project'],'dataset':bug['dataset'],**met})
    summary=mean_metrics(per)
    if method!='random':mrr_topk_bound(per)
    write_csv(out_dir/'per_bug_metrics.csv',per)
    report={'method':method,'seed':seed,'split':split['id'],'protocol':split['protocol'],
            'test':summary,'debug_subset':bool(max_bugs),'note':
            ('RETECS-style independent policy-gradient implementation, not exact published reproduction' if method=='retecs' else
             'Shared trained encoder / retrieval feature PCA (training data only) + LambdaMART' if method=='mm_lambdamart' else
             'Synthetic demo results, if supplied, are NOT publication results' )}
    write_json(out_dir/'summary.json',report)
    return report
