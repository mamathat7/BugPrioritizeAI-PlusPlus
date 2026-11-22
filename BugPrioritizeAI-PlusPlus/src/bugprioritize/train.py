"""Train/evaluate grouped bug-test ranking with leak-free retrieval memory."""
from __future__ import annotations
import csv
import json
import random
from pathlib import Path
import torch
from .utils import read_jsonl, write_json, write_csv, seed_all, sha256_file, env_summary, stable_hash
from .datasets.splits import validate_split
from .datasets.features import NumericTransform
from .evaluation.metrics import metrics, mean_metrics, order_rows, mrr_topk_bound
from .models.network import BugPrioritizeModel
from .models.memory import RetrievalMemory
from .models.loss import hybrid_loss


def load_split_data(data_file, split_file, max_bugs=0):
    bugs=list(read_jsonl(data_file))
    split=json.loads(Path(split_file).read_text())
    validate_split(bugs,split)
    by_uid={b['uid']:b for b in bugs}
    parts={part:[by_uid[u] for u in split[part]] for part in ('train','val','test')}
    if max_bugs:
        parts={k:v[:max_bugs] for k,v in parts.items()}
    return split,parts


def batches(train_groups, batch_size, seed):
    """Keep at least one positive and one negative within each eligible mini-batch."""
    rng=random.Random(seed)
    groups=list(train_groups)
    rng.shuffle(groups)
    for group in groups:
        pos=[r for r in group if r['label']==1]
        neg=[r for r in group if r['label']==0]
        rng.shuffle(pos); rng.shuffle(neg)
        if pos and neg:
            # Cycle positives through batches; avoid empty ranking constraints.
            chunk=max(1,batch_size-1)
            for j,start in enumerate(range(0,len(neg),chunk)):
                p=pos[j%len(pos)]
                block=neg[start:start+chunk]+[p]
                rng.shuffle(block)
                yield block
            # Include any positive beyond those reached in the cycling pass.
            if len(pos)>len(neg)//chunk+1:
                for start in range(0,len(pos),batch_size):
                    yield pos[start:start+batch_size]
        else:
            all_rows=group.copy();rng.shuffle(all_rows)
            for start in range(0,len(all_rows),batch_size):
                yield all_rows[start:start+batch_size]


def predict_bug(model, rows, memory=None, protocol='projectwise', batch_size=128,
                no_cost_discount=False):
    model.eval(); outputs=[]
    with torch.no_grad():
        for i in range(0,len(rows),batch_size):
            chunk=rows[i:i+batch_size]
            pred=model(chunk,memory,protocol,no_cost_discount=no_cost_discount)
            for j,r in enumerate(chunk):
                outputs.append({**{k:r[k] for k in ('uid','dataset','project','test_id','order',
                                          'label','observed_cost','cost_raw','cost_norm','source')},
                                'failure_frequency':(0. if model.variant=='history_all' else r['raw_history'][0]),
                                'probability':float(pred['probability'][j].cpu()),
                                'score':float(pred['score'][j].cpu()),
                                'modality_attention':json.dumps([round(float(x),6) for x in pred['attention'][j].cpu().tolist()]),
                                'retrieval_available':bool(pred['context_mask'][j].cpu())})
    return order_rows(outputs)


def write_ranking(path, rows):
    fields=['uid','dataset','project','test_id','rank','order','probability','score',
            'label','observed_cost','cost_raw','cost_norm','failure_frequency',
            'modality_attention','retrieval_available','source']
    write_csv(path,[{**r,'rank':i+1} for i,r in enumerate(rows)],fields)


def evaluate_model(model, bugs, transform, memory, protocol, out_dir=None, mode='rkd', batch_size=128, no_cost_discount=False):
    per_bug=[]
    for b in bugs:
        rows=transform.transform(b)
        ranking=predict_bug(model,rows,memory,protocol,batch_size,no_cost_discount=no_cost_discount)
        met=metrics(ranking)
        per_bug.append({'uid':b['uid'],'project':b['project'],'dataset':b['dataset'],**met})
        if out_dir is not None:
            name=f"{stable_hash(b['uid']):016x}.csv"
            write_ranking(Path(out_dir)/name,ranking)
    if not per_bug: raise ValueError('Empty evaluation set')
    return per_bug,mean_metrics(per_bug)


def save_checkpoint(path, model, transform, opts, split_file, data_file, seed, epoch, validation):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(), 'model_config':model.config_dict(),
                'normalisation':transform.serialize(), 'seed':seed,
                'epoch':epoch,'validation':validation,'options':opts,
                'split_sha256':sha256_file(split_file),'data_sha256':sha256_file(data_file),
                'environment':env_summary()},path)


def load_checkpoint(checkpoint, device='cpu'):
    """Load only trusted locally-produced PyTorch checkpoints (pickle-based)."""
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=BugPrioritizeModel(**state['model_config'])
    model.load_state_dict(state['state_dict']); model.to(device).eval()
    transform=NumericTransform.deserialize(state['normalisation'])
    return model,transform,state


def train_one(data_file,split_file,out_dir,seed=11,encoder='hash',hidden=256,
              variant='full',checkpoint='microsoft/codebert-base',device='auto',
              epochs=50,patience=8,batch_size=32,lr=1e-4,dropout=.2,topk=10,
              margin=1.,ranking_weight=1.,cost_weight=.5,l2_weight=1e-5,
              positive_weight=5.,max_bugs=0,memory_cap=0,use_faiss=False,vocab=65536):
    if variant=='cost_scoring':
        raise ValueError('cost_scoring is inference-only: run `ablate` with full followed by cost_scoring')
    if device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'
    seed_all(seed)
    split,parts=load_split_data(data_file,split_file,max_bugs)
    transform=NumericTransform.fit(parts['train'])
    grouped={part:[transform.transform(b) for b in bs] for part,bs in parts.items()}
    model=BugPrioritizeModel(encoder=encoder,hidden=hidden,dropout=dropout,variant=variant,
                             checkpoint=checkpoint,vocab=vocab).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    best=-float('inf'); best_epoch=-1; best_memory=None; bad=0; logs=[]
    protocol=split['protocol']; path=out_dir/'model.pt'
    opts={'encoder':encoder,'hidden':hidden,'variant':variant,'epochs':epochs,'patience':patience,
          'batch_size':batch_size,'lr':lr,'topk':topk,'margin':margin,'ranking_weight':ranking_weight,
          'cost_weight':cost_weight,'l2_weight':l2_weight,'positive_weight':positive_weight,
          'debug_max_bugs':max_bugs,'memory_cap':memory_cap,'vocab':vocab,
          'checkpoint':checkpoint,'dropout':dropout,'split_file':str(split_file)}
    if not any(any(r['label']==1 for r in group) for group in grouped['train']):
        raise ValueError('Train split has no confirmed positives')
    for epoch in range(1,epochs+1):
        # Cache keys/values after updating the trainable encoders on previous epoch.
        memory=RetrievalMemory.build(model,grouped['train'],k=topk,
                 use_faiss=use_faiss,max_items=memory_cap) if variant not in ('bugprioritize',) and variant!='retrieval' else None
        model.train()
        losses=[]
        for step,block in enumerate(batches(grouped['train'],batch_size,seed+epoch)):
            opt.zero_grad(set_to_none=True)
            pred=model(block,memory,protocol)
            loss,parts_loss=hybrid_loss(model,pred,block,ranking_weight,cost_weight,l2_weight,
                                        positive_weight,margin)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step(); losses.append(parts_loss['total'])
        model.eval()
        _,val=evaluate_model(model,parts['val'],transform,memory,protocol,batch_size=128)
        item={'epoch':epoch,'train_loss':float(sum(losses)/len(losses)),
              'val_APFD':val['APFD'],'val_APFDc':val['APFDc'],'val_MRR':val['MRR']}
        logs.append(item)
        if val['APFDc']>best+1e-10:
            best=val['APFDc'];best_epoch=epoch;bad=0
            save_checkpoint(path,model,transform,opts,split_file,data_file,seed,epoch,val)
        else: bad+=1
        if bad>=patience: break
    write_csv(out_dir/'training_log.csv',logs)
    best_model,trans,state=load_checkpoint(path,device)
    # Memory corresponding to the BEST model, never to a later overfit model.
    best_memory=RetrievalMemory.build(best_model,grouped['train'],k=topk,
                    use_faiss=use_faiss,max_items=memory_cap) if variant not in ('bugprioritize','retrieval') else None
    audit=[]
    if best_memory is not None:
        for group in grouped['test']:
            audit.extend(best_memory.audit(group,protocol))
        if audit: raise AssertionError(f'Retrieval leakage audit failed: {audit[:5]}')
    rankings_dir=out_dir/'rankings'
    per_bug,test=evaluate_model(best_model,parts['test'],trans,best_memory,protocol,rankings_dir)
    mrr_topk_bound(per_bug)
    write_csv(out_dir/'per_bug_metrics.csv',per_bug)
    report={'split':split['id'],'protocol':protocol,'seed':seed,'variant':variant,
            'best_epoch':best_epoch,'validation_APFDc':best,'test':test,
            'leakage_audit_violations':len(audit),
            'memory_duplicates_removed':len(best_memory.dropped_duplicates) if best_memory else 0,
            'candidate_bugs':len(parts['test']),
            'codebert_pretrained':encoder=='codebert','debug_subset':bool(max_bugs)}
    write_json(out_dir/'summary.json',report)
    return report
