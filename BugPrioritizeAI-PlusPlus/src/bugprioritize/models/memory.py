"""Exact, leakage-filtered historical context retrieval with an optional FAISS index."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from ..utils import date_value, tokenize_identifiers


class RetrievalMemory:
    def __init__(self, keys, raw, info, device='cpu', k=10, use_faiss=False):
        self.keys=np.asarray(keys,dtype=np.float32)
        self.raw=torch.as_tensor(np.asarray(raw,dtype=np.float32),dtype=torch.float32,device=device)
        self.key_tensor=torch.from_numpy(self.keys.copy()).to(device)
        self.info=list(info); self.k=int(k); self.device=device
        if len(self.keys)!=len(self.info): raise ValueError('Memory key/info length mismatch')
        self.index=None
        if use_faiss and len(self.keys):
            try:
                import faiss
                self.index=faiss.IndexFlatIP(self.keys.shape[1]); self.index.add(self.keys)
            except ImportError as e:
                raise ImportError('FAISS requested: pip install -e .[faiss]') from e

    @staticmethod
    def deduplicate_bug_groups(groups):
        """Memory-only near-duplicate suppression, not a silent dataset exclusion.
        SHA equality is authoritative; patch Jaccard >= .90 requires >=20 tokens.
        A missing patch (PRO) is never treated as a duplicate.
        """
        keep=[]; removed=[]; known_shas=set(); known_patches=[]
        for bug in sorted(groups,key=lambda g:date_value(g[0].get('memory_timestamp',g[0]['timestamp']))):
            r=bug[0];sha=str(r.get('fix_sha') or '').strip()
            patch=' '.join(str(v) for v in (r.get('diff',{}) or {}).values())
            tokens=set(tokenize_identifiers(patch))
            if sha and sha in known_shas:
                removed.append({'uid':r['uid'],'reason':'repeated_fix_sha'});continue
            duplicate=False
            if len(tokens)>=20:
                for prev in known_patches:
                    if len(prev)>=20 and len(tokens&prev)/max(1,len(tokens|prev))>=.9:
                        duplicate=True;break
            if duplicate:
                removed.append({'uid':r['uid'],'reason':'near_duplicate_patch'});continue
            keep.append(bug)
            if sha:known_shas.add(sha)
            if len(tokens)>=20:known_patches.append(tokens)
        return keep, removed

    @classmethod
    @torch.no_grad()
    def build(cls, model, bug_rows, k=10, batch_size=128, use_faiss=False, max_items=0):
        """Index TRAINING entries only. Caller controls the split of bug_rows.
        Memory bug labels are permitted; a query can never retrieve its own bug.
        """
        old=model.training; model.eval()
        bug_rows, removed=cls.deduplicate_bug_groups(bug_rows)
        keys=[]; raw=[]; info=[]
        for bug in bug_rows:
            for start in range(0,len(bug),batch_size):
                chunk=bug[start:start+batch_size]
                c=model.encode_components(chunk)
                keys.append(c['key'].cpu().numpy())
                raw.append(model.memory_raw(c,chunk).cpu().numpy())
                for r in chunk:
                    info.append({'uid':r['uid'],'dataset':r['dataset'],'project':r['project'],
                                 'timestamp':date_value(r.get('memory_timestamp',r['timestamp'])),
                                 'test_id':r['test_id']})
                if max_items and len(info)>=max_items: break
            if max_items and len(info)>=max_items: break
        model.train(old)
        if not keys:
            result=cls(np.zeros((0,model.hidden)),np.zeros((0,model.hidden*4+1)),[],device=model.device,k=k)
            result.dropped_duplicates=removed
            return result
        key=np.concatenate(keys); val=np.concatenate(raw)
        if max_items: key=key[:max_items];val=val[:max_items];info=info[:max_items]
        result=cls(key,val,info,device=model.device,k=k,use_faiss=use_faiss)
        result.dropped_duplicates=removed
        return result

    def _eligible(self, row, protocol):
        time=date_value(row['timestamp'])
        valid=np.ones(len(self.info),dtype=bool)
        for j,e in enumerate(self.info):
            if e['uid']==row['uid']:
                valid[j]=False
            elif protocol=='timewise' and not (0<e['timestamp']<time):
                valid[j]=False
            elif protocol=='crossdataset' and e['dataset']==row['dataset']:
                valid[j]=False
            elif protocol=='projectwise' and row.get('forbid_project_memory',False) and e['project']==row['project'] and e['dataset']==row['dataset']:
                valid[j]=False
        return valid

    def retrieve(self, queries, rows, protocol='projectwise'):
        if not len(self.info): return None,None,None
        qs=queries.detach().cpu().numpy().astype(np.float32)
        # All candidates of a bug share one query; compute once, then broadcast.
        ids=[]; uniq={}
        for i,row in enumerate(rows):
            u=row['uid']
            if u not in uniq:
                valid=self._eligible(row,protocol)
                if self.index is not None:
                    # FAISS fast-path; when exclusions discard too many hits, exact fallback.
                    search_size=min(len(self.info),max(self.k*64,256))
                    _, neighbors=self.index.search(qs[i:i+1],search_size)
                    choice=[idx for idx in neighbors[0] if idx>=0 and valid[idx]][:self.k]
                    if len(choice)<min(self.k,int(valid.sum())):
                        choice=[]
                else: choice=[]
                if not choice:
                    candidates=np.flatnonzero(valid)
                    if len(candidates):
                        scores=self.keys[candidates]@qs[i]
                        best=np.argsort(-scores,kind='stable')[:self.k]
                        choice=candidates[best].tolist()
                uniq[u]=choice
            ids.append(uniq[u])
        k=max(1,self.k)
        index=np.zeros((len(rows),k),dtype=np.int64)
        present=np.zeros((len(rows),k),dtype=bool)
        for i,choice in enumerate(ids):
            take=min(len(choice),k)
            if take: index[i,:take]=choice[:take];present[i,:take]=True
        ti=torch.from_numpy(index).to(self.raw.device)
        pm=torch.from_numpy(present).to(self.raw.device)
        return self.raw[ti],self.key_tensor[ti],pm

    def audit(self, rows, protocol):
        errors=[]
        for r in rows:
            if not self._eligible(r,protocol).any(): continue
            eligible=self._eligible(r,protocol)
            if any(self.info[i]['uid']==r['uid'] for i in np.flatnonzero(eligible)):
                errors.append(f'{r["uid"]}: current bug in memory')
            if protocol=='timewise' and any(self.info[i]['timestamp']>=date_value(r['timestamp']) for i in np.flatnonzero(eligible)):
                errors.append(f'{r["uid"]}: future memory')
        return errors
