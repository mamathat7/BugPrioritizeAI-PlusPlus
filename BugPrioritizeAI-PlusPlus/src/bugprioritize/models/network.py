"""Retrieval-augmented multimodal neural TCP with explicitly masked evidence."""
from __future__ import annotations
import math
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
from ..utils import stable_hash, tokenize_identifiers

MODES = ('bug','diff','test','history','cost','retrieval')
ABLATIONS={
    'full':(),
    'bug_direct':('bug_direct',), 'bug_all':('bug_direct','bug_memory'),
    'diff_direct':('diff_direct',), 'diff_all':('diff_direct','diff_memory'),
    'test_direct':('test_direct',), 'test_all':('test_direct','test_memory'),
    'history_direct':('history_direct',), 'history_all':('history_direct','history_memory'),
    'retrieval':('retrieval',), 'gated_fusion':('gated_fusion',),
    'cost_scoring':('cost_scoring',), 'cost_all':('cost_all',),
}


class HashEncoder(nn.Module):
    """Offline deterministic tokenizer with trainable embeddings; intended for CI/smoke."""
    def __init__(self, dim=128, vocab=65536, limit=128):
        super().__init__()
        self.table=nn.Embedding(vocab+1, dim, padding_idx=0)
        self.output_dim=dim; self.vocab=vocab; self.limit=limit

    def forward(self, strings):
        items=[]
        for s in strings:
            tok=tokenize_identifiers(s)[:self.limit]
            items.append([(stable_hash(w)%self.vocab)+1 for w in tok])
        max_len=max(1,max(map(len,items),default=1))
        ids=torch.zeros((len(strings),max_len),dtype=torch.long,device=self.table.weight.device)
        for i,v in enumerate(items):
            if v: ids[i,:len(v)]=torch.tensor(v,device=ids.device)
        em=self.table(ids)
        mask=(ids!=0).unsqueeze(-1)
        return (em*mask).sum(1)/mask.sum(1).clamp(min=1)


class CodeBERTEncoder(nn.Module):
    """Real shared pretrained CodeBERT: downloads model at first use (not bundled)."""
    def __init__(self, name='microsoft/codebert-base', max_length=192):
        super().__init__()
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError as e:
            raise ImportError('Install with: pip install -e .[codebert]') from e
        self.tokenizer=AutoTokenizer.from_pretrained(name)
        self.transformer=AutoModel.from_pretrained(name)
        self.output_dim=self.transformer.config.hidden_size
        self.max_length=max_length

    def forward(self, strings):
        device=next(self.transformer.parameters()).device
        tok=self.tokenizer([str(x) or '[EMPTY]' for x in strings], padding=True,
                           truncation=True, max_length=self.max_length, return_tensors='pt')
        toks={k:v.to(device) for k,v in tok.items()}
        output=self.transformer(**toks).last_hidden_state
        mask=toks['attention_mask'].unsqueeze(-1)
        return (output*mask).sum(1)/mask.sum(1).clamp(min=1)


class BugPrioritizeModel(nn.Module):
    def __init__(self, encoder='hash', hidden=256, dropout=.2, variant='full',
                 checkpoint='microsoft/codebert-base', vocab=65536):
        super().__init__()
        if variant not in ABLATIONS and variant not in ('bugprioritize','concat'):
            raise ValueError(f'Unknown model variant: {variant}')
        self.variant=variant; self.ablate=set(ABLATIONS.get(variant,()))
        self.encoder_name=encoder; self.checkpoint=checkpoint; self.hidden=hidden; self.vocab=vocab
        if encoder=='codebert':
            self.shared_text=CodeBERTEncoder(checkpoint)
        elif encoder=='hash':
            self.shared_text=HashEncoder(128,vocab=vocab)
        else: raise ValueError(f'Unknown encoder {encoder}')
        self.bug_proj=nn.Sequential(nn.Linear(self.shared_text.output_dim,hidden),nn.LayerNorm(hidden))
        self.diff_proj=nn.Sequential(nn.Linear(self.shared_text.output_dim,hidden),nn.LayerNorm(hidden))
        self.diff_att=nn.Linear(hidden,1)
        self.test_encoder=HashEncoder(128,vocab=vocab,limit=64)
        self.test_proj=nn.Sequential(nn.Linear(128,hidden),nn.LayerNorm(hidden))
        self.history_proj=nn.Sequential(nn.Linear(6,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.LayerNorm(hidden))
        self.cost_proj=nn.Sequential(nn.Linear(2,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.query_proj=nn.Linear(hidden*2,hidden)
        # Training-only memory stores precomputed raw modality vectors and past labels.
        self.memory_proj=nn.Sequential(nn.Linear(hidden*4+1,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.gates=nn.ModuleList(nn.Linear(hidden,hidden) for _ in MODES)
        self.att=nn.Sequential(nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.concat=nn.Sequential(nn.Linear(hidden*6+6,hidden*2),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden*2,hidden))
        self.residual=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,hidden))
        self.norm=nn.LayerNorm(hidden)
        self.head=nn.Sequential(nn.Linear(hidden,hidden//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden//2,1))
        # For the GRU comparator, temporal verdict sequences can be added to rows.
        self.history_gru=nn.GRU(input_size=1,hidden_size=hidden//2,batch_first=True)
        self.gru_out=nn.Linear(hidden//2+3,hidden)

    @property
    def device(self): return next(self.parameters()).device

    def _encode_unique(self, strings, projection, source_encoder):
        unique=list(dict.fromkeys(strings))
        index={x:i for i,x in enumerate(unique)}
        embeddings=projection(source_encoder(unique))
        ind=torch.tensor([index[x] for x in strings],device=self.device)
        return embeddings[ind]

    def encode_components(self, rows):
        """Return five projected input modalities and retrieval keys (no outcomes read)."""
        btxt=[r['bug_text'] for r in rows]
        bug=self._encode_unique(btxt,self.bug_proj,self.shared_text)
        diff_strings=[]
        for r in rows:
            d=r.get('diff',{}) or {}
            diff_strings.extend([str(d.get(part,'') or '') for part in ('added','deleted','context')])
        # Three directional streams with attention over present streams.
        de=self._encode_unique(diff_strings,self.diff_proj,self.shared_text).reshape(len(rows),3,self.hidden)
        exists=torch.tensor([[float(bool((r.get('diff',{}) or {}).get(k,''))) for k in ('added','deleted','context')] for r in rows],device=self.device)
        att=self.diff_att(de).squeeze(-1).masked_fill(exists==0,-1e4)
        weights=att.softmax(dim=1)*exists
        weights=weights/weights.sum(dim=1,keepdim=True).clamp(min=1e-8)
        diff=(weights.unsqueeze(-1)*de).sum(1)
        test=self._encode_unique([r['test_text'] for r in rows],self.test_proj,self.test_encoder)
        history=self.history_proj(torch.tensor([r['history'] for r in rows],device=self.device,dtype=torch.float32))
        cost=self.cost_proj(torch.tensor([[r['cost_norm'],r['cost_missing']] for r in rows],device=self.device,dtype=torch.float32))
        # Memory key is based only on current bug and available diff.
        bkey=bug if 'bug_memory' not in self.ablate else bug*0.
        dkey=diff if 'diff_memory' not in self.ablate else diff*0.
        if not torch.any(exists) and 'bug_memory' in self.ablate:
            # No bug nor diff evidence yields no retrieval query (safe zero vector).
            pass
        key=F.normalize(self.query_proj(torch.cat([bkey,dkey],dim=1)),dim=1)
        availability=torch.tensor([[float(bool(r['bug_text'])),float(bool(any((r.get('diff',{}) or {}).values()))),
                                    float(bool(r['test_text'])),float(r['history_mask']),
                                    1.-float(r['cost_missing']),0.] for r in rows],
                                  device=self.device,dtype=torch.float32)
        return {'bug':bug,'diff':diff,'test':test,'history':history,'cost':cost,
                'key':key,'availability':availability}

    def memory_raw(self, c, rows):
        bug=c['bug'].detach()
        diff=c['diff'].detach()
        test=c['test'].detach()
        hist=c['history'].detach()
        if 'bug_memory' in self.ablate: bug=bug*0.
        if 'diff_memory' in self.ablate: diff=diff*0.
        if 'test_memory' in self.ablate: test=test*0.
        if 'history_memory' in self.ablate: hist=hist*0.
        ys=torch.tensor([[float(r['label'])] for r in rows],device=self.device)
        return torch.cat([bug,diff,test,hist,ys],dim=1)

    def forward(self, rows, memory=None, protocol='projectwise', no_cost_discount=False):
        c=self.encode_components(rows)
        mask=c['availability'].clone()
        comps=[c[k] for k in ('bug','diff','test','history','cost')]
        for index,key in enumerate(('bug','diff','test','history','cost')):
            if f'{key}_direct' in self.ablate or (key=='cost' and 'cost_all' in self.ablate):
                mask[:,index]=0.
        if self.variant=='bugprioritize':
            no_memory=True
        else:
            no_memory='retrieval' in self.ablate
        if memory is None or no_memory:
            retrieve=torch.zeros_like(c['bug'])
        else:
            # search is done on stop-gradient keys; weights are recomputed differentiably.
            fetched, keys, present=memory.retrieve(c['key'].detach(),rows,protocol)
            if fetched is None:
                retrieve=torch.zeros_like(c['bug'])
            else:
                b,n,_=fetched.shape
                selected=self.memory_proj(fetched.reshape(-1,fetched.shape[-1])).reshape(b,n,-1)
                scores=(c['key'].unsqueeze(1)*keys).sum(-1)
                # invalid entries are masked before softmax.
                scores=scores.masked_fill(~present,-1e4)
                weights=F.softmax(scores,dim=1)*present.to(scores.dtype)
                weights=weights/weights.sum(1,keepdim=True).clamp(min=1e-8)
                retrieve=(weights.unsqueeze(-1)*selected).sum(1)
                mask[:,5]=present.any(1).float()
        comps.append(retrieve)
        x=torch.stack(comps,dim=1)                 # (B, 6, hidden)
        if self.variant=='concat' or 'gated_fusion' in self.ablate:
            fused=self.concat(torch.cat([(x*mask.unsqueeze(-1)).flatten(1),mask],dim=1))
            attention=mask/mask.sum(1,keepdim=True).clamp(min=1)
        else:
            gates=torch.stack([torch.sigmoid(layer(x[:,i])) for i,layer in enumerate(self.gates)],dim=1)
            if self.variant=='bugprioritize': gates=torch.ones_like(gates)
            gated=x*gates*mask.unsqueeze(-1)
            att=self.att(gated).squeeze(-1).masked_fill(mask==0,-1e4)
            attention=att.softmax(dim=1)*mask
            attention=attention/attention.sum(1,keepdim=True).clamp(min=1e-8)
            fused=(gated*attention.unsqueeze(-1)).sum(1)
        z=self.norm(fused+self.residual(fused))
        logits=self.head(z).squeeze(-1)
        prob=logits.sigmoid()
        cost=torch.tensor([r['cost_norm'] for r in rows],device=self.device,dtype=prob.dtype)
        available=mask[:,4]
        if no_cost_discount or 'cost_scoring' in self.ablate or 'cost_all' in self.ablate or self.variant=='bugprioritize':
            gamma=torch.ones_like(cost)
        else:
            gamma=1./(1.+torch.log1p(cost)*available)
        log_priority=F.logsigmoid(logits)+torch.log(gamma.clamp(min=1e-12))
        score=prob*gamma
        return {'logits':logits,'probability':prob,'score':score,
                'log_priority':log_priority,'attention':attention,'availability':mask,
                'key':c['key'],'context_mask':mask[:,5]}

    def config_dict(self):
        return {'encoder':self.encoder_name,'hidden':self.hidden,'variant':self.variant,'checkpoint':self.checkpoint,'vocab':self.vocab}
