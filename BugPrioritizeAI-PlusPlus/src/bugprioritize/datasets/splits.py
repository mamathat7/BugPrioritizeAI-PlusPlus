
from __future__ import annotations
from collections import defaultdict
from ..utils import date_value, read_jsonl, write_json, stable_hash


def grouped_folds(bugs, n_folds=5, seed=11):
    """Greedy balanced project bins *within each dataset*, no splitting a project."""
    grouped=defaultdict(lambda:defaultdict(list))
    for b in bugs:
        grouped[b['dataset']][b['project']].append(b['uid'])
    mapping={}
    for dataset, projects in sorted(grouped.items()):
        if len(projects) < n_folds:
            raise ValueError(f'{dataset}: need at least {n_folds} projects for grouped {n_folds}-fold; got {len(projects)}')
        items=sorted(projects.items(), key=lambda kv:(-len(kv[1]), stable_hash(f'{seed}:{kv[0]}')))
        sizes=[0]*n_folds
        for project, uids in items:
            fold=min(range(n_folds), key=lambda i: (sizes[i], i))
            mapping.update({uid: fold for uid in uids})
            sizes[fold]+=len(uids)
    splits=[]
    for k in range(n_folds):
        test={uid for uid,f in mapping.items() if f==k}
        val={uid for uid,f in mapping.items() if f==(k+1)%n_folds}
        train=set(mapping)-test-val
        splits.append({'id':f'fold_{k+1}', 'protocol':'projectwise', 'train':sorted(train),
                       'val':sorted(val), 'test':sorted(test), 'fold':k+1})
    for s in splits:
        validate_split(bugs, s)
    return splits


def temporal(bugs):
    projects=defaultdict(list)
    for b in bugs:
        projects[(b['dataset'],b['project'])].append(b)
    train=[]; val=[]; test=[]
    for group in projects.values():
        group=sorted(group,key=lambda b:(date_value(b.get('fix_timestamp',b['timestamp'])),b['uid']))
        n=len(group)
        if n<5:
            train.extend(b['uid'] for b in group)
        else:
            a=max(1,int(n*.6)); c=max(a+1,int(n*.8))
            train.extend(b['uid'] for b in group[:a])
            val.extend(b['uid'] for b in group[a:c])
            test.extend(b['uid'] for b in group[c:])
    s={'id':'timewise', 'protocol':'timewise','train':sorted(train),'val':sorted(val),'test':sorted(test)}
    validate_split(bugs,s)
    return [s]


def cross_dataset(bugs, source, target, seed=11):
    if source==target: raise ValueError('Source and target benchmarks must differ')
    source_bugs=[b for b in bugs if b['dataset'].lower()==source.lower()]
    target_bugs=[b for b in bugs if b['dataset'].lower()==target.lower()]
    if not source_bugs or not target_bugs:
        raise ValueError('Source or target benchmark is missing')
    groups=defaultdict(list)
    for b in source_bugs: groups[b['project']].append(b['uid'])
    projects=sorted(groups,key=lambda x:stable_hash(f'{seed}:{x}'))
    if len(projects)<2: raise ValueError('At least 2 source projects needed for validation')
    val=set(); n=sum(len(g) for g in groups.values()); target_count=max(1,round(n*0.1))
    for p in projects:
        if len(val) >= target_count: break
        val.update(groups[p])
    train={b['uid'] for b in source_bugs}-val
    s={'id':f'{source}_to_{target}', 'protocol':'crossdataset',
       'source':source,'target':target,'train':sorted(train),'val':sorted(val),
       'test':sorted(b['uid'] for b in target_bugs)}
    validate_split(bugs,s)
    return [s]


def validate_split(bugs, s):
    d={b['uid']:b for b in bugs}
    sets=[set(s[k]) for k in ('train','val','test')]
    if any(len(x)!=len(s[k]) for x,k in zip(sets,('train','val','test'))):
        raise ValueError('Duplicate bug UID inside a split')
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Overlapping bug IDs across splits')
    if set.union(*sets)-d.keys():
        raise ValueError('Split refers to unknown bug IDs')
    if not sets[0] or not sets[1] or not sets[2]:
        raise ValueError('Train, validation and test must be non-empty')
    if s['protocol'] in ('projectwise','crossdataset'):
        groups=[{(d[x]['dataset'],d[x]['project']) for x in part} for part in sets]
        if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):
            raise ValueError('Project leakage detected')
    if s['protocol']=='timewise':
        # Dates monotonic within each project, ties at boundary are disallowed.
        for uid in sets[1]|sets[2]:
            b=d[uid]; key=(b['dataset'],b['project']); ts=date_value(b['timestamp'])
            earlier=[date_value(d[x].get('fix_timestamp',d[x]['timestamp'])) for x in sets[0]
                     if (d[x]['dataset'],d[x]['project'])==key]
            if earlier and max(earlier)>date_value(b.get('fix_timestamp',b['timestamp'])): raise ValueError('Time-wise ordering violation')
        # Time-wise validation must be before testing within project.
        for x in sets[2]:
            b=d[x]
            earlier=[date_value(d[z].get('fix_timestamp',d[z]['timestamp'])) for z in sets[1]
                     if (d[z]['dataset'],d[z]['project'])==(b['dataset'],b['project'])]
            if earlier and max(earlier)>date_value(b.get('fix_timestamp',b['timestamp'])):
                raise ValueError('Time-wise validation crosses test boundary')
    return True


def make_files(path, outdir, protocol, source=None, target=None, seed=11):
    from pathlib import Path
    bugs=list(read_jsonl(path)); outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    if protocol=='projectwise': splits=grouped_folds(bugs,seed=seed)
    elif protocol=='timewise': splits=temporal(bugs)
    elif protocol=='crossdataset': splits=cross_dataset(bugs,source,target,seed=seed)
    else: raise ValueError(protocol)
    for s in splits: write_json(outdir/f"{s['id']}.json", s)
    return [str(outdir/f"{s['id']}.json") for s in splits]
