"""BugPrioritizeAI++ reproducible command-line interface."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .utils import read_jsonl, write_json, write_jsonl, sha256_file
from .datasets.schema import build_file
from .datasets.splits import make_files


def add_train_args(p):
    p.add_argument('--data',required=True,help='Processed bug-level JSONL')
    p.add_argument('--split',required=True,help='One split JSON from make-splits')
    p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--encoder',choices=['hash','codebert'],default='hash')
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--variant',default='full',choices=['full','bugprioritize','concat','bug_direct','bug_all',
                 'diff_direct','diff_all','test_direct','test_all','history_direct','history_all',
                 'retrieval','gated_fusion','cost_scoring','cost_all'])
    p.add_argument('--checkpoint',default='microsoft/codebert-base',help='Hugging Face text backbone ID')
    p.add_argument('--device',default='auto')
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--patience',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--dropout',type=float,default=.2)
    p.add_argument('--topk',type=int,default=10)
    p.add_argument('--margin',type=float,default=1.)
    p.add_argument('--ranking-weight',type=float,default=1.)
    p.add_argument('--cost-weight',type=float,default=.5)
    p.add_argument('--l2-weight',type=float,default=1e-5)
    p.add_argument('--positive-weight',type=float,default=5.)
    p.add_argument('--max-bugs',type=int,default=0,help='DEBUG only; caps bugs in every partition')
    p.add_argument('--memory-cap',type=int,default=0,help='DEBUG only; cap memory entries')
    p.add_argument('--faiss',action='store_true',help='Use exact FAISS IndexFlatIP if available')
    p.add_argument('--vocab',type=int,default=8192,help='Hash embedding vocabulary size (offline mode)')


def main(argv=None):
    parser=argparse.ArgumentParser(description='BugPrioritizeAI++ research reproduction')
    sub=parser.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('make-demo',help='Create synthetic smoke fixture (never a paper result)')
    demo.add_argument('--out',default='data/demo_raw.jsonl');demo.add_argument('--projects',type=int,default=10)
    demo.add_argument('--bugs-per-project',type=int,default=4);demo.add_argument('--tests-per-bug',type=int,default=12)
    demo.add_argument('--seed',type=int,default=11)
    build=sub.add_parser('build',help='Verify labels, time-order histories, and create processed JSONL')
    build.add_argument('--input',required=True);build.add_argument('--out',required=True)
    build.add_argument('--scenario',choices=['rkd','pro'],default='rkd')
    build.add_argument('--filter-log');build.add_argument('--decay',type=float,default=.8)
    splits=sub.add_parser('make-splits',help='Project-wise, chronological or cross-dataset split files')
    splits.add_argument('--data',required=True);splits.add_argument('--out',required=True)
    splits.add_argument('--protocol',choices=['projectwise','timewise','crossdataset'],default='projectwise')
    splits.add_argument('--source');splits.add_argument('--target');splits.add_argument('--seed',type=int,default=11)
    train=sub.add_parser('train',help='Train and test one model on one split')
    add_train_args(train)
    rank=sub.add_parser('rank',help='Rank external candidate tests without reading their labels')
    rank.add_argument('--checkpoint',required=True);rank.add_argument('--input',required=True)
    rank.add_argument('--out',required=True);rank.add_argument('--uid')
    rank.add_argument('--train-data',help='Processed full dataset to rebuild training-only memory')
    rank.add_argument('--split',help='Training split used to train checkpoint')
    rank.add_argument('--device',default='cpu');rank.add_argument('--no-cost-discount',action='store_true')
    evalp=sub.add_parser('evaluate',help='Evaluate retained per-test ranking CSV files')
    evalp.add_argument('--rankings',required=True);evalp.add_argument('--out',required=True)
    base=sub.add_parser('baseline',help='Fit/evaluate baseline on a fixed split')
    base.add_argument('--data',required=True);base.add_argument('--split',required=True)
    base.add_argument('--out',required=True);base.add_argument('--method',required=True,
        choices=['random','original','history','cost','bm25','lambdamart','gru','retecs','mm_lambdamart'])
    base.add_argument('--seed',type=int,default=11);base.add_argument('--epochs',type=int,default=10)
    base.add_argument('--max-bugs',type=int,default=0);base.add_argument('--trials',type=int,default=0)
    base.add_argument('--encoder-checkpoint',help='Trained full-model .pt for MM-LambdaMART')
    abl=sub.add_parser('ablate',help='Run documented ablations on one fixed split and seed')
    add_train_args(abl);abl.add_argument('--variants',nargs='*',default=[])
    collect=sub.add_parser('collect',help='Execute a real official-harness benchmark bug')
    collect.add_argument('--dataset',required=True,choices=['defects4j','bugsinpy','bears'])
    collect.add_argument('--project');collect.add_argument('--bug',required=True)
    collect.add_argument('--workspace',required=True);collect.add_argument('--out',required=True)
    collect.add_argument('--benchmark-repo',help='Bears benchmark git repository path')
    collect.add_argument('--scenario',choices=['rkd','pro'],default='rkd')
    collect.add_argument('--repetitions',type=int,default=3);collect.add_argument('--max-tests',type=int,default=0)
    collect.add_argument('--timeout',type=int,default=300);collect.add_argument('--issue-text-file')
    stat=sub.add_parser('stats',help='Paired bug-level statistics from model outputs')
    stat.add_argument('--input',required=True,help='Directory of evaluation summary runs')
    stat.add_argument('--reference',default='full');stat.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    if args.command=='make-demo':
        from .demo import generate
        rows=generate(args.projects,args.bugs_per_project,args.tests_per_bug,args.seed)
        write_jsonl(args.out,rows)
        print(json.dumps({'path':args.out,'synthetic_bugs':len(rows)},indent=2));return
    if args.command=='build':
        summary=build_file(args.input,args.out,args.scenario,args.filter_log,args.decay)
        print(json.dumps(summary,indent=2));return
    if args.command=='make-splits':
        files=make_files(args.data,args.out,args.protocol,args.source,args.target,args.seed)
        print(json.dumps(files,indent=2));return
    if args.command=='train':
        from .train import train_one
        fields={k:v for k,v in vars(args).items() if k not in ('command','data','split','out','faiss')}
        result=train_one(args.data,args.split,args.out,use_faiss=args.faiss,**fields)
        print(json.dumps(result,indent=2));return
    if args.command=='rank':
        from .train import load_checkpoint,predict_bug,write_ranking
        from .models.memory import RetrievalMemory
        model,transform,state=load_checkpoint(args.checkpoint,args.device)
        selected=[b for b in read_jsonl(args.input) if args.uid is None or b.get('uid')==args.uid]
        if not selected: raise ValueError('No matching bugs in input')
        memory=None;protocol='projectwise'
        if bool(args.train_data)!=bool(args.split):
            raise ValueError('--train-data and --split are required together to rebuild history memory')
        if args.train_data:
            if sha256_file(args.train_data)!=state.get('data_sha256') or sha256_file(args.split)!=state.get('split_sha256'):
                raise ValueError('Input benchmark or split checksum does not match trained checkpoint')
            from .train import load_split_data
            split,part=load_split_data(args.train_data,args.split,max_bugs=state['options'].get('debug_max_bugs',0))
            train_rows=[transform.transform(b) for b in part['train']]
            memory=RetrievalMemory.build(model,train_rows,k=state['options']['topk'],
                    max_items=state['options'].get('memory_cap',0))
            protocol=split['protocol']
        output=Path(args.out)
        for b in selected:
            b=dict(b)
            b.setdefault('uid', f"{b['dataset']}::{b['project']}::{b['bug_id']}")
            b.setdefault('scenario','rkd')
            for t in b['tests']:
                t.setdefault('label',-1);t.setdefault('history_features',[0.]*6)
                t.setdefault('history_mask',0)
                t.setdefault('cost_missing',t.get('estimated_execution_time') is None)
                t.setdefault('cost_estimate',t.get('estimated_execution_time'))
                t.setdefault('observed_execution_time',None)
            rows=transform.transform(b)
            for r in rows: r['label']=-1  # Defensively suppress any supplied labels at inference.
            ranked=predict_bug(model,rows,memory,protocol,no_cost_discount=args.no_cost_discount)
            dest=output if len(selected)==1 else output/f"{b['uid'].replace('::','_')}.csv"
            write_ranking(dest,ranked)
            print(f'Ranked {len(ranked)} candidates: {dest}')
        return
    if args.command=='evaluate':
        from .evaluation.summary import evaluate_saved_rankings
        summary=evaluate_saved_rankings(args.rankings,args.out)
        print(json.dumps(summary,indent=2));return
    if args.command=='baseline':
        from .baselines.run import run_baseline
        result=run_baseline(args.data,args.split,args.out,args.method,args.seed,args.epochs,args.max_bugs,args.trials,args.encoder_checkpoint)
        print(json.dumps(result,indent=2));return
    if args.command=='ablate':
        from .train import train_one
        from .models.network import ABLATIONS
        variants=args.variants or list(ABLATIONS)
        reports={}
        kw={k:v for k,v in vars(args).items() if k not in ('command','data','split','out','faiss','variants','variant')}
        if 'cost_scoring' in variants and 'full' not in variants:
            reference=Path(args.out)/'full'/'model.pt'
            if not reference.exists():raise ValueError('For inference-only cost ablation, include full variant or supply existing full/model.pt')
        if 'cost_scoring' in variants:variants=[x for x in variants if x!='cost_scoring']+['cost_scoring']
        for name in variants:
            if name not in ABLATIONS: raise ValueError(name)
            if name!='cost_scoring':
                reports[name]=train_one(args.data,args.split,str(Path(args.out)/name),variant=name,
                                        use_faiss=args.faiss,**kw)
                continue
            from .train import load_checkpoint,load_split_data,evaluate_model
            from .models.memory import RetrievalMemory
            from .utils import write_csv
            device='cuda' if args.device=='auto' and __import__('torch').cuda.is_available() else 'cpu' if args.device=='auto' else args.device
            model,trans,state=load_checkpoint(Path(args.out)/'full'/'model.pt',device)
            split,parts=load_split_data(args.data,args.split,args.max_bugs)
            group=[trans.transform(b) for b in parts['train']]
            memory=RetrievalMemory.build(model,group,k=args.topk,max_items=args.memory_cap,use_faiss=args.faiss)
            root=Path(args.out)/'cost_scoring'
            per,test=evaluate_model(model,parts['test'],trans,memory,split['protocol'],root/'rankings',no_cost_discount=True)
            write_csv(root/'per_bug_metrics.csv',per)
            report={'split':split['id'],'seed':args.seed,'variant':'cost_scoring','test':test,
                    'retrained':False,'source_checkpoint':str(Path(args.out)/'full'/'model.pt')}
            write_json(root/'summary.json',report)
            reports[name]=report
        write_json(Path(args.out)/'ablations.json',reports)
        print(json.dumps({'variants':list(reports)},indent=2));return
    if args.command=='collect':
        from .datasets.collectors import collect_defects4j,collect_bugsinpy,collect_bears
        text=Path(args.issue_text_file).read_text() if args.issue_text_file else None
        kw={'workspace':args.workspace,'scenario':args.scenario,'repetitions':args.repetitions,
            'max_tests':args.max_tests,'timeout':args.timeout,'issue_text':text}
        if args.dataset=='bears':
            if not args.benchmark_repo:raise ValueError('--benchmark-repo required for Bears')
            record=collect_bears(args.benchmark_repo,args.bug,**kw)
        elif args.dataset=='defects4j':
            if not args.project:raise ValueError('--project required')
            record=collect_defects4j(args.project,args.bug,**kw)
        else:
            if not args.project:raise ValueError('--project required')
            record=collect_bugsinpy(args.project,args.bug,**kw)
        write_jsonl(args.out,[record])
        print(f'Collected {len(record["tests"])} tests: {args.out}')
        return
    if args.command=='stats':
        from .evaluation.stats import run_stats
        print(json.dumps(run_stats(args.input,args.reference,args.out),indent=2));return

if __name__=='__main__': main()
