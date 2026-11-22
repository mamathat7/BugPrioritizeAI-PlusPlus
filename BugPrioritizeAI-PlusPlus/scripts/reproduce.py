#!/usr/bin/env python3
"""Opt-in multi-fold multi-seed experiment orchestrator (large GPU workloads)."""
import argparse
import json
from pathlib import Path
from bugprioritize.train import train_one
from bugprioritize.baselines.run import run_baseline
from bugprioritize.utils import write_json

NEURAL={'full':'full','bugprioritize':'bugprioritize','mm_concat':'concat'}
HEURISTICS={'random','original','history','cost','bm25','lambdamart','gru','retecs','mm_lambdamart'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--splits',required=True)
    p.add_argument('--out',default='results/full')
    p.add_argument('--seeds',default='11,23,37,51,71')
    p.add_argument('--methods',default='full,bugprioritize,mm_concat,random,original,history,cost,bm25,lambdamart,gru,retecs,mm_lambdamart')
    p.add_argument('--encoder',choices=['hash','codebert'],default='codebert')
    p.add_argument('--hidden',type=int,default=256)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--trials',type=int,default=0,help='Optuna TPE validation trials for LambdaMART')
    p.add_argument('--device',default='auto')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--max-folds',type=int,default=0,help='Debug only, restrict number of folds')
    p.add_argument('--max-bugs',type=int,default=0,help='Debug only, subset every partition')
    args=p.parse_args()
    splits=sorted(Path(args.splits).glob('*.json'))
    if not splits:raise ValueError('No split files in directory')
    if args.max_folds:splits=splits[:args.max_folds]
    seeds=[int(s) for s in args.seeds.split(',')]
    methods=[s.strip() for s in args.methods.split(',')]
    if any(m not in NEURAL and m not in HEURISTICS for m in methods):raise ValueError('Unsupported method')
    all_runs=[]
    for split in splits:
        for seed in seeds:
            for method in methods:
                run_dir=Path(args.out)/method/str(seed)/split.stem
                job={'seed':seed,'split':str(split),'method':method,'output':str(run_dir)}
                all_runs.append(job)
                if args.dry_run:continue
                if (run_dir/'summary.json').exists():
                    print('Skipping completed',run_dir,flush=True);continue
                if method in NEURAL:
                    train_one(args.data,split,run_dir,seed=seed,encoder=args.encoder,hidden=args.hidden,
                              variant=NEURAL[method],device=args.device,epochs=args.epochs,
                              batch_size=args.batch_size,max_bugs=args.max_bugs)
                else:
                    kwargs={}
                    if method=='mm_lambdamart':
                        kwargs['encoder_checkpoint']=str(Path(args.out)/'full'/str(seed)/split.stem/'model.pt')
                    run_baseline(args.data,split,run_dir,method=method,seed=seed,epochs=args.epochs,
                                 max_bugs=args.max_bugs,trials=args.trials if method in ('lambdamart','mm_lambdamart') else 0,
                                 **kwargs)
                print('Completed',run_dir,flush=True)
    report={'data':args.data,'seeds':seeds,'methods':methods,'jobs':all_runs,
            'dry_run':args.dry_run,'debug_subset':bool(args.max_bugs or args.max_folds)}
    if not args.dry_run:write_json(Path(args.out)/'manifest.json',report)
    print(json.dumps({'job_count':len(all_runs),'dry_run':args.dry_run},indent=2))

if __name__=='__main__':main()
