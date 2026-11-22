#!/usr/bin/env python3
"""Opt-in clone of public benchmark repositories; never silently downloads data.

Benchmark-specific runtime setup (Perl, Java8, Maven, legacy Conda) is documented
in docs/BENCHMARKS.md and deliberately not attempted during pip install.
"""
import argparse
import json
import subprocess
from pathlib import Path

REPOS={
 'defects4j':('https://github.com/rjust/defects4j.git','v2.0.1'),
 'bugsinpy':('https://github.com/soarsmu/BugsInPy.git',None),
 'bears':('https://github.com/bears-bugs/bears-benchmark.git',None),
}


def run(cmd,cwd=None):
    subprocess.run(cmd,cwd=cwd,check=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dest',default='data/benchmarks')
    p.add_argument('--datasets',nargs='+',choices=REPOS,default=list(REPOS))
    p.add_argument('--defects4j-ref',default='v2.0.1')
    p.add_argument('--bugsinpy-ref',default=None)
    p.add_argument('--bears-ref',default=None)
    p.add_argument('--execute',action='store_true',help='Actually clone; without this flag shows the plan')
    args=p.parse_args()
    root=Path(args.dest).resolve();plan=[]
    for name in args.datasets:
        url,default=REPOS[name];ref=getattr(args,name+'_ref') or default
        plan.append({'name':name,'url':url,'requested_ref':ref,'dest':str(root/name)})
    if not args.execute:
        print(json.dumps({'dry_run':True,'plan':plan},indent=2));return
    root.mkdir(parents=True,exist_ok=True)
    for item in plan:
        path=Path(item['dest'])
        if not path.exists():run(['git','clone',item['url'],str(path)])
        if item['requested_ref']:
            run(['git','checkout','--detach',item['requested_ref']],cwd=path)
        item['resolved_sha']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
        item['note']='No benchmark initialization or test execution was performed.'
    (root/'acquisition_manifest.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan,indent=2))

if __name__=='__main__':main()
