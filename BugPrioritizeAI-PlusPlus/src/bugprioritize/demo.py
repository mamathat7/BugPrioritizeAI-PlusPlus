
from __future__ import annotations
import random
from datetime import datetime, timedelta, timezone


def generate(projects=10,bugs_per_project=4,tests_per_bug=12,seed=11):
    rng=random.Random(seed); output=[]
    if projects<5 or tests_per_bug<3:raise ValueError('Need >=5 projects and >=3 tests per bug')
    for pi in range(projects):
        project=f'project_{pi:02d}'
        dataset=['Defects4J','BugsInPy','Bears'][pi%3]
        for bi in range(bugs_per_project):
            token=['parser','cache','stream','timestamp','matrix','encoding','format','tokenizer'][(pi+bi)%8]
            ts=(datetime(2024,1,1,tzinfo=timezone.utc)+timedelta(days=20*bi+pi)).isoformat()
            tests=[]; positives={bi%tests_per_bug, (bi+5)%tests_per_bug}
            for ti in range(tests_per_bug):
                label=int(ti in positives)
                # Verdicts are explicitly generated for verifying the ingestion pipeline.
                v='fail' if label else 'pass'
                n=f'test_{token}_boundary_{ti}' if label or ti%4==0 else f'test_unrelated_{ti}'
                costs=round(0.2+rng.random()*2,4)
                tests.append({'test_id':f'{project}.{n}','name':n,'path':f'tests/test_{token}.py',
                              'class':f'{project}.Tests','order':ti,
                              'buggy_runs':[{'verdict':v,'duration':costs} for _ in range(3)],
                              'fixed_runs':[{'verdict':'pass','duration':costs} for _ in range(3)],
                              'observed_execution_time':costs,
                              'estimated_execution_time':costs*(.9+.2*rng.random()),
                              'present_in_buggy':True,'history':[]})
            output.append({'dataset':dataset,'project':project,'bug_id':str(bi+1),
                           'timestamp':ts,'bug_text':f'{token} does not validate boundary condition',
                           'bug_text_source':'synthetic','source':'synthetic_smoke_only',
                           'diff':{'added':f'if {token}.valid(): return True',
                                   'deleted':f'if {token}: return True',
                                   'context':f'def validate_{token}(value):'},
                           'tests':tests})
    return output
