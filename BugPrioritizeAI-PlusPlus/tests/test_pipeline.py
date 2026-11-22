import csv
import json
from pathlib import Path
import pytest
from bugprioritize.demo import generate
from bugprioritize.utils import write_jsonl,read_jsonl
from bugprioritize.datasets.schema import build_file
from bugprioritize.datasets.splits import make_files
from bugprioritize.train import train_one,load_checkpoint,predict_bug
from bugprioritize.baselines.run import run_baseline,bm25_scores,handcrafted
from bugprioritize.evaluation.summary import evaluate_saved_rankings
from bugprioritize.evaluation.stats import run_stats
from bugprioritize.datasets.features import NumericTransform


def dataset(tmp_path):
    raw=tmp_path/'raw.jsonl';processed=tmp_path/'processed.jsonl'
    write_jsonl(raw,generate(projects=15,bugs_per_project=2,tests_per_bug=6,seed=12))
    summary=build_file(raw,processed)
    assert summary['positives']==60
    split=make_files(processed,tmp_path/'splits','projectwise')[0]
    return processed,split


def test_offline_train_rank_and_evaluate(tmp_path):
    p,s=dataset(tmp_path)
    result=train_one(p,s,tmp_path/'full',encoder='hash',hidden=12,vocab=128,
                     epochs=1,batch_size=6,topk=2,l2_weight=0,seed=11)
    assert (tmp_path/'full/model.pt').exists()
    assert not result['leakage_audit_violations']
    recalc=evaluate_saved_rankings(tmp_path/'full/rankings',tmp_path/'verify')
    assert recalc['APFD']==pytest.approx(result['test']['APFD'])
    model,trans,state=load_checkpoint(tmp_path/'full/model.pt')
    assert state['model_config']['hidden']==12
    # Target labels are overwritten for blind ranking in CLI; the network never reads them.
    b=next(read_jsonl(p))
    rows=trans.transform(b)
    rank=predict_bug(model,rows)
    for r in rows:r['label']=1-r['label']
    rank2=predict_bug(model,rows)
    assert [r['score'] for r in rank]==pytest.approx([r['score'] for r in rank2])


def test_baselines_and_statistics(tmp_path):
    p,s=dataset(tmp_path)
    full=train_one(p,s,tmp_path/'full',encoder='hash',hidden=12,vocab=128,
                   epochs=1,batch_size=6,topk=2,l2_weight=0)
    base=run_baseline(p,s,tmp_path/'bm25',method='bm25')
    assert base['test']['bugs_evaluated']>0
    stat=run_stats(tmp_path,'full',tmp_path/'stats',resamples=200)
    assert stat['tests']>=1


def test_feature_count_and_bm25():
    p=generate(5,1,5)
    from bugprioritize.datasets.schema import preprocess_records
    bugs,_=preprocess_records(p)
    row=NumericTransform.fit(bugs).transform(bugs[0])
    assert len(handcrafted(row[0]))==42
    assert len(bm25_scores(row))==5
