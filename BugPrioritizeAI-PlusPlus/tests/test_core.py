import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from bugprioritize.datasets.schema import classify_test,preprocess_records,history_features
from bugprioritize.datasets.splits import grouped_folds,temporal,cross_dataset,validate_split
from bugprioritize.datasets.features import NumericTransform
from bugprioritize.demo import generate
from bugprioritize.models.network import BugPrioritizeModel
from bugprioritize.models.memory import RetrievalMemory
from bugprioritize.models.loss import hybrid_loss
from bugprioritize.evaluation.metrics import metrics,mean_metrics,mrr_topk_bound,order_rows
from bugprioritize.evaluation.stats import holm,paired_rank_biserial
from bugprioritize.datasets.collectors import parse_diff,discover_java_methods,discover_pytest
from bugprioritize.utils import stable_hash, date_value,read_jsonl,write_jsonl


def processed(projects=15,bugs_per_project=6,tests_per_bug=8):
    bugs,excluded=preprocess_records(generate(projects,bugs_per_project,tests_per_bug))
    assert not excluded
    return bugs


def test_label_verification():
    t={'buggy_runs':['fail']*3,'fixed_runs':['pass']*3}
    assert classify_test(t)[0]==1
    assert classify_test({'buggy_runs':['pass']*3,'fixed_runs':['pass']*3})[0]==0
    assert classify_test({'buggy_runs':['fail','pass','fail'],'fixed_runs':['pass']*3})[1]=='flaky'
    assert classify_test({'buggy_runs':['fail']*3,'fixed_runs':['fail']*3})[1]=='unrelated_failure'
    assert classify_test({'buggy_runs':['fail']*2,'fixed_runs':['pass']*3})[1]=='insufficient_repetitions'
    assert classify_test({'label':1})[0] is None
    assert classify_test({'label':1,'label_provenance':'recorded'})[0]==1


def test_real_history_is_strictly_prior():
    records=[{'timestamp':'2024-01-01T00:00:00Z','failed':True},
             {'timestamp':'2024-01-02T00:00:00Z','failed':False},
             {'timestamp':'2024-12-01T00:00:00Z','failed':True}]
    h,m=history_features(records,date_value('2024-01-03T00:00:00Z'))
    assert m==1 and h[0]==.5
    h2,m2=history_features(records,date_value('2023-01-01T00:00:00Z'))
    assert m2==0 and h2==[0.]*6


def test_pro_masks_repair_and_fix_only_test():
    b=generate(5,1,5)
    b[0]['dataset']='Defects4J';b[0]['bug_text_source']='fix_commit'
    b[0]['tests'][0]['added_by_fix']=True
    a,ex=preprocess_records(b,scenario='pro')
    assert a[0]['diff']=={'added':'','deleted':'','context':''}
    assert a[0]['bug_text']==''
    assert len(a[0]['tests'])==4
    assert any(e['reason']=='fix_only_test_in_pro' for e in ex)


def test_project_and_time_splits():
    bugs=processed()
    pw=grouped_folds(bugs)
    assert len(pw)==5
    assert {u for s in pw for u in s['test']}=={b['uid'] for b in bugs}
    for s in pw:assert validate_split(bugs,s)
    tw=temporal(bugs)
    assert validate_split(bugs,tw[0])
    xd=cross_dataset(bugs,'Defects4J','BugsInPy')
    assert validate_split(bugs,xd[0])


def test_apfd_apfdc_single_fault():
    rs=[{'score':2,'label':0,'cost_raw':1,'observed_cost':2,'test_id':'a','order':0},
        {'score':1,'label':1,'cost_raw':1,'observed_cost':2,'test_id':'b','order':1}]
    m=metrics(rs)
    assert m['APFD']==pytest.approx(.25)
    assert m['APFDc']==pytest.approx(.25)
    assert m['MRR']==pytest.approx(.5)
    assert m['Recall@5']==1
    assert mrr_topk_bound([m])>=m['MRR']


def test_masked_modalities_receive_zero_attention():
    bugs=processed(6,2,6)
    transform=NumericTransform.fit(bugs[:6])
    b=bugs[0].copy();b['bug_text']='';b['diff']={}
    group=transform.transform(b)
    for r in group:r['history_mask']=0.;r['cost_missing']=1.
    mod=BugPrioritizeModel('hash',hidden=16,vocab=128)
    pred=mod(group)
    assert pred['attention'].shape==(6,6)
    assert (pred['attention'][:,[0,1,3,4,5]]==0).all()
    assert torch.allclose(pred['attention'][:,2],torch.ones(6),atol=1e-6)
    loss,_=hybrid_loss(mod,pred,group,l2_weight=0)
    loss.backward()
    assert torch.isfinite(loss)
    assert mod.head[-1].weight.grad is not None


def test_memory_cannot_retrieve_target_or_future():
    bugs=processed(6,2,6)
    trans=NumericTransform.fit(bugs)
    a,b=trans.transform(bugs[0]),trans.transform(bugs[1])
    model=BugPrioritizeModel('hash',hidden=16,vocab=128)
    mem=RetrievalMemory.build(model,[a,b],k=2)
    _,_,present=mem.retrieve(torch.randn(len(a),16),a,protocol='timewise')
    assert present.sum()==0
    assert not mem.audit(a,'timewise')
    _,_,present2=mem.retrieve(torch.randn(len(b),16),b,protocol='timewise')
    assert present2.sum()>0
    assert not mem.audit(b,'timewise')


def test_train_fold_transform_is_stable():
    bugs=processed(6,2,6)
    t=NumericTransform.fit(bugs[:5])
    original=t.serialize()
    b=bugs[-1]
    r=t.transform(b)
    assert len(r)==len(b['tests'])
    assert t.serialize()==original
    assert all(0<=z['cost_norm']<=1 for z in r)


def test_diff_and_method_discovery(tmp_path):
    d=parse_diff('--- a/A.java\n+++ b/A.java\n@@ -1 +1 @@\n-x\n+y\n z\n')
    assert d=={'added':'y','deleted':'x','context':'z'}
    root=tmp_path/'src';root.mkdir()
    (root/'ParserTest.java').write_text('package a.b; class ParserTest { @Test public void testParser() { } }')
    methods=discover_java_methods(root)
    assert len(methods)==1
    assert methods[0]['test_id']=='a.b.ParserTest::testParser'
    (root/'test_parser.py').write_text('def test_parser():\n  assert True\n')
    assert len(discover_pytest(root))==1


def test_holm_monotonic():
    p=holm([.001,.02,.03])
    assert p[0]<=p[1]<=p[2]
    assert paired_rank_biserial([1,2,3])==1
