
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import math
import statistics
from typing import Optional
from ..utils import clean_bug_text, date_value, read_jsonl, write_csv, write_jsonl

PASS = {'pass', 'passed', 'success', 'ok'}
FAIL = {'fail', 'failed', 'failure', 'error'}
BAD = {'timeout', 'infra', 'infrastructure', 'skip', 'skipped', 'unknown'}


def verdict(v):
    if isinstance(v, dict):
        return str(v.get('verdict', v.get('status', 'unknown'))).lower()
    if isinstance(v, bool):
        return 'fail' if v else 'pass'
    if isinstance(v, (int, float)):
        return 'fail' if v == 1 else 'pass' if v == 0 else 'unknown'
    return str(v).strip().lower()


def classify_test(test, required_repetitions=3):
    """Returns (label, reason); no inference of labels from test names or harness triggers.

    Pre-labelled records are accepted only with explicit provenance, and marked as such.
    """
    if 'label' in test and test['label'] is not None:
        if test.get('label_provenance') not in ('recorded', 'verified', 'synthetic', 'imported'):
            return None, 'unverified_label_provenance'
        if int(test['label']) not in (0, 1):
            return None, 'invalid_label'
        return int(test['label']), 'retained_recorded_label'
    buggy = [verdict(x) for x in test.get('buggy_runs', [])]
    fixed = [verdict(x) for x in test.get('fixed_runs', [])]
    if len(buggy) < required_repetitions or len(fixed) < required_repetitions:
        return None, 'insufficient_repetitions'
    buggy, fixed = buggy[:required_repetitions], fixed[:required_repetitions]
    if any(x in BAD or x not in (PASS | FAIL) for x in buggy + fixed):
        return None, 'timeout_or_infrastructure'
    if len(set(buggy)) != 1 or len(set(fixed)) != 1:
        return None, 'flaky'
    if buggy[0] in FAIL and fixed[0] in PASS:
        return 1, 'verified_target_failure'
    if buggy[0] in FAIL and fixed[0] in FAIL:
        return None, 'unrelated_failure'
    if buggy[0] in PASS and fixed[0] in FAIL:
        return None, 'fixed_only_failure'
    return 0, 'verified_passing'


def history_features(records, current_timestamp, lam=0.8):
    """Uses strictly earlier chronological entries, never current/future verdicts."""
    past = sorted((r for r in records if 0 < date_value(r.get('timestamp')) < current_timestamp),
                  key=lambda x: date_value(x['timestamp']))[-50:]
    if not past:
        return [0.0] * 6, 0
    ys = [float(bool(r.get('failed', False))) for r in past]
    n = len(ys)
    fail_freq = sum(ys) / n
    weighted_recency = sum((lam ** (n-1-i)) * y for i, y in enumerate(ys)) / max(1e-9, sum(lam**j for j in range(n)))
    transitions = sum(ys[i] != ys[i-1] for i in range(1, n))
    stability = 1.0 - transitions / max(1, n-1)
    ci = [float(bool(r['ci_failed'])) for r in past if r.get('ci_failed') is not None]
    ci_freq = sum(ci) / len(ci) if ci else 0.0
    return [fail_freq, weighted_recency, stability, transitions / max(1, n-1), ci_freq, math.log1p(n)], 1


def preprocess_records(records, scenario='rkd', lam=0.8, required_repetitions=3):
    """Normalise records and construct histories in strictly chronological pass.

    PRO candidate test suites must be supplied from the buggy revision; no repair input
    can be reused, including fix commit text and fix-added test methods.
    """
    inputs = list(records)
    seen = set(); accepted = []; exclusions = []
    for bug in sorted(inputs, key=lambda x: (date_value(x.get('timestamp')), str(x.get('project')), str(x.get('bug_id')))):
        for key in ('dataset', 'project', 'bug_id', 'timestamp', 'tests'):
            if key not in bug:
                raise ValueError(f'Bug missing required {key}: {bug}')
        key = f"{bug['dataset']}::{bug['project']}::{bug['bug_id']}"
        if key in seen:
            raise ValueError(f'Duplicate bug UID: {key}')
        seen.add(key)
        b = dict(bug); b['uid'] = key; b['scenario'] = scenario
        # Fix-derived information becomes available after the fix, while an initial
        # bug report has an earlier decision timestamp when supplied.
        b['fix_timestamp'] = bug.get('fix_timestamp',bug['timestamp'])
        b['timestamp'] = bug.get('report_created_at',bug['timestamp'])
        b['time_provenance'] = ('report_created_at' if bug.get('report_created_at') else 'fix_timestamp_fallback')
        b['timestamp_numeric'] = date_value(b['timestamp'])
        diff = b.get('diff', {}) or {}
        if scenario == 'pro':
            if b.get('bug_text_source') == 'fix_commit':
                b['bug_text'] = ''
            # For Bears, an explicitly supplied pre-failure diff is admissible.
            if b['dataset'].lower() == 'bears':
                diff = dict(b.get('prefailure_diff', {}) or {})
            else:
                diff = {}
        b['diff'] = {k: str(diff.get(k, '') or '') for k in ('added', 'deleted', 'context')}
        b['bug_text'] = clean_bug_text(b.get('bug_text', ''))
        b['source'] = b.get('source', 'unreported')
        tests=[]
        for i, t0 in enumerate(b['tests']):
            t=dict(t0)
            if scenario == 'pro' and (t.get('added_by_fix') or t.get('modified_by_fix') or t.get('present_in_buggy') is False):
                exclusions.append({'uid':key,'test_id':t.get('test_id',''),'reason':'fix_only_test_in_pro'}); continue
            t['test_id'] = str(t.get('test_id', t.get('name', f'test_{i}')))
            t['order'] = int(t.get('order', i))
            t['name'] = str(t.get('name', t['test_id']))
            t['path'] = str(t.get('path', ''))
            t['class'] = str(t.get('class', ''))
            label, reason = classify_test(t, required_repetitions)
            if label is None:
                exclusions.append({'uid': key, 'test_id': t['test_id'], 'reason': reason}); continue
            t['label'] = label
            t['label_reason'] = reason
            # Stated external history already has timestamps; prohibit use of current/future.
            t['history'] = [r for r in t.get('history', []) if 0 < date_value(r.get('timestamp')) < b['timestamp_numeric']]
            tests.append(t)
        b['tests'] = tests
        if not tests:
            exclusions.append({'uid':key,'test_id':'*','reason':'no_valid_tests'}); continue
        accepted.append(b)
    # Build history from all strictly prior dataset bug instances. Never fold labels directly into features.
    by_project_test=defaultdict(list)
    for b in sorted(accepted, key=lambda x: x['timestamp_numeric']):
        ts=b['timestamp_numeric']; uid=b['uid']
        for t in b['tests']:
            k=(b['dataset'], b['project'], t['test_id'])
            history=by_project_test[k] + t['history']
            hist, h_mask=history_features(history, ts, lam=lam)
            t['history_sequence']=[int(bool(r.get('failed',False))) for r in sorted(history,key=lambda x:date_value(x.get('timestamp'))) if 0 < date_value(r.get('timestamp')) < ts][-50:]
            t['history_features']=hist; t['history_mask']=h_mask
            # Historical cost only, never this bug's observed duration, for prediction.
            older_costs = [r['execution_time'] for r in history if
                           r.get('execution_time') is not None and 0 < date_value(r.get('timestamp')) < ts]
            provided=t.get('estimated_execution_time')
            if provided is not None and float(provided) > 0:
                t['cost_estimate']=float(provided); t['cost_missing']=False
            elif older_costs:
                t['cost_estimate']=float(statistics.median(older_costs)); t['cost_missing']=False
            else:
                t['cost_estimate']=None; t['cost_missing']=True
            observed = t.get('observed_execution_time')
            if observed is None:
                times = [float(v['duration']) for v in t.get('buggy_runs', []) if isinstance(v, dict) and v.get('duration') is not None]
                observed = statistics.median(times) if times else None
            t['observed_execution_time']=float(observed) if observed is not None and float(observed)>0 else None
        # Update only after feature construction to avoid within-bug label exposure.
        for t in b['tests']:
            by_project_test[(b['dataset'],b['project'],t['test_id'])].append({
                'timestamp': b['fix_timestamp'], 'failed': bool(t['label']),
                'execution_time': t['observed_execution_time']})
    return accepted, exclusions


def build_file(input_path, output_path, scenario='rkd', filter_log=None, lam=0.8):
    b, exclusions = preprocess_records(read_jsonl(input_path), scenario=scenario, lam=lam)
    write_jsonl(output_path, b)
    if filter_log:
        write_csv(filter_log, exclusions, ['uid','test_id','reason'])
    return {'bugs': len(b), 'tests': sum(len(x['tests']) for x in b),
            'positives': sum(int(t['label']) for x in b for t in x['tests']),
            'excluded': len(exclusions)}
