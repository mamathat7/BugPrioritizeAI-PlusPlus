
from __future__ import annotations
import difflib
import os
import re
import shutil
import statistics
import time
from pathlib import Path
from ..utils import run_command, write_json, write_jsonl

EXCLUDE_DIRS={'target','build','.git','node_modules','venv','env','.tox','dist','__pycache__'}


def parse_diff(patch):
    added=[];removed=[];context=[]
    for line in str(patch).splitlines():
        if line.startswith(('+++','---','@@','diff ','index ','Binary files')): continue
        if line.startswith('+'): added.append(line[1:])
        elif line.startswith('-'): removed.append(line[1:])
        elif line.startswith(' '): context.append(line[1:])
    return {'added':'\n'.join(added)[:40000],'deleted':'\n'.join(removed)[:40000],
            'context':'\n'.join(context)[:40000]}


def source_diff(buggy, fixed, extensions=('.java','.py')):
    """Offline text-diff source trees; excludes build outputs and copied metadata."""
    buggy=Path(buggy);fixed=Path(fixed)
    def collect(root):
        files={}
        for p in root.rglob('*'):
            rel=p.relative_to(root).parts
            if p.suffix.lower() not in extensions or any(part in EXCLUDE_DIRS for part in rel): continue
            if any(rel[i]=='src' and i+1<len(rel) and rel[i+1]=='test' for i in range(len(rel))): continue
            if rel and rel[0] in ('tests','test'):continue
            if p.is_file(): files[str(p.relative_to(root))]=p
        return files
    old,new=collect(buggy),collect(fixed)
    pieces=[]
    for name in sorted(old.keys()|new.keys()):
        a=old[name].read_text(encoding='utf-8',errors='replace').splitlines(keepends=True) if name in old else []
        b=new[name].read_text(encoding='utf-8',errors='replace').splitlines(keepends=True) if name in new else []
        if a!=b:
            pieces.extend(difflib.unified_diff(a,b,fromfile='a/'+name,tofile='b/'+name))
        if len(pieces)>100000: break
    return parse_diff(''.join(pieces)), '\n'.join(pieces)[:200000]


def discover_java_methods(root, allowed_classes=None, max_tests=0):
    """Static JUnit3/JUnit4-style method enumeration; parameterized/dynamic tests need XML import."""
    root=Path(root);out=[];seen=set()
    pat=re.compile(r'(?:public|protected|private)?\s*(?:static\s+)?(?:final\s+)?(?:void|[A-Z][\w<>.?]*)\s+(test\w+|\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,. ]+)?\s*\{')
    for f in sorted(root.rglob('*.java')):
        if any(part in EXCLUDE_DIRS for part in f.relative_to(root).parts): continue
        text=f.read_text(errors='replace');package=re.search(r'\bpackage\s+([\w.]+)\s*;',text)
        clazz=(package.group(1)+'.' if package else '')+f.stem
        if allowed_classes and clazz not in allowed_classes: continue
        for m in pat.finditer(text):
            name=m.group(1)
            # annotation before method, or JUnit3 test* method
            prefix=text[max(0,m.start()-220):m.start()]
            if not name.startswith('test') and not re.search(r'@(?:org\.junit\.)?Test\b|@ParameterizedTest\b',prefix): continue
            if name in ('test','testing'): continue
            test_id=f'{clazz}::{name}'
            if test_id in seen: continue
            seen.add(test_id);out.append({'test_id':test_id,'name':name,'class':clazz,
                                          'path':str(f.relative_to(root)),'order':len(out)})
            if max_tests and len(out)>=max_tests:return out
    return out


def discover_pytest(root,max_tests=0):
    """Conservative source-level discovery; runners can also supply collected pytest node IDs."""
    root=Path(root);found=[];seen=set()
    for f in sorted(root.rglob('test_*.py')):
        if any(part in EXCLUDE_DIRS for part in f.relative_to(root).parts): continue
        text=f.read_text(errors='replace')
        for m in re.finditer(r'^\s*(?:async\s+)?def\s+(test_\w+)\s*\(',text,flags=re.MULTILINE):
            node=f'{f.relative_to(root).as_posix()}::{m.group(1)}'
            if node in seen: continue
            seen.add(node);found.append({'test_id':node,'name':m.group(1),'class':'',
                                         'path':str(f.relative_to(root)), 'order':len(found)})
            if max_tests and len(found)>=max_tests:return found
    return found


def _status(result, failure_file=None):
    """Classify nonzero exits as failures only with evidence of a test failure."""
    text=result.stdout+'\n'+result.stderr
    ff=Path(failure_file) if failure_file else None
    if ff is not None and ff.exists():
        lines=ff.read_text(errors='replace').splitlines()
        if any(l.strip().startswith('--- ') for l in lines): return 'fail'
        if lines and lines[0].strip().isdigit() and int(lines[0])>0: return 'fail'
    if re.search(r'(?im)^\s*FAILED\s+(\S+|\d+)',text) or re.search(r'(?i)Tests run: \d+, Failures: [1-9]',text):
        return 'fail'
    if result.returncode==0:return 'pass'
    return 'infra'


def _run_test(command,cwd,timeout,verdict_file=None):
    start=time.perf_counter()
    try:
        r=run_command(command,cwd=cwd,timeout=timeout,check=False)
        duration=time.perf_counter()-start
        return {'verdict':_status(r,verdict_file),'duration':duration,
                'returncode':r.returncode,'log_tail':(r.stdout+'\n'+r.stderr)[-500:]}
    except TimeoutError:
        return {'verdict':'timeout','duration':time.perf_counter()-start,'returncode':124,'log_tail':'timeout'}


def _git_timestamp(path):
    r=run_command(['git','log','-1','--format=%cI'],cwd=path,check=False)
    if r.returncode!=0 or not r.stdout.strip():
        raise RuntimeError('Missing git history: cannot assign an auditable bug timestamp')
    return r.stdout.strip()


def _git_message(path):
    r=run_command(['git','log','-1','--format=%B'],cwd=path,check=False)
    return r.stdout.strip() if r.returncode==0 else ''


def _compile_d4j(where):run_command(['defects4j','compile'],cwd=where,timeout=1200)


def collect_defects4j(project,bug_id,workspace,scenario='rkd',repetitions=3,max_tests=0,timeout=300,
                      issue_text=None):
    """Executable Defects4J adapter; expensive real tests are opt-in."""
    if not shutil.which('defects4j'):raise RuntimeError('defects4j not in PATH; install and initialise the pinned official benchmark')
    root=Path(workspace).resolve();root.mkdir(parents=True,exist_ok=True)
    buggy=root/f'{project}_{bug_id}_b';fixed=root/f'{project}_{bug_id}_f'
    for version,directory in (('b',buggy),('f',fixed)):
        if not directory.exists():
            run_command(['defects4j','checkout','-p',project,'-v',str(bug_id)+version,'-w',str(directory)],timeout=1200)
        _compile_d4j(directory)
    def export(where,p):
        res=run_command(['defects4j','export','-p',p],cwd=where,check=False)
        return res.stdout.strip() if res.returncode==0 else ''
    candidate=fixed if scenario=='rkd' else buggy
    prop='tests.relevant' if scenario=='rkd' else 'tests.all'
    allowed={x.strip() for x in export(candidate,prop).splitlines() if x.strip()}
    test_dir=export(candidate,'dir.src.tests')
    if not test_dir: raise RuntimeError('Defects4J missing dir.src.tests property')
    discovered=discover_java_methods(candidate/test_dir,allowed_classes=allowed or None,max_tests=max_tests)
    buggy_test_dir=export(buggy,'dir.src.tests') or test_dir
    buggy_inventory={t['test_id'] for t in discover_java_methods(buggy/buggy_test_dir)}
    if not discovered: raise RuntimeError('Could not statically enumerate candidate JUnit methods; supply verified method-level JSONL instead')
    bug_src=export(buggy,'dir.src.classes');fix_src=export(fixed,'dir.src.classes')
    if not bug_src or not fix_src:raise RuntimeError('Missing production source tree path')
    diff,_=source_diff(buggy/bug_src,fixed/fix_src,extensions=('.java',))
    tests=[]
    for t in discovered:
        # Defects4J test ID convention: method::fully.qualified.Class
        class_name=t['class'];method=t['name']
        d4j_id=f'{method}::{class_name}'
        reps=[]
        for where in (buggy,fixed):
            outcomes=[]
            for _ in range(repetitions):
                file=where/'failing_tests'
                if file.exists():file.unlink()
                outcomes.append(_run_test(['defects4j','test','-t',d4j_id],where,timeout,file))
            reps.append(outcomes)
        t['buggy_runs'],t['fixed_runs']=reps
        # explicit comparison only; no positive test selection from tests.trigger.
        t['present_in_buggy']=t['test_id'] in buggy_inventory
        t['added_by_fix']=not t['present_in_buggy']
        if t['present_in_buggy']:
            oldfile=buggy/buggy_test_dir/t['path'];newfile=candidate/test_dir/t['path']
            t['modified_by_fix']=oldfile.exists() and newfile.exists() and oldfile.read_bytes()!=newfile.read_bytes()
        tests.append(t)
    return {'dataset':'Defects4J','project':project,'bug_id':str(bug_id),'timestamp':_git_timestamp(fixed),
            'bug_text':issue_text if issue_text is not None else (_git_message(fixed) if scenario=='rkd' else ''),
            'bug_text_source':'initial_issue' if issue_text is not None else ('fix_commit' if scenario=='rkd' else 'missing'),
            'diff':diff,'tests':tests,'source':'official_harness_execution',
            'provenance':{'buggy_checkout':str(buggy),'fixed_checkout':str(fixed),
                          'repetitions':repetitions,'candidate_property':prop,
                          'warning':'Static JUnit enumeration can miss parameterised/dynamic methods; verify coverage.'}}


def collect_bugsinpy(project,bug_id,workspace,scenario='rkd',repetitions=3,max_tests=0,timeout=300,
                     issue_text=None):
    for c in ('bugsinpy-checkout','bugsinpy-compile','bugsinpy-test'):
        if not shutil.which(c):raise RuntimeError(f'{c} unavailable; install official BugsInPy and project environments')
    root=Path(workspace).resolve();root.mkdir(parents=True,exist_ok=True)
    buggy=root/f'{project}_{bug_id}_b';fixed=root/f'{project}_{bug_id}_f'
    for v,path in (('0',buggy),('1',fixed)):
        if not path.exists():
            run_command(['bugsinpy-checkout','-p',project,'-i',str(bug_id),'-v',v,'-w',str(path)],timeout=1200)
        run_command(['bugsinpy-compile','-w',str(path)],timeout=1800)
    candidate=fixed if scenario=='rkd' else buggy
    tests=discover_pytest(candidate,max_tests=max_tests)
    if not tests: raise RuntimeError('No pytest functions statically found; use verified benchmark exports for unittest-based projects')
    diff,_=source_diff(buggy,fixed,extensions=('.py',))
    buggy_inventory={t['test_id'] for t in discover_pytest(buggy)}
    for t in tests:
        node=t['test_id']
        t['present_in_buggy']=t['test_id'] in buggy_inventory
        t['added_by_fix']=not t['present_in_buggy']
        oldfile=buggy/t['path'];newfile=fixed/t['path']
        t['modified_by_fix']=oldfile.exists() and newfile.exists() and oldfile.read_bytes()!=newfile.read_bytes()
        if scenario=='pro' and t['added_by_fix']: continue
        for kind,path in (('buggy',buggy),('fixed',fixed)):
            t[kind+'_runs']=[]
            for _ in range(repetitions):
                r=_run_test(['bugsinpy-test','-w',str(path),'-t',node],path,timeout)
                output=path/'bugsinpy_singletest.txt'
                if output.exists():
                    txt=output.read_text(errors='replace')
                    match=re.search(r'BugsInPy test:.*:\s*(\d+)\s*$',txt,re.MULTILINE)
                    if match:
                        r['verdict']='pass' if int(match.group(1))==0 else 'fail'
                t[kind+'_runs'].append(r)
    return {'dataset':'BugsInPy','project':project,'bug_id':str(bug_id),'timestamp':_git_timestamp(fixed),
            'bug_text':issue_text if issue_text is not None else (_git_message(fixed) if scenario=='rkd' else ''),
            'bug_text_source':'initial_issue' if issue_text is not None else ('fix_commit' if scenario=='rkd' else 'missing'),
            'diff':diff,'tests':tests,'source':'official_harness_execution',
            'provenance':{'repetitions':repetitions,'note':'Static pytest collection excludes unittest/dynamic cases; import verified output for those.'}}


def collect_bears(bears_repo,bug_id,workspace,scenario='rkd',repetitions=3,max_tests=0,timeout=300,issue_text=None):
    """Worktree-based Bears collector. Requires initialized bears-benchmark Git repository and Maven."""
    import json
    for c in ('git','mvn'):
        if not shutil.which(c):raise RuntimeError(f'{c} is required for Bears reproduction')
    repo=Path(bears_repo).resolve()
    metadata=repo/'scripts/data/bug_id_and_branch.json'
    if not metadata.exists():raise FileNotFoundError('Bears 118-bug reproducible metadata missing in repository')
    bugs=json.loads(metadata.read_text())
    matching=[b for b in bugs if b['bugId'].lower()==str(bug_id).lower()]
    if not matching:raise ValueError(f'{bug_id} is not in the currently reproducible Bears list')
    branch=matching[0]['bugBranch']
    # Branch commits: 1 buggy base, 2 test changes, 3 repair, 4 metadata.
    log=run_command(['git','rev-list','--reverse',branch],cwd=repo)
    commits=[v for v in log.stdout.splitlines() if v.strip()]
    if len(commits)<4:raise ValueError('Unexpected Bears branch history: expected >=4 commits')
    # Find labeled commits, robust to earlier project ancestry.
    latest=run_command(['git','log','-4','--format=%H%x09%s',branch],cwd=repo).stdout.splitlines()
    labels=list(reversed([x.split('\t',1) for x in latest]))
    if len(labels)<4:raise RuntimeError('Cannot identify Bears reproduction chain')
    buggy_sha=labels[1][0]  # tests-changed, bug remains in program
    fixed_sha=labels[2][0]  # human repair patch
    root=Path(workspace).resolve();root.mkdir(parents=True,exist_ok=True)
    buggy=root/f'{bug_id}_b';fixed=root/f'{bug_id}_f'
    for sha,where in ((buggy_sha,buggy),(fixed_sha,fixed)):
        if not where.exists():run_command(['git','worktree','add','--detach',str(where),sha],cwd=repo,timeout=600)
    candidate=fixed if scenario=='rkd' else buggy
    tests=discover_java_methods(candidate/'src/test/java',max_tests=max_tests)
    if not tests:
        # multi-module Maven project
        tests=[]
        for directory in sorted(candidate.glob('*/src/test/java')):
            tests.extend(discover_java_methods(directory,max_tests=max_tests-len(tests) if max_tests else 0))
    if not tests:raise RuntimeError('No JUnit methods discovered; provide an externally verified test-suite inventory')
    diff,_=source_diff(buggy,fixed,extensions=('.java',))
    buggy_inventory={t['test_id'] for t in discover_java_methods(buggy/'src/test/java')}
    for t in tests:
        t['present_in_buggy']=t['test_id'] in buggy_inventory
        t['added_by_fix']=not t['present_in_buggy']
        oldfile=buggy/'src/test/java'/t['path'];newfile=fixed/'src/test/java'/t['path']
        t['modified_by_fix']=oldfile.exists() and newfile.exists() and oldfile.read_bytes()!=newfile.read_bytes()
        testflag=f"-Dtest={t['class'].split('.')[-1]}#{t['name']}"
        for kind,where in (('buggy',buggy),('fixed',fixed)):
            t[kind+'_runs']=[]
            for _ in range(repetitions):
                # In real projects Maven reactor/submodule targeting may require --maven-module.
                cmd=['mvn','-q','-B','-Dcheckstyle.skip=true','-DskipITs=true',testflag,'test']
                t[kind+'_runs'].append(_run_test(cmd,where,timeout))
    pref={}
    if scenario=='pro':
        # Actual last passing -> failing build diff must come from CI metadata;
        # avoid substituting the repair or synthetic earlier commit as that diff.
        pref={}
    return {'dataset':'Bears','project':branch.rsplit('-',2)[0],'bug_id':str(bug_id),
            'timestamp':_git_timestamp(fixed),'bug_text':issue_text or '',
            'bug_text_source':'initial_issue' if issue_text else 'missing',
            'diff':diff,'prefailure_diff':pref,'tests':tests,'source':'official_harness_execution',
            'provenance':{'branch':branch,'buggy_sha':buggy_sha,'fixed_sha':fixed_sha,
                          'note':'For PRO, supply a verified pre-failure CI diff; absent diff is masked.'}}
