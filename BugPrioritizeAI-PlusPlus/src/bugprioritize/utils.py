"""Stable utilities; no network calls, implicit benchmark downloads or magic results."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import random
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np


def stable_hash(value: str) -> int:
    return int(hashlib.sha256(str(value).encode('utf-8')).hexdigest()[:16], 16)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed: int):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def date_value(value) -> float:
    """Unix seconds; missing dates are explicitly 0, not current wallclock time."""
    if value is None or str(value).strip() == '':
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError as e:
        raise ValueError(f'Invalid timestamp {value!r}') from e


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f'{path}:{line_no}: {e}') from e


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding='utf-8')


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def run_command(args, cwd=None, timeout=300, check=True, env=None):
    """Run a command without shell interpolation. Capture stdout and stderr."""
    try:
        res = subprocess.run([str(a) for a in args], cwd=cwd, env=env, capture_output=True,
                             text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f'{args!r} exceeded {timeout}s') from e
    if check and res.returncode != 0:
        raise RuntimeError(f'Command failed ({res.returncode}): {args!r}\n{res.stdout[-1000:]}\n{res.stderr[-1000:]}')
    return res


def tokenize_identifiers(text):
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', str(text))
    return re.findall(r'[a-zA-Z_][a-zA-Z_0-9]*|\d+', s.lower().replace('_', ' '))


def clean_bug_text(text):
    s = re.sub(r'https?://\S+', ' URL ', str(text or ''))
    return re.sub(r'\s+', ' ', s).strip()[:12000]


def env_summary():
    import platform
    import torch
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
