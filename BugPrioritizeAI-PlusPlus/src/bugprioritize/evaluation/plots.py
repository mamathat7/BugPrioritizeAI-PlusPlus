
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np


def make_plots(results,out,metric='APFDc',dpi=600):
    import matplotlib.pyplot as plt
    from .stats import _aggregate,read_runs
    vals=_aggregate(read_runs(results))
    if not vals:raise ValueError('No validated results')
    names=[];means=[];sd=[]
    for method,data in sorted(vals.items()):
        arr=[v[metric] for v in data.values() if metric in v]
        if not arr:continue
        names.append(method);means.append(np.mean(arr));sd.append(np.std(arr,ddof=1) if len(arr)>1 else 0.)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar(range(len(names)),means,yerr=sd,capsize=3)
    ax.set_xticks(range(len(names)),names,rotation=35,ha='right')
    ax.set_ylabel(metric);ax.set_ylim(0,1.05)
    fig.tight_layout();fig.savefig(out/f'{metric}_comparison.png',dpi=dpi,bbox_inches='tight');plt.close(fig)
    return str(out/f'{metric}_comparison.png')
