
import argparse
import json
import statistics
import time
from pathlib import Path
import torch
from bugprioritize.train import load_checkpoint,predict_bug,load_split_data
from bugprioritize.models.memory import RetrievalMemory
from bugprioritize.utils import read_jsonl,write_csv,write_json


def sync():
    if torch.cuda.is_available():torch.cuda.synchronize()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',required=True,help='Processed candidate bug JSONL')
    p.add_argument('--out',default='results/latency')
    p.add_argument('--sizes',default='10,100,1000')
    p.add_argument('--warmups',type=int,default=10)
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--device',default='auto')
    p.add_argument('--train-data',help='Same processed dataset used to train this checkpoint')
    p.add_argument('--split',help='Same split used to train this checkpoint')
    a=p.parse_args()
    device='cuda' if a.device=='auto' and torch.cuda.is_available() else 'cpu' if a.device=='auto' else a.device
    model,transform,state=load_checkpoint(a.checkpoint,device)
    bug=next(read_jsonl(a.data))
    allrows=transform.transform(bug);record=[]
    mem=None;protocol='projectwise'
    if bool(a.train_data)!=bool(a.split):raise ValueError('Both --train-data and --split are required together')
    if a.train_data:
        split,parts=load_split_data(a.train_data,a.split)
        mem=RetrievalMemory.build(model,[transform.transform(x) for x in parts['train']],k=state['options'].get('topk',10))
        protocol=split['protocol']
    for n in [int(x) for x in a.sizes.split(',')]:
        batch=[dict(allrows[i%len(allrows)],test_id=f'test_{i}') for i in range(n)]
        for _ in range(a.warmups):predict_bug(model,batch,mem,protocol)
        times=[]
        for _ in range(a.repeats):
            sync();start=time.perf_counter();predict_bug(model,batch,mem,protocol);sync()
            times.append(time.perf_counter()-start)
        s=sorted(times)
        record.append({'tests':n,'repeats':a.repeats,'mean_s':statistics.mean(s),
                       'median_s':statistics.median(s),'q1_s':s[len(s)//4],
                       'q3_s':s[(3*len(s))//4], 'device':device,
                       'includes_model_load':False,'includes_dataset_acquisition':False,
                       'retrieval_memory_loaded':mem is not None})
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'ranking_latency.csv',record);write_json(out/'summary.json',record)
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
