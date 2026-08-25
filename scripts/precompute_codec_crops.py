#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import soundfile as sf
import torch
STEMS=("instrumental","vocals","mix")

def read_crop(path,start,frames):
    data,sr=sf.read(str(path),start=start,frames=frames,dtype='float32',always_2d=True)
    x=torch.from_numpy(data).t().mean(0,keepdim=True)
    if x.shape[-1]<frames:x=torch.nn.functional.pad(x,(0,frames-x.shape[-1]))
    return x,sr

def best_random_crop(path,frames,rng,tries=10):
    f=sf.SoundFile(str(path)); total=len(f); f.close(); best=None
    for _ in range(tries):
        start=0 if total<=frames else rng.randint(0,total-frames)
        x,sr=read_crop(path,start,frames); rms=float(x.pow(2).mean().sqrt())
        if best is None or rms>best[0]:best=(rms,start,x,sr)
    return best[1],best[2],best[3],best[0]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--split',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seconds',type=float,default=1.0);ap.add_argument('--train-crops-per-stem',type=int,default=3);ap.add_argument('--val-crops',type=int,default=4);ap.add_argument('--seed',type=int,default=1337);args=ap.parse_args()
    root=Path(args.dataset);split=json.load(open(args.split));meta=json.load(open(root/'dataset.json'));sr=int(meta['sample_rate']);frames=int(sr*args.seconds);rng=random.Random(args.seed);by={s['id']:s for s in meta['songs']}
    xs=[];weights=[];info=[]
    for sid in split['train']:
        for stem in STEMS:
            path=root/'songs'/sid/f'{stem}.wav'
            for _ in range(args.train_crops_per_stem):
                start,x,s,rms=best_random_crop(path,frames,rng,tries=8);assert s==sr
                xs.append(x);weights.append(float(by[sid].get('training_weight',1.0)));info.append({'song':sid,'stem':stem,'start':start,'rms':rms})
    vx=[];vinfo=[];vrng=random.Random(args.seed+99)
    for sid in split['validation']:
        path=root/'songs'/sid/'mix.wav'
        for _ in range(args.val_crops):
            start,x,s,rms=best_random_crop(path,frames,vrng,tries=20);assert s==sr
            vx.append(x);vinfo.append({'song':sid,'stem':'mix','start':start,'rms':rms})
    payload={'sample_rate':sr,'frames':frames,'seconds':args.seconds,'train':torch.stack(xs).half(),'train_weights':torch.tensor(weights),'train_info':info,'validation':torch.stack(vx).half(),'validation_info':vinfo,'split':split}
    torch.save(payload,args.output);print('train',payload['train'].shape,'val',payload['validation'].shape,'val_info',vinfo)
if __name__=='__main__':main()
