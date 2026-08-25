#!/usr/bin/env python3
import argparse
import torch
from fnf_ai.vq_codec import TemporalVQCodec

def kmeans(x,k,iters=25,seed=1337):
    g=torch.Generator().manual_seed(seed);centers=[x[torch.randint(len(x),(1,),generator=g).item()]];dist=((x-centers[0])**2).sum(1)
    for _ in range(1,k):
        probs=dist/dist.sum().clamp_min(1e-8);idx=torch.multinomial(probs,1,generator=g).item();centers.append(x[idx]);nd=((x-centers[-1])**2).sum(1);dist=torch.minimum(dist,nd)
    c=torch.stack(centers)
    for i in range(iters):
        d=x.pow(2).sum(1,keepdim=True)+c.pow(2).sum(1).unsqueeze(0)-2*x@c.t();a=d.argmin(1);new=[]
        for j in range(k):
            pts=x[a==j];new.append(pts.mean(0) if len(pts) else c[j])
        nc=torch.stack(new);shift=float((nc-c).pow(2).mean().sqrt());c=nc
        if i%5==0 or i==iters-1:print('kmeans',i,'shift',shift,'used',int(a.unique().numel()),flush=True)
    return c

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);args=ap.parse_args();ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False);d=torch.load(args.data,map_location='cpu',weights_only=False);c=ck['config'];m=TemporalVQCodec(c['base_channels'],c['latent_dim'],c['codebook_size'],c['commitment']);m.load_state_dict(ck['model']);m.eval();zs=[]
    with torch.no_grad():
        for x in d['train'].float().split(9):zs.append(m.encoder(x).permute(0,2,1).reshape(-1,c['latent_dim']))
    z=torch.cat(zs);print('latent vectors',z.shape,flush=True);centers=kmeans(z,c['codebook_size']);m.quantizer.embedding.weight.data.copy_(centers);outck={**ck,'model':m.state_dict(),'codebook_fitted':'kmeans_on_continuous_latents'};torch.save(outck,args.output);print('saved',args.output)
if __name__=='__main__':main()
