#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from fnf_ai.vq_codec import TemporalVQCodec
from fnf_ai.codec_losses import codec_loss

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--config',required=True);ap.add_argument('--output',required=True);ap.add_argument('--resume');args=ap.parse_args()
    cfg=json.load(open(args.config));seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);data=torch.load(args.data,map_location='cpu',weights_only=False)
    train=data['train'].float();val=data['validation'].float();dl=DataLoader(TensorDataset(train),batch_size=cfg['batch_size'],shuffle=True,drop_last=True);vdl=DataLoader(TensorDataset(val),batch_size=cfg['batch_size'])
    model=TemporalVQCodec(cfg['base_channels'],cfg['latent_dim'],cfg['codebook_size'],cfg['commitment']);opt=torch.optim.AdamW(list(model.encoder.parameters())+list(model.decoder.parameters()),lr=cfg['learning_rate'],weight_decay=cfg.get('weight_decay',0),betas=(.9,.95));start=1;best=1e9;metrics=[]
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['epoch']+1;best=ck.get('best',1e9);metrics=ck.get('metrics',[])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);print('parameters',sum(p.numel() for p in model.parameters()),'start',start,flush=True)
    for ep in range(start,cfg['epochs']+1):
        model.train();tl=0.;n=0
        for (x,) in dl:
            opt.zero_grad(set_to_none=True);z=model.encoder(x);y=model.decode(z,x.shape[-1]);zero=z.new_zeros(());loss,m=codec_loss(y,x,zero,cfg['waveform_weight'],cfg['stft_weight'],0.0);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();tl+=float(loss.detach());n+=1
        model.eval();vl=vw=0.;vn=0
        with torch.no_grad():
            for (x,) in vdl:
                z=model.encoder(x);y=model.decode(z,x.shape[-1]);zero=z.new_zeros(());loss,m=codec_loss(y,x,zero,cfg['waveform_weight'],cfg['stft_weight'],0.0);vl+=float(loss)*len(x);vw+=m['wave_l1']*len(x);vn+=len(x)
        row={'epoch':ep,'train_loss':tl/n,'val_loss':vl/vn,'val_wave_l1':vw/vn};metrics.append(row);best=min(best,row['val_loss']);ck={'epoch':ep,'model':model.state_dict(),'optimizer':opt.state_dict(),'config':cfg,'best':best,'metrics':metrics,'data_meta':{'sample_rate':data['sample_rate'],'seconds':data['seconds'],'split':data['split'],'validation_info':data['validation_info']}};torch.save(ck,out/'last.pt');json.dump(metrics,open(out/'metrics.json','w'),indent=2)
        if row['val_loss']<=best+1e-12:torch.save(ck,out/'best.pt')
        if ep%cfg['checkpoint_every']==0:torch.save(ck,out/f'epoch_{ep:03d}.pt')
        print(f"epoch {ep:03d} train={row['train_loss']:.4f} val={row['val_loss']:.4f} wave={row['val_wave_l1']:.4f}",flush=True)
if __name__=='__main__':main()
