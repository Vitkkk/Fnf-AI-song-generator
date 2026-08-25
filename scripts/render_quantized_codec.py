#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import torch,torchaudio
from fnf_ai.vq_codec import TemporalVQCodec
ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);ap.add_argument('--index',type=int,default=4);a=ap.parse_args();ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);d=torch.load(a.data,map_location='cpu',weights_only=False);c=ck['config'];m=TemporalVQCodec(c['base_channels'],c['latent_dim'],c['codebook_size'],c['commitment']);m.load_state_dict(ck['model']);m.eval();x=d['validation'][a.index:a.index+1].float()
with torch.no_grad():z=m.encoder(x);q,tok,vq,ppl,usage=m.quantizer(z);y=m.decode(q,x.shape[-1]);yc=m.decode(z,x.shape[-1])
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);sr=d['sample_rate'];sil=torch.zeros(1,1,int(.3*sr));torchaudio.save(str(out/'original.wav'),x[0],sr);torchaudio.save(str(out/'continuous.wav'),yc[0],sr);torchaudio.save(str(out/'quantized.wav'),y[0],sr);torchaudio.save(str(out/'comparison_original_continuous_quantized.wav'),torch.cat([x,sil,yc,sil,y],-1)[0],sr);stats={'info':d['validation_info'][a.index],'continuous_mae':float((x-yc).abs().mean()),'quantized_mae':float((x-y).abs().mean()),'perplexity':float(ppl),'usage':float(usage),'unique_tokens':int(tok.unique().numel()),'tokens':int(tok.numel())};json.dump(stats,open(out/'stats.json','w'),indent=2);print(json.dumps(stats,indent=2))
