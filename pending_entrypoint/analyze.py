#!/usr/bin/env python3
"""Analysis-only entry points; inputs are CSV tables, outputs are CSV."""
import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('task',choices=['car','mib','neighborhood','roc','pseudobulk','fourier','mixed','mics'])
    ap.add_argument('--input',required=True)
    ap.add_argument('--output',default=os.environ.get('OUTPUT_DIR','results'))
    ap.add_argument('--features',default='')
    ap.add_argument('--nperm',type=int,default=300)
    ap.add_argument('--nboot',type=int,default=2000)
    a=ap.parse_args(); d=pd.read_csv(a.input); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    features=[v for v in a.features.split(',') if v]
    if a.task=='car':
        from car_call import call_spots
        result,rate=call_spots(d); result.to_csv(out/'car_calls.csv',index=False)
        pd.DataFrame([dict(control_rate=rate)]).to_csv(out/'car_background.csv',index=False)
    elif a.task=='mics':
        from mics import transform
        transform(d).to_csv(out/'mics_cells.csv',index=False)
    elif a.task=='mib':
        from mib import annotate
        if 'section' in d:
            result=pd.concat([annotate(g,g[['x','y']]).assign(section=section) for section,g in d.groupby('section',sort=False)]).sort_index()
        else:
            result=annotate(d,d[['x','y']])
        result.to_csv(out/'mib_regions.csv',index=False)
    elif a.task=='pseudobulk':
        from pseudobulk import aggregate
        pb,meta=aggregate(d[features],d.section,d.selected.astype(bool))
        pb.to_csv(out/'pseudobulk_counts.csv'); meta.to_csv(out/'pseudobulk_metadata.csv',index=False)
    elif a.task=='roc':
        from roc import auc,perm_p
        result=auc(d.score,d.label.astype(bool),nboot=a.nboot)
        p,obs,_=perm_p(d.score,np.where(d.label.astype(bool),'PR','GR'),d.section,nperm=a.nperm)
        result.update(P=p,predictor='score',target='label')
        pd.DataFrame([result]).to_csv(out/'roc.csv',index=False)
    elif a.task=='neighborhood':
        from neighborhood import curve_knn,bh
        rows=[]; rng=np.random.default_rng(0)
        for section,g in d.groupby('section',sort=False):
            obs,null=curve_knn(g.x.to_numpy(),g.y.to_numpy(),g.X.astype(bool).to_numpy(),g.Y.astype(bool).to_numpy(),n_perm=a.nperm,rng=rng)
            if obs is None: continue
            p=(1+(null>=obs).sum(0))/(a.nperm+1)
            q=bh(p)
            rows.extend(dict(section=section,k=i+1,enrichment=v,P=p[i],q=q[i],family=section) for i,v in enumerate(obs))
        pd.DataFrame(rows).to_csv(out/'enrichment.csv',index=False)
    elif a.task=='mixed':
        from model_family import fit_family
        fit_family(d,features).to_csv(out/'mixed_family.csv',index=False)
    elif a.task=='fourier':
        from fourier import to_grid,fill,spec,masks,recon,BANDS,band_corr,toroidal_null,pooled_plus_one_p,bh
        from scipy.spatial import cKDTree
        fields={}; power=[]
        for section,g in d.groupby('section',sort=False):
            xy=g[['x','y']].to_numpy(); spacing=np.median(cKDTree(xy).query(xy,k=2)[0][:,1])
            fields[section]={}
            for key in ['scissor']+features:
                F,r=spec(fill(to_grid(g.x.to_numpy(),g.y.to_numpy(),g[key].to_numpy(),spacing)))
                ms=masks(r); fields[section][key]=recon(F,ms)
                if key=='scissor':
                    pw=np.array([(np.abs(F)**2*m).sum() for m in ms]); pw=pw/pw.sum()
                    power.extend(dict(section=section,band=BANDS[i][2],fraction=v) for i,v in enumerate(pw))
        rng=np.random.default_rng(0); rows=[]
        for feature in features:
            for i,band in enumerate(BANDS):
                obs=[]; nulls=[]
                for section,fields_ in fields.items():
                    f,s=fields_[feature][i],fields_['scissor'][i]
                    obs.append(band_corr(f,s)); nulls.append(toroidal_null(f,s,a.nperm,rng))
                r=float(np.nanmean(obs)); p=pooled_plus_one_p(r,nulls)
                rows.append(dict(feature=feature,band=band[2],mean_r=r,P=p,n_sections=len(obs),n_null=len(obs)*a.nperm))
        result=pd.DataFrame(rows); result['q']=bh(result.P.to_numpy())
        result.to_csv(out/'fourier_correlations.csv',index=False)
        pd.DataFrame(power).to_csv(out/'fourier_power.csv',index=False)
if __name__=='__main__': main()
