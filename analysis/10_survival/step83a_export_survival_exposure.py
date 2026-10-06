"""Step83a: 导出 bin50_v3 的 perviMIB / MIB 丰度 (逐切片 + 患者层面) 供生存分析。
口径: perviMIB_mouse = 鼠口径 mib_pv; perviMIB_DestVI = score_mm&caf 双 top25% 且 knn>=3 血管周带;
      CCR1_MIB_mouse = 鼠口径 mib;  MIB_DestVI = score_mm&caf 双 top25%
患者层面: Post(治疗后, 主分析) / Pre / All 三种取法 (对同一患者多切片取均值)
输出: prognosis/survival_exposure_slide.csv, survival_exposure_patient.csv
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
WD = os.environ.get('ST_WD', PROJECT_ROOT + '/destvi_260911')
OUT = os.path.join(WD, 'prognosis')
meta = pd.read_csv(os.path.join(WD, 'tables', 'st_image_mapping.csv'))
clin = pd.read_csv(os.path.join(WD, 'tables', 'clinical_parsed.csv'))
C = clin.drop_duplicates('patient').set_index('patient')[['os_event', 'os_days', 'pfs_event', 'pfs_days', 'response', 'group']]
rows = []
for _, m in meta.iterrows():
    s = m['sample']
    d = pd.read_csv(os.path.join(WD, 'update_statistics', 'per_sample', f'{s}.csv'))
    g = os.path.join(OUT, f'niche_spots_{s}.csv.gz')
    if os.path.exists(g):
        d = d.merge(pd.read_csv(g)[['x', 'y', 'mib', 'mib_pv']], on=['x', 'y'], how='left')
    xy = np.c_[d.x.values, d.y.values]
    tree = cKDTree(xy)
    q = lambda v: np.asarray(v) >= np.quantile(v, 0.75)
    band = q(d.vasc.values)[tree.query(xy, k=7, workers=8)[1][:, 1:]].sum(1) >= 3
    piv_m = pd.to_numeric(d.get('mib_pv'), errors='coerce').fillna(0).values > 0
    mib_m = pd.to_numeric(d.get('mib'), errors='coerce').fillna(0).values > 0
    mib_d = q(d.score_mm.values) & q(d.caf.values)
    piv_d = mib_d & band
    rows.append(dict(sample=s, patient=m['patient'], time=m['time'], resp='PR' if m['response'] == 'PR' else 'SD/PD', tissue=m['tissue'], n_spot=len(d), perviMIB_mouse=float(piv_m.mean()), perviMIB_DestVI=float(piv_d.mean()), CCR1_MIB_mouse=float(mib_m.mean()), MIB_DestVI=float(mib_d.mean())))
S = pd.DataFrame(rows)
S.to_csv(os.path.join(OUT, 'survival_exposure_slide.csv'), index=False)
P = []
for (pat, tp), g in S.groupby(['patient', 'time']):
    P.append(dict(patient=pat, time=tp, n_slide=len(g), perviMIB_mouse=g.perviMIB_mouse.mean(), perviMIB_DestVI=g.perviMIB_DestVI.mean(), CCR1_MIB_mouse=g.CCR1_MIB_mouse.mean(), MIB_DestVI=g.MIB_DestVI.mean(), resp=g.resp.iloc[0]))
P = pd.DataFrame(P)
wide = P.pivot(index='patient', columns='time', values=['perviMIB_mouse', 'perviMIB_DestVI', 'CCR1_MIB_mouse', 'MIB_DestVI'])
wide.columns = [f'{a}_{b}' for a, b in wide.columns]
allm = S.groupby('patient')[['perviMIB_mouse', 'perviMIB_DestVI', 'CCR1_MIB_mouse', 'MIB_DestVI']].mean().add_suffix('_All')
resp = S.groupby('patient').resp.first()
T = wide.join(allm).join(resp).join(C)
T.to_csv(os.path.join(OUT, 'survival_exposure_patient.csv'))
pd.set_option('display.width', 250)
print('=== 逐切片丰度 ===')
print(S.round(4).to_string(index=False))
print('\n=== 患者层面 (含生存) ===')
cols = [c for c in T.columns if any((k in c for k in ['perviMIB', 'CCR1_MIB', 'MIB_DestVI', 'os_', 'pfs_', 'resp']))]
print(T[cols].round(4).to_string())
print('\n事件: OS 事件 %d/%d; PFS 事件 %d/%d' % (int(T.os_event.sum()), len(T), int(pd.to_numeric(T.pfs_event, errors='coerce').fillna(0).sum()), len(T)))
print('有 Post 切片的患者: %d; 有 Pre: %d' % (T.perviMIB_mouse_Post.notna().sum(), T.perviMIB_mouse_Pre.notna().sum()))
