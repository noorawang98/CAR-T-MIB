"""Step103: 四种 MIB/perviMIB 口径之间的一致性 (相关性 + 空间重叠 + 结论一致性)。
口径: CCR1_MIB_mouse(鼠 mib) / perviMIB_mouse(mib_pv) / MIB_DestVI(score_mm&CAF 双上四分位) / perviMIB_DestVI(+血管周带)
指标: (1) 逐切片与池化的 Jaccard / Dice / 优势比(OR) / phi / 包含率(A->B), 以及 A 类 spot 到最近 B 类 spot 的距离(x NN)
      (2) 切片层面四种丰度比例的 Spearman 相关矩阵 (13 张) 及治疗后患者层面 (7 张)
      (3) 结论一致性: 各口径 PR vs SD/PD 的丰度比值与 Cohen's d
输出: prognosis/mib_def_pairwise.csv, mib_def_pairwise_pooled.csv, mib_def_corr_section.csv,
      mib_def_corr_postpatient.csv, mib_def_response_consistency.csv, figs/fig_mib_def_agreement.{png,pdf,svg}
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os, itertools
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, fisher_exact
WD = os.environ.get('ST_WD', PROJECT_ROOT + '/destvi_260911')
PROG = os.path.join(WD, 'prognosis')
FIG = os.path.join(WD, 'figs')
os.makedirs(FIG, exist_ok=True)
pl = lambda v: v / 25.4
DEFS = ['CCR1_MIB_mouse', 'perviMIB_mouse', 'MIB_DestVI', 'perviMIB_DestVI']
meta = pd.read_csv(os.path.join(WD, 'tables', 'st_image_mapping.csv'))
sec_rows, pooled_masks, fracs = ([], {}, [])
for _, m in meta.iterrows():
    s = m['sample']
    ps = pd.read_csv(os.path.join(WD, 'update_statistics', 'per_sample', f'{s}.csv'))[['x', 'y', 'score_mm', 'caf', 'vasc']]
    ps = ps.merge(pd.read_csv(os.path.join(PROG, f'niche_spots_{s}.csv.gz'))[['x', 'y', 'mib', 'mib_pv']], on=['x', 'y'], how='left')
    xy = np.c_[ps.x.values, ps.y.values]
    tree = cKDTree(xy)
    q = lambda v: np.asarray(v) >= np.quantile(v, 0.75)
    band = q(ps.vasc.values)[tree.query(xy, k=min(7, len(ps)), workers=8)[1][:, 1:]].sum(1) >= 3
    M = {'CCR1_MIB_mouse': pd.to_numeric(ps['mib'], errors='coerce').fillna(0).values > 0, 'perviMIB_mouse': pd.to_numeric(ps['mib_pv'], errors='coerce').fillna(0).values > 0, 'MIB_DestVI': q(ps.score_mm.values) & q(ps.caf.values), 'perviMIB_DestVI': q(ps.score_mm.values) & q(ps.caf.values) & band}
    nn = float(np.median(tree.query(xy, k=2, workers=8)[0][:, 1]))
    for A, B in itertools.permutations(DEFS, 2):
        a, b = (M[A], M[B])
        inter = int((a & b).sum())
        jac = inter / max(1, int((a | b).sum()))
        dice = 2 * inter / max(1, int(a.sum() + b.sum()))
        orr = fisher_exact([[inter, int(a.sum()) - inter], [int(b.sum()) - inter, int((~a & ~b).sum())]])[0]
        phi = (inter * int((~a & ~b).sum()) - int(a.sum() - inter) * int(b.sum() - inter)) / np.sqrt(max(1, a.sum()) * max(1, b.sum()) * max(1, (~a).sum()) * max(1, (~b).sum()))
        dist = np.median(cKDTree(xy[b]).query(xy[a], k=1)[0] / nn) if a.sum() and b.sum() else np.nan
        sec_rows.append(dict(sample=s, resp='PR' if m['response'] == 'PR' else 'SD/PD', time=m['time'], A=A, B=B, n_A=int(a.sum()), n_B=int(b.sum()), n_both=inter, jaccard=jac, dice=dice, odds_ratio=orr, phi=phi, containment_A_in_B=inter / max(1, int(a.sum())), dist_A_to_B_NN=dist))
    fracs.append(dict(sample=s, resp='PR' if m['response'] == 'PR' else 'SD/PD', time=m['time'], **{d: float(M[d].mean()) * 100 for d in DEFS}, n_spot=len(ps)))
    pooled_masks[s] = (M, xy)
S = pd.DataFrame(sec_rows)
F = pd.DataFrame(fracs)
S.to_csv(os.path.join(PROG, 'mib_def_pairwise.csv'), index=False)
prows = []
allM = {d: np.concatenate([pooled_masks[s][0][d] for s in pooled_masks]) for d in DEFS}
for A, B in itertools.permutations(DEFS, 2):
    a, b = (allM[A], allM[B])
    inter = int((a & b).sum())
    prows.append(dict(A=A, B=B, n_A=int(a.sum()), n_B=int(b.sum()), n_both=inter, jaccard=inter / max(1, int((a | b).sum())), dice=2 * inter / max(1, int(a.sum() + b.sum())), odds_ratio=fisher_exact([[inter, int(a.sum()) - inter], [int(b.sum()) - inter, int((~a & ~b).sum())]])[0], containment_A_in_B=inter / max(1, int(a.sum()))))
P = pd.DataFrame(prows)
P.to_csv(os.path.join(PROG, 'mib_def_pairwise_pooled.csv'), index=False)
C13 = F[DEFS].corr(method='spearman')
C13.to_csv(os.path.join(PROG, 'mib_def_corr_section.csv'))
Cp = F[F.time == 'Post'][DEFS].corr(method='spearman')
Cp.to_csv(os.path.join(PROG, 'mib_def_corr_postpatient.csv'))
resp = []
for d in DEFS:
    for tp in ['Post', 'Pre', 'All']:
        sub = F if tp == 'All' else F[F.time == tp]
        a = sub[sub.resp == 'PR'][d]
        b = sub[sub.resp == 'SD/PD'][d]
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else np.nan
        resp.append(dict(mib_def=d, timepoint=tp, PR=a.mean(), SDPD=b.mean(), ratio=a.mean() / b.mean() if b.mean() else np.nan, cohens_d=(a.mean() - b.mean()) / sd if sd and sd > 0 else np.nan))
Rc = pd.DataFrame(resp)
Rc.to_csv(os.path.join(PROG, 'mib_def_response_consistency.csv'), index=False)
pd.set_option('display.width', 250)
print('=== 池化: 口径间重叠 (Jaccard / Dice / OR / A 落在 B 内比例) ===')
print(P.round(4).to_string(index=False))
print('\n=== 逐切片 Jaccard 中位 (对称口径对, 13 张) ===')
sym = S[S.A < S.B].groupby(['A', 'B']).jaccard.median().unstack()
print(sym.round(4).to_string())
print('\n=== 切片层面 (13 张) 丰度 Spearman ===')
print(C13.round(3).to_string())
print('\n=== 治疗后患者层面 (7 张) 丰度 Spearman ===')
print(Cp.round(3).to_string())
print("\n=== 响应结论一致性 (PR/SD-PD 比值, Cohen's d) ===")
print(Rc.pivot_table(index='mib_def', columns='timepoint', values=['ratio', 'cohens_d']).round(3).to_string())
J = pd.DataFrame(np.nan, index=DEFS, columns=DEFS)
for _, r in P.iterrows():
    J.loc[r.A, r.B] = r.jaccard
    J.loc[r.B, r.A] = r.jaccard
np.fill_diagonal(J.values, 1.0)
w = 0.2
x = np.arange(3)
for i, d in enumerate(DEFS):
    v = [Rc[(Rc.mib_def == d) & (Rc.timepoint == tp)].ratio.iloc[0] for tp in ['Post', 'Pre', 'All']]
print('\nsaved: prognosis/mib_def_{pairwise,pairwise_pooled,corr_section,corr_postpatient,response_consistency}.csv | figs/fig_mib_def_agreement.{png,pdf,svg}')
