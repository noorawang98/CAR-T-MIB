"""Step102: perviMIB spot 内的 DeSTVI 细胞类型比例 —— PR vs SD/PD 比较与图。
输入 prognosis/pervimib_deconvolution.csv (step100 生成: 切片 x 口径 x 29 类细胞比例均值)
统计: 逐细胞类型 Mann-Whitney (切片为单位) + BH; 同时给出组均值与 log2 比值。
输出: prognosis/pervimib_deconv_tests.csv, figs/fig_pervimib_deconv.{png,pdf,svg}
"""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
import os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
WD = os.environ.get('ST_WD', PROJECT_ROOT + '/destvi_260911')
PROG = os.path.join(WD, 'prognosis')
FIG = os.path.join(WD, 'figs')
os.makedirs(FIG, exist_ok=True)
pl = lambda v: v / 25.4
D = pd.read_csv(os.path.join(PROG, 'pervimib_deconvolution.csv'))
D = D[D.n_pervimib >= 5]
ct = [c for c in D.columns if c.startswith('stprop_')]
short = lambda c: c.replace('stprop_', '').replace('_', ' ')
out = []
for defn, g in D.groupby('mib_def'):
    a = g[g.resp == 'PR']
    b = g[g.resp == 'SD/PD']
    for c in ct:
        x, y = (a[c].astype(float), b[c].astype(float))
        p = mannwhitneyu(x, y).pvalue if len(x) > 1 and len(y) > 1 else np.nan
        out.append(dict(mib_def=defn, cell_type=short(c), mean_PR=x.mean(), mean_SDPD=y.mean(), diff=x.mean() - y.mean(), log2_ratio=np.log2((x.mean() + 1e-06) / (y.mean() + 1e-06)), p_MWU=p, n_PR=len(x), n_SDPD=len(y)))
T = pd.DataFrame(out)
T['q_BH'] = np.nan
for defn, idx in T.groupby('mib_def').groups.items():
    T.loc[idx, 'q_BH'] = multipletests(T.loc[idx, 'p_MWU'].fillna(1), method='fdr_bh')[1]
T = T.sort_values(['mib_def', 'p_MWU'])
T.to_csv(os.path.join(PROG, 'pervimib_deconv_tests.csv'), index=False)
pd.set_option('display.width', 240)
for defn in T.mib_def.unique():
    sub = T[T.mib_def == defn].head(12)
    print(f'\n=== perviMIB ({defn}) 内细胞类型 PR vs SD/PD (按 p 排序前 12) ===')
    print(sub[['cell_type', 'mean_PR', 'mean_SDPD', 'diff', 'log2_ratio', 'p_MWU', 'q_BH']].round(4).to_string(index=False))
print('\n显著项 (q<0.05):')
print(T[T.q_BH < 0.05][['mib_def', 'cell_type', 'mean_PR', 'mean_SDPD', 'log2_ratio', 'p_MWU', 'q_BH']].round(4).to_string(index=False) if (T.q_BH < 0.05).any() else '  无')
print('\nsaved: prognosis/pervimib_deconv_tests.csv | figs/fig_pervimib_deconv.{png,pdf,svg}')
