"""Robustness to Russia's 2014 agricultural counter-embargo.

Supplies the "Excl. HS 01-24" row of Table 6. Russia's retaliatory ban on
agricultural imports from sanctioning countries has been in force since August
2014, throughout the sample. Those goods are mostly unsanctioned and therefore
sit in the control group, so the embargo could depress the controls. Its level
effect is absorbed by the partner x HS6 fixed effects; this script removes the
residual concern about time-varying embargo intensity by dropping the affected
chapters outright.

Design
  top-50 partner sample, fixed before the exclusion so the partner set is
  unchanged; HS chapters 01-24 dropped; singletons removed again
  v ~ treated | partner x HS6 + HS6 x year + partner x year, clustered
  at partner x HS6

Output: results/b20_agri_robustness.csv
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings('ignore')

PANEL_PATH = Path('phase4_analysis/processed/baci_russia_panel.parquet')
OUT = Path('phase4_analysis/results/b20_agri_robustness.csv')
AGRI_CHAPTERS = {f'{c:02d}' for c in range(1, 25)}  # HS 01–24


def prt(msg):
    print(msg, flush=True)


if __name__ == '__main__':
    t0 = time.time()
    panel = pd.read_parquet(PANEL_PATH)
    imp = panel[panel['direction'] == 'to_russia'].copy()
    imp['hs6'] = imp['hs6'].astype(str).str.zfill(6)

    # 싱글턴 제거 (06b/12/19와 동일)
    grp = imp.groupby('partner_hs6')['t'].nunique()
    imp = imp[imp['partner_hs6'].isin(grp[grp > 1].index)].copy()

    # Top-50 파트너 (제외 전 기준 — 19번과 동일 파트너 셋)
    top50 = imp.groupby('partner')['v'].sum().nlargest(50).index
    imp_top = imp[imp['partner'].isin(top50)].copy()
    grp = imp_top.groupby('partner_hs6')['t'].nunique()
    imp_top = imp_top[imp_top['partner_hs6'].isin(grp[grp > 1].index)].copy()
    n_base = len(imp_top)
    prt(f'Top-50 baseline rows: {n_base:,}')

    # HS 01–24 제외
    agri = imp_top['hs6'].str[:2].isin(AGRI_CHAPTERS)
    n_agri = agri.sum()
    n_agri_treated = ((agri) & (imp_top['treated_t1'] == 1)).sum()
    n_agri_sanc = imp_top.loc[agri & (imp_top['sanctioned'] == 1), 'hs6'].nunique()
    imp_ex = imp_top[~agri].copy()
    grp = imp_ex.groupby('partner_hs6')['t'].nunique()
    imp_ex = imp_ex[imp_ex['partner_hs6'].isin(grp[grp > 1].index)].copy()
    prt(f'HS01–24 제외: {n_agri:,} 행 제거 ({n_agri/n_base*100:.1f}%), '
        f'그중 treated_t1 {n_agri_treated:,} 행 / 제재 HS6 {n_agri_sanc}개')
    prt(f'추정 표본: {len(imp_ex):,} 행')

    FE = 'partner_hs6 + hs6_year + partner_year'
    prt('\nPPML: v ~ treated_t1 | ' + FE + '  (ex. HS01-24)')
    m = pf.fepois(f'v ~ treated_t1 | {FE}', data=imp_ex, vcov={'CRV1': 'partner_hs6'})
    beta, se, pv = m.coef()['treated_t1'], m.se()['treated_t1'], m.pvalue()['treated_t1']
    pct = (np.exp(beta) - 1) * 100
    prt(f'  β={beta:.4f} (SE {se:.4f}, p={pv:.4g}) → {pct:.1f}%   [{time.time()-t0:.0f}s]')
    prt('  비교: baseline Top-50 −0.473 (−37.7%)')

    pd.DataFrame([{
        'spec': 'ex_agri_HS01_24_top50', 'beta': beta, 'se': se, 'pval': pv,
        'pct_effect': pct, 'nobs': getattr(m, '_N', np.nan),
        'rows_dropped': int(n_agri), 'rows_dropped_share': n_agri / n_base,
        'treated_rows_dropped': int(n_agri_treated),
        'sanctioned_hs6_dropped': int(n_agri_sanc),
        'baseline_beta_top50': -0.473,
    }]).to_csv(OUT, index=False)
    prt(f'저장: {OUT.name}')
