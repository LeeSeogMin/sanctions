#!/usr/bin/env python3
"""Military technology excluding pharmaceutical and odoriferous chapters.

Supplies the "excl. HS 30/33" row of Table 4 and the corresponding series in
Figure 3.

The military-technology category includes CBRN-related pharmaceutical and
chemical codes. Six codes in HS chapters 30 and 33 account for roughly 86% of
coalition-to-Russia military-technology trade value and more than 90% in
treated coalition cells. That trade continued after 2022 under the medical and
pharmaceutical end-use exemption, which dominates any estimate computed in
levels. Dropping those six codes isolates the remaining 90 BACI-observed
military-related codes.

Estimates
  A  static, all cohorts, singleton removal: OLS and PPML
  B  dynamic, 2022 cohort, unbinned, reference exactly t-1: OLS and PPML

Output: results/rev_military_expharma.csv
"""

import pandas as pd
import numpy as np
import pyfixest as pf
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

PANEL_PATH = Path('phase4_analysis/processed/baci_russia_panel.parquet')
RESULTS_DIR = Path('phase4_analysis/results')
CONTROLS = ['non_sanctioned', 'not_in_eu_list']
FE = 'partner_hs6 + hs6_year + partner_year'
VCOV = {'CRV1': 'partner_hs6'}
EVENT_TIMES = [-7, -6, -5, -4, -3, -2, 0, 1, 2]  # v3: 순수 t=-1 기준
EXCLUDE_CH = {'30', '33'}


def prt(msg):
    print(msg, flush=True)


def et_col(et):
    et = int(et)
    return f'et_m{abs(et)}_x_coal' if et < 0 else f'et_p{et}_x_coal'


def drop_pharma_military(df):
    """military_tech 코드 중 HS 챕터 30·33 행 제거 (통제군은 유지)"""
    ch = df['hs6'].astype(str).str.zfill(6).str[:2]
    drop = (df['category'] == 'military_tech') & ch.isin(EXCLUDE_CH)
    return df[~drop].copy()


if __name__ == '__main__':
    t_start = time.time()
    panel = pd.read_parquet(PANEL_PATH)
    imp_all = panel[panel['direction'] == 'to_russia'].copy()
    results = []

    # ── A. 정적 (06b 설계) ──
    prt('=== A. STATIC (excl. HS 30/33) ===')
    grp = imp_all.groupby('partner_hs6')['t'].nunique()
    imp = imp_all[imp_all['partner_hs6'].isin(grp[grp > 1].index)].copy()
    imp['ln_v'] = np.log(imp['v'] + 1)
    sub = imp[(imp['category'] == 'military_tech') | (imp['category'].isin(CONTROLS))]
    sub = drop_pharma_military(sub)
    sub['treated_cat'] = ((sub['category'] == 'military_tech') &
                          (sub['hs6_sanctioned_at_t'] == 1) &
                          (sub['coalition_tier1'] == 1)).astype(int)
    prt(f'rows: {len(sub):,}, treated cells: {sub["treated_cat"].sum():,}')

    for method, fn, dv in [('OLS', pf.feols, 'ln_v'), ('PPML', pf.fepois, 'v')]:
        t0 = time.time()
        try:
            m = fn(f'{dv} ~ treated_cat | {FE}', data=sub, vcov=VCOV)
            beta, se, pval = m.coef()['treated_cat'], m.se()['treated_cat'], m.pvalue()['treated_cat']
            prt(f'  {method:4s} static: β={beta:+.4f} ({se:.4f}) [{(np.exp(beta)-1)*100:+.1f}%] '
                f'p={pval:.2e} ({time.time()-t0:.0f}s)')
            results.append({'design': 'static', 'spec': method, 'event_time': np.nan,
                            'beta': beta, 'se': se, 'pval': pval,
                            'pct_effect': (np.exp(beta)-1)*100})
        except Exception as e:
            prt(f'  {method} static ERROR: {e}')
        pd.DataFrame(results).to_csv(RESULTS_DIR / 'rev_military_expharma.csv', index=False)

    # ── B. 동학 (2022 코호트, b6 설계) ──
    prt('\n=== B. DYNAMICS 2022 cohort (excl. HS 30/33) ===')
    df = imp_all.copy()
    df['ln_v'] = np.log(df['v'] + 1)
    df['et'] = df['t'] - 2022
    df = df[(df['sanction_year'] == 2022) | (df['sanctioned'].fillna(0) == 0)]
    df = df[(df['category'] == 'military_tech') | (df['category'].isin(CONTROLS))]
    df = drop_pharma_military(df)
    for e in EVENT_TIMES:
        df[et_col(e)] = ((df['et'] == e) & (df['sanctioned'] == 1) &
                         (df['coalition_tier1'] == 1)).astype(int)
    et_vars = ' + '.join(et_col(e) for e in EVENT_TIMES)
    prt(f'rows: {len(df):,}')

    for method, fn, dv in [('OLS', pf.feols, 'ln_v'), ('PPML', pf.fepois, 'v')]:
        t0 = time.time()
        try:
            m = fn(f'{dv} ~ {et_vars} | {FE}', data=df, vcov=VCOV)
            prt(f'  {method} ES converged ({time.time()-t0:.0f}s):')
            for e in EVENT_TIMES + [-1]:
                if e == -1:
                    results.append({'design': 'dynamic', 'spec': method, 'event_time': -1,
                                    'beta': 0.0, 'se': 0.0, 'pval': 1.0, 'pct_effect': 0.0})
                    continue
                col = et_col(e)
                b, s, pv = m.coef()[col], m.se()[col], m.pvalue()[col]
                sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
                prt(f'    t={e:+d}: β={b:+.4f} ({s:.4f}){sig} [{(np.exp(b)-1)*100:+.1f}%]')
                results.append({'design': 'dynamic', 'spec': method, 'event_time': e,
                                'beta': b, 'se': s, 'pval': pv,
                                'pct_effect': (np.exp(b)-1)*100})
        except Exception as e2:
            prt(f'  {method} ES ERROR: {e2}')
        pd.DataFrame(results).to_csv(RESULTS_DIR / 'rev_military_expharma.csv', index=False)

    prt(f'\nSaved: rev_military_expharma.csv')
    prt(f'=== Total: {(time.time()-t_start)/60:.1f} min ===')
