#!/usr/bin/env python3
"""Decompose the value effect into quantity and unit-value components.

Produces Online Appendix J. A binding export restriction lowers quantities, but
if the prices of surviving transactions rise, the value effect is smaller than
the quantity effect, or even positive. This script tests that directly.

Design (same specification as the static estimates)
  OLS, fixed effects partner x HS6, HS6 x year, partner x year
  clustered at partner x HS6
  outcomes  ln(v+1) value / ln(q) quantity (q > 0) / ln(v/q) unit value (q > 0)
  treatment aggregate, then category by category

Quantity and unit-value samples condition on q > 0 and therefore differ from
the value sample; the decomposition is additive only within the q > 0 sample.

Output: results/rev_price_decomposition.csv
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

CATEGORIES = ['dual_use', 'industrial_cap', 'luxury', 'military_tech']
CONTROLS = ['non_sanctioned', 'not_in_eu_list']
FE = 'partner_hs6 + hs6_year + partner_year'
VCOV = {'CRV1': 'partner_hs6'}


def prt(msg):
    print(msg, flush=True)


def get_nobs(m):
    try:
        return m.nobs
    except Exception:
        return -1


def run_spec(df, treat_var, outcome, label_sample, label_outcome, results):
    t0 = time.time()
    try:
        m = pf.feols(f'{outcome} ~ {treat_var} | {FE}', data=df, vcov=VCOV)
        beta = m.coef()[treat_var]
        se = m.se()[treat_var]
        pval = m.pvalue()[treat_var]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        prt(f'  {label_sample:16s} | {label_outcome:10s}: β={beta:+.4f} ({se:.4f}){sig} '
            f'[{(np.exp(beta)-1)*100:+.1f}%], N={get_nobs(m):,} ({time.time()-t0:.0f}s)')
        results.append({
            'sample': label_sample, 'outcome': label_outcome, 'beta': beta, 'se': se,
            'pval': pval, 'pct_effect': (np.exp(beta) - 1) * 100, 'nobs': get_nobs(m),
        })
    except Exception as e:
        prt(f'  {label_sample} | {label_outcome}: ERROR {e}')


if __name__ == '__main__':
    t_start = time.time()
    prt('=== Loading Panel ===')
    panel = pd.read_parquet(PANEL_PATH)
    imp = panel[panel['direction'] == 'to_russia'].copy()
    grp = imp.groupby('partner_hs6')['t'].nunique()
    imp = imp[imp['partner_hs6'].isin(grp[grp > 1].index)].copy()
    imp['ln_v'] = np.log(imp['v'] + 1)
    prt(f'Rows: {len(imp):,}')

    results = []

    def prep_outcomes(df):
        d = df.copy()
        dq = d[(d['q'] > 0) & d['q'].notna()].copy()
        dq['ln_q'] = np.log(dq['q'])
        dq['ln_uv'] = np.log(dq['v'] / dq['q'])
        return d, dq

    # ── 전체 (treated_t1) ──
    prt('\n=== AGGREGATE (treated_t1) ===')
    d, dq = prep_outcomes(imp)
    run_spec(d, 'treated_t1', 'ln_v', 'aggregate', 'value', results)
    run_spec(dq, 'treated_t1', 'ln_q', 'aggregate', 'quantity', results)
    run_spec(dq, 'treated_t1', 'ln_uv', 'aggregate', 'unit_value', results)

    # ── 카테고리별 ──
    for cat in CATEGORIES:
        prt(f'\n=== CATEGORY: {cat} ===')
        sub = imp[(imp['category'] == cat) | (imp['category'].isin(CONTROLS))].copy()
        g = sub.groupby('partner_hs6')['t'].nunique()
        sub = sub[sub['partner_hs6'].isin(g[g > 1].index)].copy()
        sub['treated_cat'] = ((sub['category'] == cat) &
                              (sub['hs6_sanctioned_at_t'] == 1) &
                              (sub['coalition_tier1'] == 1)).astype(int)
        d, dq = prep_outcomes(sub)
        run_spec(d, 'treated_cat', 'ln_v', cat, 'value', results)
        run_spec(dq, 'treated_cat', 'ln_q', cat, 'quantity', results)
        run_spec(dq, 'treated_cat', 'ln_uv', cat, 'unit_value', results)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'rev_price_decomposition.csv', index=False)
    prt(f'\nSaved: rev_price_decomposition.csv ({len(df)} specs)')

    # 분해 요약 (q>0 표본 내 근사: value ≈ quantity + unit_value)
    prt('\n=== DECOMPOSITION SUMMARY (β, q>0 표본 기준 근사) ===')
    for s in df['sample'].unique():
        sd = df[df['sample'] == s].set_index('outcome')
        try:
            bq = sd.loc['quantity', 'beta']
            bu = sd.loc['unit_value', 'beta']
            bv = sd.loc['value', 'beta']
            prt(f'  {s:16s}: value {bv:+.3f} | quantity {bq:+.3f} + unit_value {bu:+.3f} '
                f'= {bq+bu:+.3f}')
        except KeyError:
            pass

    prt(f'\n=== Total: {(time.time()-t_start)/60:.1f} min ===')
