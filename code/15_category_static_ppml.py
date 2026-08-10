#!/usr/bin/env python3
"""Category-specific static treatment effects estimated by PPML.

Supplies the PPML columns of Table 4, alongside the OLS estimates. The two
estimators are reported together because they diverge for luxury goods and
military technology: OLS weights each product flow equally, whereas PPML fits
the conditional mean in levels, so large flows carry more influence.

Design matches the aggregate static specification (all cohorts, singleton
removal, treatment = category x sanctioned-at-t x coalition); only the
estimator changes to fepois.

Output: results/rev_category_static_ppml.csv
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


if __name__ == '__main__':
    t_start = time.time()
    prt('=== Loading Panel (06b design) ===')
    panel = pd.read_parquet(PANEL_PATH)
    imp = panel[panel['direction'] == 'to_russia'].copy()
    grp = imp.groupby('partner_hs6')['t'].nunique()
    imp = imp[imp['partner_hs6'].isin(grp[grp > 1].index)].copy()
    imp['ln_v'] = np.log(imp['v'] + 1)
    prt(f'Rows: {len(imp):,}')

    results = []
    for cat in CATEGORIES:
        sub = imp[(imp['category'] == cat) | (imp['category'].isin(CONTROLS))].copy()
        sub['treated_cat'] = ((sub['category'] == cat) &
                              (sub['hs6_sanctioned_at_t'] == 1) &
                              (sub['coalition_tier1'] == 1)).astype(int)
        prt(f'\n=== {cat} ({len(sub):,} rows) ===')

        # OLS 재현 (Table 4 대조 검증)
        t0 = time.time()
        try:
            m = pf.feols('ln_v ~ treated_cat | ' + FE, data=sub, vcov=VCOV)
            beta, se, pval = m.coef()['treated_cat'], m.se()['treated_cat'], m.pvalue()['treated_cat']
            prt(f'  OLS : β={beta:+.4f} ({se:.4f}) [{(np.exp(beta)-1)*100:+.1f}%] ({time.time()-t0:.0f}s)')
            results.append({'category': cat, 'method': 'OLS', 'beta': beta, 'se': se,
                            'pval': pval, 'pct_effect': (np.exp(beta)-1)*100, 'nobs': get_nobs(m)})
        except Exception as e:
            prt(f'  OLS ERROR: {e}')

        # PPML
        t0 = time.time()
        try:
            m = pf.fepois('v ~ treated_cat | ' + FE, data=sub, vcov=VCOV)
            beta, se, pval = m.coef()['treated_cat'], m.se()['treated_cat'], m.pvalue()['treated_cat']
            prt(f'  PPML: β={beta:+.4f} ({se:.4f}) [{(np.exp(beta)-1)*100:+.1f}%] '
                f'(p={pval:.2e}, {time.time()-t0:.0f}s)')
            results.append({'category': cat, 'method': 'PPML', 'beta': beta, 'se': se,
                            'pval': pval, 'pct_effect': (np.exp(beta)-1)*100, 'nobs': get_nobs(m)})
        except Exception as e:
            prt(f'  PPML ERROR ({time.time()-t0:.0f}s): {e}')

        pd.DataFrame(results).to_csv(RESULTS_DIR / 'rev_category_static_ppml.csv', index=False)
        prt('  Saved (누적): rev_category_static_ppml.csv')

    # Table 4 대조
    prt('\n=== 검증: OLS vs Table 4 발표치 ===')
    b3 = RESULTS_DIR / 'b3_category_heterogeneity.csv'
    if b3.exists():
        pub = pd.read_csv(b3)
        new = pd.DataFrame(results)
        for cat in CATEGORIES:
            try:
                b_new = new[(new['category'] == cat) & (new['method'] == 'OLS')].iloc[0]['beta']
                b_pub = pub[pub['category'] == cat].iloc[0]['beta']
                prt(f'  {cat:16s}: Δβ = {abs(b_new-b_pub):.2e}')
            except Exception:
                pass

    prt(f'\n=== Total: {(time.time()-t_start)/60:.1f} min ===')
