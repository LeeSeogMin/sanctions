#!/usr/bin/env python3
"""
04_gateway_test.py
Coalition → Gateway export test (transshipment evidence)

If coalition members increase exports of sanctioned products to gateway countries
after 2022, this is direct evidence of transshipment (goods flow through gateways).
"""

import pandas as pd
import numpy as np
import pyfixest as pf
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

BACI_DIR = Path('data/baci')
RESULTS_DIR = Path('output/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EU27 = {40, 56, 100, 191, 196, 203, 208, 233, 246, 251, 276, 300,
        348, 372, 380, 428, 440, 442, 470, 528, 616, 620, 642,
        703, 705, 724, 752}
TIER1_NON_EU = {826, 842, 124, 392, 579, 757}
TIER1 = EU27 | TIER1_NON_EU
RUSSIA = 643

# Top gateway countries (from B-4 exposure analysis)
GATEWAY = {784, 51, 417, 792, 156, 860, 398, 762, 268, 356}
# UAE=784, ARM=51, KGZ=417, TUR=792, CHN=156, UZB=860, KAZ=398, TJK=762, GEO=268, IND=356

COUNTRY_NAMES = {784: 'UAE', 51: 'Armenia', 417: 'Kyrgyzstan', 792: 'Turkey',
                 156: 'China', 860: 'Uzbekistan', 398: 'Kazakhstan', 762: 'Tajikistan',
                 268: 'Georgia', 356: 'India'}

def prt(msg):
    print(msg, flush=True)

def get_nobs(m):
    try: return m.nobs
    except:
        try: return m._N
        except:
            try: return len(m._Y)
            except: return -1

def stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return ''


def main():
    prt('='*60)
    prt('B-7.2: Coalition→Gateway Export Test')
    prt('='*60)

    # Load sanctions classification
    sanctions = pd.read_csv('data/processed/sanctions_hs6_master.csv')
    sanctions['hs6'] = sanctions['hs6'].astype(str).str.zfill(6)
    sanctioned_set = set(sanctions[sanctions['sanctioned'] == 1]['hs6'])
    sanc_year_map = sanctions[sanctions['sanctioned'] == 1].set_index('hs6')['sanction_year'].to_dict()

    prt(f'Sanctioned HS6 codes: {len(sanctioned_set)}')

    # Load BACI data: coalition exporter → gateway/non-gateway importer
    all_gateway = []
    all_nongw = []

    # First pass: identify top-10 non-gateway countries (for placebo)
    prt('\nIdentifying top non-gateway destinations (2019)...')
    fpath_2019 = BACI_DIR / 'BACI_HS92_Y2019_V202601.csv'
    df_2019 = pd.read_csv(fpath_2019, dtype={'k': str})
    coal_exports = df_2019[df_2019['i'].isin(TIER1)]
    non_gw_targets = set(coal_exports['j'].unique()) - TIER1 - GATEWAY - {RUSSIA}
    top_nongw = coal_exports[coal_exports['j'].isin(non_gw_targets)].groupby('j')['v'].sum().sort_values(ascending=False).head(10).index
    NON_GATEWAY = set(top_nongw)
    prt(f'  Top-10 non-gateway: {sorted(NON_GATEWAY)}')
    del df_2019, coal_exports

    # Load year by year
    for year in range(2015, 2025):
        fpath = BACI_DIR / f'BACI_HS92_Y{year}_V202601.csv'
        if not fpath.exists():
            prt(f'  {year}: not found')
            continue

        t0 = time.time()
        df = pd.read_csv(fpath, dtype={'k': str})

        # Coalition → Gateway
        df_cg = df[(df['i'].isin(TIER1)) & (df['j'].isin(GATEWAY))].copy()
        df_cg['hs6'] = df_cg['k'].str.zfill(6)
        agg_cg = df_cg.groupby(['i', 'j', 'hs6']).agg(v=('v', 'sum'), q=('q', 'sum')).reset_index()
        agg_cg['t'] = year
        agg_cg.rename(columns={'i': 'exporter', 'j': 'importer'}, inplace=True)
        agg_cg['is_gateway'] = 1
        all_gateway.append(agg_cg)

        # Coalition → Non-Gateway (placebo)
        df_cng = df[(df['i'].isin(TIER1)) & (df['j'].isin(NON_GATEWAY))].copy()
        df_cng['hs6'] = df_cng['k'].str.zfill(6)
        agg_cng = df_cng.groupby(['i', 'j', 'hs6']).agg(v=('v', 'sum'), q=('q', 'sum')).reset_index()
        agg_cng['t'] = year
        agg_cng.rename(columns={'i': 'exporter', 'j': 'importer'}, inplace=True)
        agg_cng['is_gateway'] = 0
        all_nongw.append(agg_cng)

        prt(f'  {year}: Gateway {len(agg_cg):,}, Non-GW {len(agg_cng):,} ({time.time()-t0:.0f}s)')

        del df

    cg = pd.concat(all_gateway, ignore_index=True)
    cng = pd.concat(all_nongw, ignore_index=True)

    # Add sanctions variables to both
    for df in [cg, cng]:
        df['sanctioned'] = df['hs6'].isin(sanctioned_set).astype(int)
        df['sanction_year'] = df['hs6'].map(sanc_year_map).fillna(0).astype(int)
        df['hs6_sanctioned_at_t'] = ((df['sanctioned'] == 1) &
                                       (df['t'] >= df['sanction_year']) &
                                       (df['sanction_year'] > 0)).astype(int)
        df['post'] = (df['t'] >= 2022).astype(int)
        df['sanc_post'] = (df['sanctioned'] * df['post']).astype(int)
        df['ln_v'] = np.log(df['v'] + 1)
        df['exporter_importer_hs6'] = (df['exporter'].astype(str) + '_' +
                                         df['importer'].astype(str) + '_' + df['hs6'])
        df['exporter_importer'] = df['exporter'].astype(str) + '_' + df['importer'].astype(str)
        df['hs6_year'] = df['hs6'] + '_' + df['t'].astype(str)
        df['ei_year'] = df['exporter_importer'] + '_' + df['t'].astype(str)

    prt(f'\nCoalition→Gateway: {len(cg):,} rows, {cg["exporter"].nunique()} exporters, {cg["importer"].nunique()} gateways')
    prt(f'Coalition→Non-GW:  {len(cng):,} rows')

    # ===== Descriptive =====
    prt('\n' + '='*60)
    prt('Descriptive: Coalition→Gateway sanctioned-product trade')
    prt('='*60)

    sanc_cg = cg[cg['sanctioned'] == 1]
    n_pre = sanc_cg[sanc_cg['t'] < 2022]['t'].nunique()
    n_post = sanc_cg[sanc_cg['t'] >= 2022]['t'].nunique()

    pre_ann = sanc_cg[sanc_cg['t'] < 2022]['v'].sum() / max(n_pre, 1) / 1e6
    post_ann = sanc_cg[sanc_cg['t'] >= 2022]['v'].sum() / max(n_post, 1) / 1e6
    pct = (post_ann - pre_ann) / pre_ann * 100 if pre_ann > 0 else 0

    prt(f'  Annual sanctioned exports to gateways:')
    prt(f'    Pre-2022:  ${pre_ann:.1f}B')
    prt(f'    Post-2022: ${post_ann:.1f}B')
    prt(f'    Change:    {pct:+.1f}%')

    prt('\n  By gateway country:')
    for gw in sorted(GATEWAY):
        gw_sanc = sanc_cg[sanc_cg['importer'] == gw]
        gw_pre = gw_sanc[gw_sanc['t'] < 2022]['v'].sum() / max(n_pre, 1) / 1e6
        gw_post = gw_sanc[gw_sanc['t'] >= 2022]['v'].sum() / max(n_post, 1) / 1e6
        gw_pct = (gw_post - gw_pre) / gw_pre * 100 if gw_pre > 0.01 else 0
        name = COUNTRY_NAMES.get(gw, str(gw))
        prt(f'    {name:15s}: ${gw_pre:8.2f}B → ${gw_post:8.2f}B ({gw_pct:+.1f}%)')

    # Non-sanctioned comparison
    nonsanc_cg = cg[cg['sanctioned'] == 0]
    ns_pre = nonsanc_cg[nonsanc_cg['t'] < 2022]['v'].sum() / max(n_pre, 1) / 1e6
    ns_post = nonsanc_cg[nonsanc_cg['t'] >= 2022]['v'].sum() / max(n_post, 1) / 1e6
    ns_pct = (ns_post - ns_pre) / ns_pre * 100 if ns_pre > 0 else 0
    prt(f'\n  Non-sanctioned exports to gateways:')
    prt(f'    Pre: ${ns_pre:.1f}B → Post: ${ns_post:.1f}B ({ns_pct:+.1f}%)')

    # ===== Regression =====
    prt('\n' + '='*60)
    prt('Regression: Coalition→Gateway DiD')
    prt('='*60)

    results = []

    # KEY: sanc_post varies at HS6×year level, so hs6_year FE absorbs it.
    # Use exporter_year + importer_year FE instead (no hs6_year) for within-pair specs.
    # For pooled spec, use is_gateway interaction with hs6_year FE (cross-sectional variation).

    cg['exporter_year'] = cg['exporter'].astype(str) + '_' + cg['t'].astype(str)
    cg['importer_year'] = cg['importer'].astype(str) + '_' + cg['t'].astype(str)
    cng['exporter_year'] = cng['exporter'].astype(str) + '_' + cng['t'].astype(str)
    cng['importer_year'] = cng['importer'].astype(str) + '_' + cng['t'].astype(str)

    # Singleton removal for gateway
    grp = cg.groupby('exporter_importer_hs6')['t'].nunique()
    multi = grp[grp > 1].index
    cg_clean = cg[cg['exporter_importer_hs6'].isin(multi)].copy()

    # Spec 1: Gateway — sanc×post | pair_hs6 + exporter_year + importer_year
    prt('\n--- Spec 1: Coalition→Gateway (sanc×post | pair_hs6 + exp_yr + imp_yr) ---')
    try:
        t0 = time.time()
        m = pf.feols('ln_v ~ sanc_post | exporter_importer_hs6 + exporter_year + importer_year',
                      data=cg_clean, vcov={'CRV1': 'exporter_importer_hs6'})
        b, se, p = m.coef()['sanc_post'], m.se()['sanc_post'], m.pvalue()['sanc_post']
        pct = (np.exp(b) - 1) * 100
        prt(f'  β={b:.4f} ({se:.4f}){stars(p)}, %={pct:.1f}%, N={get_nobs(m):,} ({time.time()-t0:.0f}s)')
        results.append({'spec': 'Coalition→Gateway (sanc×post)', 'beta': b, 'se': se, 'pval': p, 'pct': pct, 'nobs': get_nobs(m)})
    except Exception as e:
        prt(f'  ERROR: {e}')

    # Spec 2: Gateway — staggered
    prt('\n--- Spec 2: Coalition→Gateway (staggered) ---')
    try:
        t0 = time.time()
        m = pf.feols('ln_v ~ hs6_sanctioned_at_t | exporter_importer_hs6 + exporter_year + importer_year',
                      data=cg_clean, vcov={'CRV1': 'exporter_importer_hs6'})
        b, se, p = m.coef()['hs6_sanctioned_at_t'], m.se()['hs6_sanctioned_at_t'], m.pvalue()['hs6_sanctioned_at_t']
        pct = (np.exp(b) - 1) * 100
        prt(f'  β={b:.4f} ({se:.4f}){stars(p)}, %={pct:.1f}%, N={get_nobs(m):,} ({time.time()-t0:.0f}s)')
        results.append({'spec': 'Coalition→Gateway (staggered)', 'beta': b, 'se': se, 'pval': p, 'pct': pct, 'nobs': get_nobs(m)})
    except Exception as e:
        prt(f'  ERROR: {e}')

    # Spec 3: Non-Gateway placebo — sanc×post
    prt('\n--- Spec 3: Coalition→Non-Gateway PLACEBO (sanc×post) ---')
    grp = cng.groupby('exporter_importer_hs6')['t'].nunique()
    multi = grp[grp > 1].index
    cng_clean = cng[cng['exporter_importer_hs6'].isin(multi)].copy()
    try:
        t0 = time.time()
        m = pf.feols('ln_v ~ sanc_post | exporter_importer_hs6 + exporter_year + importer_year',
                      data=cng_clean, vcov={'CRV1': 'exporter_importer_hs6'})
        b, se, p = m.coef()['sanc_post'], m.se()['sanc_post'], m.pvalue()['sanc_post']
        pct = (np.exp(b) - 1) * 100
        prt(f'  β={b:.4f} ({se:.4f}){stars(p)}, %={pct:.1f}%, N={get_nobs(m):,} ({time.time()-t0:.0f}s)')
        results.append({'spec': 'Coalition→Non-Gateway PLACEBO', 'beta': b, 'se': se, 'pval': p, 'pct': pct, 'nobs': get_nobs(m)})
    except Exception as e:
        prt(f'  ERROR: {e}')

    # Spec 4: Pooled with Gateway interaction + hs6_year FE
    # Here sanc_post_gw = sanc_post × is_gateway has cross-sectional variation
    # (gateway vs non-gateway) within hs6_year cells
    prt('\n--- Spec 4: Pooled (Gateway × sanc×post, with hs6_year FE) ---')
    pooled = pd.concat([cg, cng], ignore_index=True)
    pooled['sanc_post_gw'] = (pooled['sanc_post'] * pooled['is_gateway']).astype(int)
    pooled['exporter_year'] = pooled['exporter'].astype(str) + '_' + pooled['t'].astype(str)

    grp = pooled.groupby('exporter_importer_hs6')['t'].nunique()
    multi = grp[grp > 1].index
    pooled_clean = pooled[pooled['exporter_importer_hs6'].isin(multi)].copy()

    try:
        t0 = time.time()
        m = pf.feols('ln_v ~ sanc_post_gw | exporter_importer_hs6 + hs6_year + exporter_year',
                      data=pooled_clean, vcov={'CRV1': 'exporter_importer_hs6'})
        b_gw = m.coef()['sanc_post_gw']
        se_gw = m.se()['sanc_post_gw']
        p_gw = m.pvalue()['sanc_post_gw']
        pct_gw = (np.exp(b_gw) - 1) * 100
        prt(f'  sanc_post×Gateway:    β={b_gw:.4f} ({se_gw:.4f}){stars(p_gw)}, %={pct_gw:.1f}%')
        prt(f'  N={get_nobs(m):,} ({time.time()-t0:.0f}s)')
        results.append({'spec': 'Pooled: Gateway×sanc_post (hs6_year FE)', 'beta': b_gw, 'se': se_gw, 'pval': p_gw, 'pct': pct_gw, 'nobs': get_nobs(m)})
    except Exception as e:
        prt(f'  ERROR: {e}')

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'b7_coalition_gateway.csv', index=False)
    prt(f'\nSaved: {RESULTS_DIR}/b7_coalition_gateway.csv')

    # ===== Interpretation =====
    prt('\n' + '='*60)
    prt('INTERPRETATION')
    prt('='*60)

    if len(results) >= 1:
        gw = results[0]
        if gw['pval'] < 0.05 and gw['beta'] > 0:
            prt('✓ Coalition INCREASED sanctioned-product exports to gateways')
            prt('  → TRANSSHIPMENT EVIDENCE: goods flow Coalition→Gateway→Russia')
            prt('  → Strengthens rerouting = circumvention interpretation')
        elif gw['pval'] < 0.05 and gw['beta'] < 0:
            prt('✗ Coalition DECREASED sanctioned-product exports to gateways')
            prt('  → Sanctions compliance even toward gateways')
            prt('  → Rerouting more likely = demand substitution')
        else:
            prt('? No significant change in coalition→gateway sanctioned exports')

    if len(results) >= 3:
        gw_b = results[0]['beta']
        ngw_b = results[2]['beta']
        prt(f'\n  Gateway β={gw_b:.4f} vs Non-Gateway β={ngw_b:.4f}')
        diff = gw_b - ngw_b
        prt(f'  Difference: {diff:.4f}')
        if diff > 0 and results[0]['pval'] < 0.05:
            prt('  → Gateway-specific increase supports transshipment')


if __name__ == '__main__':
    main()
