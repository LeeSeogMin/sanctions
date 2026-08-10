"""Exporter-specific treatment defined from jurisdiction-level sanctions lists.

Supplies Online Appendix L. The baseline assigns the EU product list to every
coalition member. This script instead treats exporter i in product k at year t
only if i's own jurisdiction had sanctioned k by t, using the nine
jurisdiction-specific lists in OpenICPSR 230342.

Design
  jurisdictions  EU-27, GB, US, CA, JP, KR, CH, AU, TW
                 (CH is Switzerland, not China; China stays untreated)
  coalition members without a list (e.g. Norway) are coded untreated
  hierarchical codes are prefix-expanded as in 18_ekmn_concordance.py
  annual timing: treated once year(implementation) <= t
  sample, fixed effects and clustering match the baseline

Specifications, top-50 partner sample for direct comparability
  S1  baseline treatment (reproduction gate on the identical sample)
  S2  jurisdiction-specific treatment
  S3  both terms together, exploratory

Output: results/b19_ekmn_exporter_ppml.csv
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings('ignore')

PANEL_PATH = Path('phase4_analysis/processed/baci_russia_panel.parquet')
EKMN_PATH = Path('phase3_data/raw/EKMN_230342/data_sa_20241230.dta')
CC_PATH = Path('phase3_data/raw/country_codes.csv')
OUT = Path('phase4_analysis/results/b19_ekmn_exporter_ppml.csv')

EU27 = ['AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA',
        'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD',
        'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE']
JURIS_ISO3 = {'EU': EU27, 'GB': ['GBR'], 'US': ['USA'], 'CA': ['CAN'],
              'JP': ['JPN'], 'KR': ['KOR'], 'CH': ['CHE'], 'AU': ['AUS']}
TW_BACI_CODE = 490  # BACI 'Other Asia, nes' = 대만 (country_codes.csv 외 — 직접 지정)


def prt(msg):
    print(msg, flush=True)


def build_treatment_lookup():
    """관할권 → {hs6: 최초 제재 연도} (prefix 확장)"""
    ekmn = pd.read_stata(EKMN_PATH)
    ekmn['prefix'] = ekmn.apply(
        lambda r: str(r['code_6']) if r['len'] >= 6 else str(r['code']), axis=1)
    ekmn['year'] = ekmn['date_s'].dt.year

    panel_hs6 = pd.read_parquet(PANEL_PATH, columns=['hs6'])['hs6'].astype(str).str.zfill(6).unique()

    lookup = {}  # juris -> {hs6: first_year}
    for juris, g in ekmn.groupby('country2'):
        pref_year = g.groupby('prefix')['year'].min()
        by_len = {L: {p: y for p, y in pref_year.items() if len(p) == L} for L in (2, 4, 6)}
        d = {}
        for h in panel_hs6:
            ys = [by_len[L][h[:L]] for L in (2, 4, 6) if h[:L] in by_len[L]]
            if ys:
                d[h] = min(ys)
        lookup[juris] = d
        prt(f'  {juris}: {len(d):,} HS6 (유니버스 내)')
    return lookup


def load_panel(lookup):
    prt('=== Loading Panel ===')
    panel = pd.read_parquet(PANEL_PATH)
    imp = panel[panel['direction'] == 'to_russia'].copy()
    imp['hs6'] = imp['hs6'].astype(str).str.zfill(6)

    # 싱글턴 제거 (12/06b와 동일)
    grp = imp.groupby('partner_hs6')['t'].nunique()
    imp = imp[imp['partner_hs6'].isin(grp[grp > 1].index)].copy()
    prt(f'After singleton removal: {len(imp):,} rows')

    # 관할권 매핑: partner(comtrade code) → EKMN 관할권
    cc = pd.read_csv(CC_PATH)
    iso2code = cc.set_index('iso3')['comtrade_code']
    code2juris = {}
    for juris, isos in JURIS_ISO3.items():
        for iso in isos:
            code2juris[int(iso2code[iso])] = juris
    code2juris[TW_BACI_CODE] = 'TW'
    imp['ekmn_juris'] = imp['partner'].map(code2juris)
    n_map = imp['ekmn_juris'].notna().sum()
    prt(f'EKMN 관할권 매핑 행: {n_map:,} ({imp.loc[imp["ekmn_juris"].notna(), "partner"].nunique()}개국)')

    # exporter-specific 처치
    def first_year(row_juris, row_hs6):
        if pd.isna(row_juris):
            return np.nan
        return lookup.get(row_juris, {}).get(row_hs6, np.nan)

    # 벡터화: (juris, hs6) 쌍 단위 매핑
    pairs = imp[['ekmn_juris', 'hs6']].drop_duplicates()
    pairs['fy'] = [first_year(j, h) for j, h in zip(pairs['ekmn_juris'], pairs['hs6'])]
    imp = imp.merge(pairs, on=['ekmn_juris', 'hs6'], how='left')
    imp['treated_ekmn'] = ((imp['fy'].notna()) & (imp['t'] >= imp['fy'])).astype(int)
    prt(f'treated_ekmn=1: {imp["treated_ekmn"].sum():,} 행 '
        f'(기존 treated_t1=1: {imp["treated_t1"].sum():,})')
    both = ((imp['treated_ekmn'] == 1) & (imp['treated_t1'] == 1)).sum()
    prt(f'  중첩: {both:,} | EKMN-only: {(imp["treated_ekmn"]==1).sum()-both:,} | '
        f'T1-only: {(imp["treated_t1"]==1).sum()-both:,}')

    # Top-50 (12/06b와 동일)
    top50 = imp.groupby('partner')['v'].sum().nlargest(50).index
    imp_top = imp[imp['partner'].isin(top50)].copy()
    grp = imp_top.groupby('partner_hs6')['t'].nunique()
    imp_top = imp_top[imp_top['partner_hs6'].isin(grp[grp > 1].index)].copy()
    prt(f'Top-50 subset: {len(imp_top):,} rows')
    return imp_top


def run_spec(label, formula, data, results):
    prt(f'\n--- {label}: {formula} ---')
    t0 = time.time()
    m = pf.fepois(formula, data=data, vcov={'CRV1': 'partner_hs6'})
    for var in m.coef().index:
        beta, se, pv = m.coef()[var], m.se()[var], m.pvalue()[var]
        prt(f'  {var}: β={beta:.4f} (SE {se:.4f}, p={pv:.4g}) → {(np.exp(beta)-1)*100:.1f}%')
        results.append({'spec': label, 'var': var, 'beta': beta, 'se': se, 'pval': pv,
                        'pct_effect': (np.exp(beta) - 1) * 100,
                        'nobs': getattr(m, '_N', np.nan), 'elapsed_s': round(time.time() - t0, 1)})
    pd.DataFrame(results).to_csv(OUT, index=False)
    prt(f'  ({time.time()-t0:.0f}s) → saved {OUT.name}')


if __name__ == '__main__':
    t_all = time.time()
    prt('=== EKMN 처치 lookup 구축 ===')
    lookup = build_treatment_lookup()
    imp_top = load_panel(lookup)

    FE = 'partner_hs6 + hs6_year + partner_year'
    results = []
    run_spec('S1_baseline_t1', f'v ~ treated_t1 | {FE}', imp_top, results)
    run_spec('S2_ekmn_exporter', f'v ~ treated_ekmn | {FE}', imp_top, results)
    run_spec('S3_joint', f'v ~ treated_t1 + treated_ekmn | {FE}', imp_top, results)
    prt(f'\n총 소요: {(time.time()-t_all)/60:.1f}분')
