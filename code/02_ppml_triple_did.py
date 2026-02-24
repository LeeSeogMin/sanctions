#!/usr/bin/env python3
"""
02_ppml_triple_did.py
PPML Triple-DID: direct effect of sanctions on bilateral trade

Strategy:
1. OLS ln(v+1): full sample (1M rows, 157K FE)
2. PPML value: top-50 partners (95.6% of trade)
3. Event study: OLS-based staggered DiD
4. Category heterogeneity: OLS-based
"""

import pandas as pd
import numpy as np
import pyfixest as pf
from pathlib import Path
import time
import sys
import warnings
warnings.filterwarnings('ignore')

PANEL_PATH = Path('data/processed/baci_russia_panel.parquet')
RESULTS_DIR = Path('output/results')
FIGURES_DIR = Path('output/figures')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def prt(msg):
    print(msg, flush=True)


def get_nobs(m):
    """pyfixest 모델에서 관측치 수 추출"""
    try:
        return m.nobs
    except:
        try:
            return m._N
        except:
            try:
                return len(m._Y)
            except:
                return -1


def load_data():
    """패널 로딩 + 전처리"""
    prt('=== Loading Panel ===')
    panel = pd.read_parquet(PANEL_PATH)
    imp = panel[panel['direction'] == 'to_russia'].copy()
    prt(f'Full sample: {len(imp):,} rows')

    # 싱글턴 제거
    grp = imp.groupby('partner_hs6')['t'].nunique()
    multi = grp[grp > 1].index
    imp = imp[imp['partner_hs6'].isin(multi)].copy()
    prt(f'After singleton removal: {len(imp):,} rows')

    # 변수 생성
    imp['ln_v'] = np.log(imp['v'] + 1)
    imp['unit_value'] = np.where((imp['q'] > 0) & imp['q'].notna(), imp['v'] / imp['q'], np.nan)

    # EU-only coalition
    EU27 = {40, 56, 100, 191, 196, 203, 208, 233, 246, 251, 276, 300,
            348, 372, 380, 428, 440, 442, 470, 528, 616, 620, 642,
            703, 705, 724, 752}
    imp['coalition_eu'] = imp['partner'].isin(EU27).astype(int)
    imp['treated_eu'] = imp['hs6_sanctioned_at_t'] * imp['coalition_eu']

    # Event time
    imp['event_time'] = np.nan
    mask = imp['sanctioned'] == 1
    imp.loc[mask, 'event_time'] = imp.loc[mask, 't'] - imp.loc[mask, 'sanction_year']
    imp['et_bin'] = imp['event_time'].copy()
    imp.loc[imp['et_bin'] < -5, 'et_bin'] = -5
    imp.loc[imp['et_bin'] > 2, 'et_bin'] = 2

    # Top-50 파트너 서브셋 (PPML용)
    top50 = imp.groupby('partner')['v'].sum().nlargest(50).index
    imp_top = imp[imp['partner'].isin(top50)].copy()
    prt(f'Top-50 partner subset: {len(imp_top):,} rows ({len(imp_top)/len(imp)*100:.1f}%)')
    prt(f'Top-50 partner_hs6 FE: {imp_top["partner_hs6"].nunique():,}')

    return imp, imp_top


def run_ols_battery(imp):
    """OLS 배터리 — 전체 표본"""
    prt('\n' + '='*60)
    prt('OLS SPECIFICATIONS (full sample)')
    prt('='*60)

    results = []
    specs = [
        ('OLS | Full FE | T1 | value',
         'ln_v ~ treated_t1 | partner_hs6 + hs6_year + partner_year'),
        ('OLS | Reduced FE | T1 | value',
         'ln_v ~ treated_t1 + coalition_tier1 | partner_hs6 + hs6_year'),
        ('OLS | Full FE | T2 | value',
         'ln_v ~ treated_t2 | partner_hs6 + hs6_year + partner_year'),
        ('OLS | Full FE | EU | value',
         'ln_v ~ treated_eu | partner_hs6 + hs6_year + partner_year'),
    ]

    for label, formula in specs:
        t0 = time.time()
        prt(f'\n--- {label} ---')
        try:
            m = pf.feols(formula, data=imp, vcov={'CRV1': 'partner_hs6'})
            coef_name = [k for k in m.coef().keys() if 'treated' in k][0]
            beta = m.coef()[coef_name]
            se = m.se()[coef_name]
            pval = m.pvalue()[coef_name]
            pct = (np.exp(beta) - 1) * 100
            nobs = get_nobs(m)
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            prt(f'  β = {beta:.4f} ({se:.4f}){sig}, % = {pct:.1f}%, N = {nobs:,} ({time.time()-t0:.0f}s)')
            results.append({
                'label': label, 'method': 'OLS', 'beta': beta, 'se': se,
                'pval': pval, 'pct_effect': pct, 'nobs': nobs
            })
        except Exception as e:
            prt(f'  ERROR: {e}')

    # Unit value
    imp_uv = imp[imp['unit_value'].notna() & (imp['unit_value'] > 0)].copy()
    imp_uv['ln_uv'] = np.log(imp_uv['unit_value'])
    prt(f'\n--- OLS | Full FE | T1 | unit_value ---')
    try:
        t0 = time.time()
        m = pf.feols('ln_uv ~ treated_t1 | partner_hs6 + hs6_year + partner_year',
                     data=imp_uv, vcov={'CRV1': 'partner_hs6'})
        beta = m.coef()['treated_t1']
        se = m.se()['treated_t1']
        pval = m.pvalue()['treated_t1']
        pct = (np.exp(beta) - 1) * 100
        nobs = get_nobs(m)
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        prt(f'  β = {beta:.4f} ({se:.4f}){sig}, % = {pct:.1f}%, N = {nobs:,} ({time.time()-t0:.0f}s)')
        results.append({
            'label': 'OLS | Full FE | T1 | unit_value', 'method': 'OLS',
            'beta': beta, 'se': se, 'pval': pval, 'pct_effect': pct, 'nobs': nobs
        })
    except Exception as e:
        prt(f'  ERROR: {e}')

    # Quantity OLS
    imp_q = imp[imp['q'] > 0].copy()
    imp_q['ln_q'] = np.log(imp_q['q'])
    prt(f'\n--- OLS | Full FE | T1 | quantity ---')
    try:
        t0 = time.time()
        m = pf.feols('ln_q ~ treated_t1 | partner_hs6 + hs6_year + partner_year',
                     data=imp_q, vcov={'CRV1': 'partner_hs6'})
        beta = m.coef()['treated_t1']
        se = m.se()['treated_t1']
        pval = m.pvalue()['treated_t1']
        pct = (np.exp(beta) - 1) * 100
        nobs = get_nobs(m)
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        prt(f'  β = {beta:.4f} ({se:.4f}){sig}, % = {pct:.1f}%, N = {nobs:,} ({time.time()-t0:.0f}s)')
        results.append({
            'label': 'OLS | Full FE | T1 | quantity', 'method': 'OLS',
            'beta': beta, 'se': se, 'pval': pval, 'pct_effect': pct, 'nobs': nobs
        })
    except Exception as e:
        prt(f'  ERROR: {e}')

    return results


def run_ppml_top50(imp_top):
    """PPML — Top-50 파트너 서브셋"""
    prt('\n' + '='*60)
    prt('PPML SPECIFICATIONS (Top-50 partners)')
    prt('='*60)

    results = []

    # 다시 싱글턴 제거 (서브셋에서)
    grp = imp_top.groupby('partner_hs6')['t'].nunique()
    multi = grp[grp > 1].index
    imp_top = imp_top[imp_top['partner_hs6'].isin(multi)].copy()
    prt(f'Top-50 after singletons: {len(imp_top):,} rows, {imp_top["partner_hs6"].nunique():,} FE groups')

    specs = [
        ('PPML | Full FE | T1 | value (Top50)',
         'v ~ treated_t1 | partner_hs6 + hs6_year + partner_year', 'treated_t1'),
        ('PPML | Full FE | T2 | value (Top50)',
         'v ~ treated_t2 | partner_hs6 + hs6_year + partner_year', 'treated_t2'),
        ('PPML | Full FE | EU | value (Top50)',
         'v ~ treated_eu | partner_hs6 + hs6_year + partner_year', 'treated_eu'),
    ]

    for label, formula, coef_name in specs:
        t0 = time.time()
        prt(f'\n--- {label} ---')
        try:
            m = pf.fepois(formula, data=imp_top, vcov={'CRV1': 'partner_hs6'})
            beta = m.coef()[coef_name]
            se = m.se()[coef_name]
            pval = m.pvalue()[coef_name]
            pct = (np.exp(beta) - 1) * 100
            nobs = get_nobs(m)
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            prt(f'  β = {beta:.4f} ({se:.4f}){sig}, % = {pct:.1f}%, N = {nobs:,} ({time.time()-t0:.0f}s)')
            results.append({
                'label': label, 'method': 'PPML', 'beta': beta, 'se': se,
                'pval': pval, 'pct_effect': pct, 'nobs': nobs
            })
        except Exception as e:
            prt(f'  ERROR: {e} ({time.time()-t0:.0f}s)')
            results.append({'label': label, 'method': 'PPML', 'error': str(e)})

    # PPML Quantity (Top50)
    imp_q = imp_top[imp_top['q'] > 0].copy()
    prt(f'\n--- PPML | Full FE | T1 | quantity (Top50) ---')
    try:
        t0 = time.time()
        m = pf.fepois('q ~ treated_t1 | partner_hs6 + hs6_year + partner_year',
                      data=imp_q, vcov={'CRV1': 'partner_hs6'})
        beta = m.coef()['treated_t1']
        se = m.se()['treated_t1']
        pval = m.pvalue()['treated_t1']
        pct = (np.exp(beta) - 1) * 100
        nobs = get_nobs(m)
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        prt(f'  β = {beta:.4f} ({se:.4f}){sig}, % = {pct:.1f}%, N = {nobs:,} ({time.time()-t0:.0f}s)')
        results.append({
            'label': 'PPML | Full FE | T1 | quantity (Top50)', 'method': 'PPML',
            'beta': beta, 'se': se, 'pval': pval, 'pct_effect': pct, 'nobs': nobs
        })
    except Exception as e:
        prt(f'  ERROR: {e}')

    return results


def run_event_study(imp):
    """OLS Event Study — 전체 표본"""
    prt('\n' + '='*60)
    prt('EVENT STUDY (OLS, full sample)')
    prt('='*60)

    imp_es = imp.copy()
    imp_es['et_bin'] = imp_es['et_bin'].fillna(-99)

    event_times = sorted([et for et in imp_es['et_bin'].unique()
                          if et != -99 and et != -1])
    prt(f'Event times (excl. ref=-1): {event_times}')

    for et in event_times:
        col = f'et_{int(et)}_x_coal'
        imp_es[col] = ((imp_es['et_bin'] == et) &
                       (imp_es['coalition_tier1'] == 1)).astype(int)

    et_vars = ' + '.join([f'et_{int(et)}_x_coal' for et in event_times])
    imp_es['ln_v'] = np.log(imp_es['v'] + 1)
    formula = f'ln_v ~ {et_vars} | partner_hs6 + hs6_year + partner_year'

    try:
        t0 = time.time()
        m = pf.feols(formula, data=imp_es, vcov={'CRV1': 'partner_hs6'})
        prt(f'  Completed in {time.time()-t0:.1f}s')

        es_results = []
        for et in event_times:
            col = f'et_{int(et)}_x_coal'
            beta = m.coef()[col]
            se = m.se()[col]
            pval = m.pvalue()[col]
            pct = (np.exp(beta) - 1) * 100
            es_results.append({
                'event_time': int(et), 'beta': beta, 'se': se,
                'pval': pval, 'pct_effect': pct,
                'ci_lower': beta - 1.96 * se, 'ci_upper': beta + 1.96 * se
            })
        es_results.append({
            'event_time': -1, 'beta': 0, 'se': 0, 'pval': 1.0,
            'pct_effect': 0, 'ci_lower': 0, 'ci_upper': 0
        })

        es_df = pd.DataFrame(es_results).sort_values('event_time')

        prt('\nEvent Study Coefficients:')
        for _, row in es_df.iterrows():
            sig = '***' if row['pval'] < 0.001 else '**' if row['pval'] < 0.01 else '*' if row['pval'] < 0.05 else ''
            prt(f'  t={int(row["event_time"]):+d}: β={row["beta"]:.4f} ({row["se"]:.4f}){sig} [{row["pct_effect"]:.1f}%]')

        # Pre-trend
        pre = es_df[es_df['event_time'] < -1]
        if len(pre) > 0:
            min_p = pre['pval'].min()
            any_sig = (pre['pval'] < 0.05).any()
            prt(f'\nPre-trend: min p = {min_p:.4f}, any sig (p<0.05): {any_sig}')

        return es_df

    except Exception as e:
        prt(f'  ERROR: {e}')
        return pd.DataFrame()


def run_categories(imp):
    """카테고리별 이질성 (OLS)"""
    prt('\n' + '='*60)
    prt('CATEGORY HETEROGENEITY (OLS)')
    prt('='*60)

    results = []
    categories = ['dual_use', 'industrial_cap', 'luxury', 'military_tech']

    for cat in categories:
        sub = imp[(imp['category'] == cat) |
                  (imp['category'].isin(['non_sanctioned', 'not_in_eu_list']))].copy()
        sub['treated_cat'] = ((sub['category'] == cat) &
                              (sub['hs6_sanctioned_at_t'] == 1) &
                              (sub['coalition_tier1'] == 1)).astype(int)
        sub['ln_v'] = np.log(sub['v'] + 1)

        try:
            t0 = time.time()
            m = pf.feols('ln_v ~ treated_cat | partner_hs6 + hs6_year + partner_year',
                        data=sub, vcov={'CRV1': 'partner_hs6'})
            beta = m.coef()['treated_cat']
            se = m.se()['treated_cat']
            pval = m.pvalue()['treated_cat']
            pct = (np.exp(beta) - 1) * 100
            nobs = get_nobs(m)
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            prt(f'  {cat:20s}: β={beta:.4f} ({se:.4f}){sig}, %={pct:.1f}%, N={nobs:,} ({time.time()-t0:.0f}s)')
            results.append({
                'category': cat, 'beta': beta, 'se': se,
                'pval': pval, 'pct_effect': pct, 'nobs': nobs
            })
        except Exception as e:
            prt(f'  {cat}: ERROR: {e}')

    return results


def plot_event_study(es_df):
    """Event Study 시각화"""
    import matplotlib.pyplot as plt
    if es_df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: β coefficients
    ax1.errorbar(
        es_df['event_time'], es_df['beta'],
        yerr=1.96 * es_df['se'],
        fmt='o-', color='#2166ac', capsize=4, capthick=1.5,
        markersize=7, linewidth=2
    )
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(-0.5, color='red', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Event Time (years)', fontsize=12)
    ax1.set_ylabel(r'$\beta$ (OLS coefficient)', fontsize=12)
    ax1.set_title('A. Coefficient Plot', fontsize=13)

    # Right: % effect
    ax2.bar(es_df['event_time'], es_df['pct_effect'],
            color=np.where(es_df['event_time'] >= 0, '#b2182b', '#4393c3'),
            alpha=0.7, edgecolor='white')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(-0.5, color='red', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Event Time (years)', fontsize=12)
    ax2.set_ylabel('% Effect', fontsize=12)
    ax2.set_title('B. Percentage Effect', fontsize=13)

    fig.suptitle('Staggered Event Study: Coalition × Sanctioned HS6\n'
                 'OLS Triple-DID with partner-HS6, HS6-year, partner-year FE',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'b3_event_study.png', dpi=150, bbox_inches='tight')
    prt(f'Saved: {FIGURES_DIR}/b3_event_study.png')
    plt.close()


if __name__ == '__main__':
    t_start = time.time()

    # Load
    imp, imp_top = load_data()

    # OLS (전체 표본, ~35초/spec)
    ols_results = run_ols_battery(imp)

    # Event Study (OLS, ~35초)
    es_df = run_event_study(imp)
    if not es_df.empty:
        es_df.to_csv(RESULTS_DIR / 'b3_event_study.csv', index=False)
        plot_event_study(es_df)

    # Category (OLS, ~30초/cat)
    cat_results = run_categories(imp)

    # PPML (Top-50 파트너)
    ppml_results = run_ppml_top50(imp_top)

    # ── Summary ──
    prt('\n' + '='*70)
    prt('B-3 RESULTS SUMMARY')
    prt('='*70)

    all_results = ols_results + ppml_results
    valid = [r for r in all_results if 'error' not in r]
    prt(f'\n--- Main Results ({len(valid)} specs) ---')
    for r in valid:
        sig = '***' if r['pval'] < 0.001 else '**' if r['pval'] < 0.01 else '*' if r['pval'] < 0.05 else ''
        prt(f"  {r['label']:45s}: β={r['beta']:.4f} ({r['se']:.4f}){sig}  → {r['pct_effect']:.1f}%")

    if cat_results:
        prt('\n--- Category Heterogeneity ---')
        for r in cat_results:
            sig = '***' if r['pval'] < 0.001 else '**' if r['pval'] < 0.01 else '*' if r['pval'] < 0.05 else ''
            prt(f"  {r['category']:20s}: β={r['beta']:.4f} ({r['se']:.4f}){sig}  → {r['pct_effect']:.1f}%")

    # Save
    pd.DataFrame(valid).to_csv(RESULTS_DIR / 'b3_ppml_main.csv', index=False)
    if cat_results:
        pd.DataFrame(cat_results).to_csv(RESULTS_DIR / 'b3_category_heterogeneity.csv', index=False)

    errors = [r for r in all_results if 'error' in r]
    if errors:
        prt(f'\n--- Failed ({len(errors)}) ---')
        for r in errors:
            prt(f"  {r['label']}: {r.get('error','?')[:80]}")

    prt(f'\n=== Total: {time.time()-t_start:.1f}s ===')
