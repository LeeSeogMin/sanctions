#!/usr/bin/env python3
"""
05_robustness_checks.py
Robustness checks: balanced-panel PPML + rerouting identification tests

Analysis:
  1. Balanced panel (zero-filled) PPML to address extensive margin
  2. Rerouting identification: high-exposure DiD, supply-gap correlation, gateway test

Output:
  - output/results/b7_balanced_panel_ppml.csv
  - output/results/b7_rerouting_identification.csv
"""

import pandas as pd
import numpy as np
import pyfixest as pf
from pathlib import Path
from itertools import product as cart_product
import time
import warnings
warnings.filterwarnings('ignore')

PANEL_PATH = Path('data/processed/baci_russia_panel.parquet')
RESULTS_DIR = Path('output/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EU27 = {40, 56, 100, 191, 196, 203, 208, 233, 246, 251, 276, 300,
        348, 372, 380, 428, 440, 442, 470, 528, 616, 620, 642,
        703, 705, 724, 752}
TIER1_NON_EU = {826, 842, 124, 392, 579, 757}
TIER1 = EU27 | TIER1_NON_EU
RUSSIA = 643

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


# =============================================================
# B-7.1: Balanced Panel with Zeros → PPML
# =============================================================
def b71_balanced_panel_ppml():
    """
    Address BACI extensive margin concern:
    BACI reports only positive flows → if sanctions eliminate a flow,
    it disappears rather than becoming zero → truncation bias.

    Solution: Construct balanced panel for all (partner, HS6) pairs
    ever observed, fill missing years with v=0, re-estimate PPML.
    Compare with unbalanced result.
    """
    prt('\n' + '='*60)
    prt('B-7.1: Balanced Panel PPML (Extensive Margin Test)')
    prt('='*60)

    panel = pd.read_parquet(PANEL_PATH)
    to_rus = panel[panel['direction'] == 'to_russia'].copy()
    prt(f'Original to_russia: {len(to_rus):,} rows')

    # --- Step 1: Identify top-50 partners ---
    partner_rank = to_rus.groupby('partner')['v'].sum().sort_values(ascending=False)
    top50 = set(partner_rank.head(50).index)
    to_rus_50 = to_rus[to_rus['partner'].isin(top50)].copy()
    prt(f'Top-50 subset: {len(to_rus_50):,} rows')

    # --- Step 2: Get all unique (partner, HS6) pairs ever observed ---
    pair_keys = to_rus_50[['partner', 'hs6']].drop_duplicates()
    prt(f'Unique (partner, HS6) pairs: {len(pair_keys):,}')

    # --- Step 3: Create full grid: partner × HS6 × year ---
    years = sorted(to_rus_50['t'].unique())
    prt(f'Years: {years[0]}-{years[-1]} ({len(years)} years)')

    # Efficient balanced panel construction
    prt('Constructing balanced panel...')
    t0 = time.time()
    pair_list = pair_keys.values.tolist()  # list of [partner, hs6]

    # Create full grid using cross join
    grid = pair_keys.assign(key=1).merge(
        pd.DataFrame({'t': years, 'key': 1}), on='key'
    ).drop('key', axis=1)
    prt(f'Full grid: {len(grid):,} rows ({time.time()-t0:.1f}s)')

    # --- Step 4: Merge with observed data ---
    prt('Merging with observed flows...')
    t0 = time.time()
    balanced = grid.merge(
        to_rus_50[['partner', 'hs6', 't', 'v', 'q']],
        on=['partner', 'hs6', 't'],
        how='left'
    )
    balanced['v'] = balanced['v'].fillna(0)
    balanced['q'] = balanced['q'].fillna(0)

    n_zeros = (balanced['v'] == 0).sum()
    prt(f'Balanced panel: {len(balanced):,} rows, zeros: {n_zeros:,} ({n_zeros/len(balanced)*100:.1f}%)')
    prt(f'Merge time: {time.time()-t0:.1f}s')

    # --- Step 5: Attach treatment variables ---
    prt('Attaching treatment variables...')

    # Load sanctions data
    sanctions = pd.read_csv('data/processed/sanctions_hs6_master.csv')
    sanctions['hs6'] = sanctions['hs6'].astype(str).str.zfill(6)

    # sanctioned: time-invariant cohort indicator
    sanctioned_codes = set(sanctions[sanctions['sanctioned'] == 1]['hs6'].values)
    balanced['sanctioned'] = balanced['hs6'].isin(sanctioned_codes).astype(int)

    # sanction_year: when the code was first sanctioned
    sanc_year_map = sanctions[sanctions['sanctioned'] == 1].groupby('hs6')['sanction_year'].min().to_dict()
    balanced['sanction_year'] = balanced['hs6'].map(sanc_year_map)

    # hs6_sanctioned_at_t: time-varying treatment (1 if sanctioned at time t)
    balanced['hs6_sanctioned_at_t'] = (
        (balanced['sanctioned'] == 1) &
        (balanced['t'] >= balanced['sanction_year'].fillna(9999))
    ).astype(int)

    # coalition
    balanced['coalition_tier1'] = balanced['partner'].isin(TIER1).astype(int)

    # treated_t1: triple-DID treatment
    balanced['treated_t1'] = (balanced['hs6_sanctioned_at_t'] * balanced['coalition_tier1']).astype(int)

    # FE variables
    balanced['partner_hs6'] = balanced['partner'].astype(str) + '_' + balanced['hs6']
    balanced['hs6_year'] = balanced['hs6'] + '_' + balanced['t'].astype(str)
    balanced['partner_year'] = balanced['partner'].astype(str) + '_' + balanced['t'].astype(str)

    # --- Step 6: Singleton removal ---
    grp = balanced.groupby('partner_hs6')['t'].nunique()
    multi = grp[grp > 1].index
    balanced = balanced[balanced['partner_hs6'].isin(multi)].copy()
    prt(f'After singleton removal: {len(balanced):,} rows')

    results = []

    # --- Step 7A: PPML on balanced panel (with zeros) ---
    prt('\n--- PPML on Balanced Panel (with zeros) ---')
    try:
        t0 = time.time()
        m = pf.fepois('v ~ treated_t1 | partner_hs6 + hs6_year + partner_year',
                       data=balanced, vcov={'CRV1': 'partner_hs6'})
        b, se, p = m.coef()['treated_t1'], m.se()['treated_t1'], m.pvalue()['treated_t1']
        pct = (np.exp(b) - 1) * 100
        elapsed = time.time() - t0
        prt(f'  β={b:.4f} ({se:.4f}){stars(p)}, %={pct:.1f}%, N={get_nobs(m):,} ({elapsed:.0f}s)')
        results.append({
            'spec': 'PPML Balanced (with zeros)', 'sample': 'Top-50',
            'beta': b, 'se': se, 'pval': p, 'pct': pct,
            'nobs': get_nobs(m), 'n_zeros': n_zeros, 'zero_pct': n_zeros/len(balanced)*100,
            'time_s': elapsed
        })
    except Exception as e:
        prt(f'  ERROR: {e}')
        results.append({
            'spec': 'PPML Balanced (with zeros)', 'sample': 'Top-50',
            'error': str(e)
        })

    # --- Step 7B: PPML on original unbalanced panel (comparison) ---
    prt('\n--- PPML on Original Unbalanced Panel (comparison) ---')
    # Use the original top-50 data with singleton removal
    orig = to_rus_50.copy()
    grp_orig = orig.groupby('partner_hs6')['t'].nunique()
    multi_orig = grp_orig[grp_orig > 1].index
    orig = orig[orig['partner_hs6'].isin(multi_orig)].copy()
    try:
        t0 = time.time()
        m = pf.fepois('v ~ treated_t1 | partner_hs6 + hs6_year + partner_year',
                       data=orig, vcov={'CRV1': 'partner_hs6'})
        b, se, p = m.coef()['treated_t1'], m.se()['treated_t1'], m.pvalue()['treated_t1']
        pct = (np.exp(b) - 1) * 100
        elapsed = time.time() - t0
        prt(f'  β={b:.4f} ({se:.4f}){stars(p)}, %={pct:.1f}%, N={get_nobs(m):,} ({elapsed:.0f}s)')
        results.append({
            'spec': 'PPML Unbalanced (original)', 'sample': 'Top-50',
            'beta': b, 'se': se, 'pval': p, 'pct': pct,
            'nobs': get_nobs(m), 'n_zeros': 0, 'zero_pct': 0,
            'time_s': elapsed
        })
    except Exception as e:
        prt(f'  ERROR: {e}')
        results.append({
            'spec': 'PPML Unbalanced (original)', 'sample': 'Top-50',
            'error': str(e)
        })

    # --- Step 7C: OLS ln(v+1) on balanced panel (for comparison) ---
    prt('\n--- OLS ln(v+1) on Balanced Panel ---')
    balanced['ln_v'] = np.log(balanced['v'] + 1)
    try:
        t0 = time.time()
        m = pf.feols('ln_v ~ treated_t1 | partner_hs6 + hs6_year + partner_year',
                       data=balanced, vcov={'CRV1': 'partner_hs6'})
        b, se, p = m.coef()['treated_t1'], m.se()['treated_t1'], m.pvalue()['treated_t1']
        pct = (np.exp(b) - 1) * 100
        elapsed = time.time() - t0
        prt(f'  β={b:.4f} ({se:.4f}){stars(p)}, %={pct:.1f}%, N={get_nobs(m):,} ({elapsed:.0f}s)')
        results.append({
            'spec': 'OLS Balanced (with zeros)', 'sample': 'Top-50',
            'beta': b, 'se': se, 'pval': p, 'pct': pct,
            'nobs': get_nobs(m), 'n_zeros': n_zeros, 'zero_pct': n_zeros/len(balanced)*100,
            'time_s': elapsed
        })
    except Exception as e:
        prt(f'  ERROR: {e}')

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'b7_balanced_panel_ppml.csv', index=False)
    prt(f'\nSaved: {RESULTS_DIR}/b7_balanced_panel_ppml.csv')

    # --- Summary ---
    prt('\n--- Extensive Margin Assessment ---')
    for _, row in df.iterrows():
        if 'error' not in row or pd.isna(row.get('error', np.nan)):
            prt(f"  {row['spec']:40s}: β={row['beta']:.4f}, %={row['pct']:.1f}%")

    return df


# =============================================================
# B-7.2: Rerouting Identification — Circumvention vs Demand Substitution
# =============================================================
def b72_rerouting_identification():
    """
    Address rerouting ≠ circumvention concern with two tests:

    Test 1: Product-level supply gap correlation
      - For each sanctioned HS6: compute (a) coalition reduction, (b) gateway increase
      - If positive correlation → circumvention (gateways fill coalition's supply gap)
      - If no correlation → demand substitution (Russia buys what it needs, regardless of gap)

    Test 2: Exposure-interaction triple-DID
      - Rerouting × high_exposure interaction → if circumvention concentrates
        in high-exposure countries (which have trade infrastructure for transshipment)

    Test 3: Non-sanctioned placebo for gateways
      - Do gateways also increase non-sanctioned product exports to Russia?
      - If yes → general demand shift, not circumvention-specific
      - If no → circumvention is product-specific, supporting our interpretation
    """
    prt('\n' + '='*60)
    prt('B-7.2: Rerouting Identification Tests')
    prt('='*60)

    panel = pd.read_parquet(PANEL_PATH)
    panel['is_coalition'] = panel['partner'].isin(TIER1).astype(int)
    panel['is_third'] = ((~panel['partner'].isin(TIER1)) & (panel['partner'] != RUSSIA)).astype(int)
    panel['post'] = (panel['t'] >= 2022).astype(int)
    panel['ln_v'] = np.log(panel['v'] + 1)
    panel['hs2'] = panel['hs6'].str[:2]
    panel['hs2_year'] = panel['hs2'] + '_' + panel['t'].astype(str)

    to_rus = panel[panel['direction'] == 'to_russia'].copy()

    # Load exposure index
    exp_df = pd.read_csv(RESULTS_DIR / 'b4_exposure_index.csv')
    exp_map = dict(zip(exp_df['partner'], exp_df['exposure_index']))
    to_rus['exposure'] = to_rus['partner'].map(exp_map).fillna(0)

    # Define gateway countries (top-5 by exposure)
    gateways = set(exp_df.sort_values('exposure_index', ascending=False).head(10)['partner'].values)
    to_rus['is_gateway'] = to_rus['partner'].isin(gateways).astype(int)

    results = []

    # =========================================================
    # Test 1: Product-Level Supply Gap Correlation
    # =========================================================
    prt('\n--- Test 1: Product-Level Supply Gap Correlation ---')
    prt('For each sanctioned HS6: coalition reduction vs gateway increase')

    sanctioned_codes = to_rus[to_rus['sanctioned'] == 1]['hs6'].unique()
    prt(f'Sanctioned HS6 codes in panel: {len(sanctioned_codes)}')

    gap_data = []
    for hs6 in sanctioned_codes:
        hs6_data = to_rus[to_rus['hs6'] == hs6]

        # Coalition trade
        coal = hs6_data[hs6_data['is_coalition'] == 1]
        coal_pre = coal[coal['t'] < 2022]['v'].sum()
        coal_post = coal[coal['t'] >= 2022]['v'].sum()
        n_pre = coal[coal['t'] < 2022]['t'].nunique()
        n_post = coal[coal['t'] >= 2022]['t'].nunique()
        coal_pre_ann = coal_pre / max(n_pre, 1)
        coal_post_ann = coal_post / max(n_post, 1)
        coal_change = coal_post_ann - coal_pre_ann  # Negative = reduction

        # Gateway trade
        gw = hs6_data[hs6_data['is_gateway'] == 1]
        gw_pre = gw[gw['t'] < 2022]['v'].sum()
        gw_post = gw[gw['t'] >= 2022]['v'].sum()
        gw_pre_ann = gw_pre / max(n_pre, 1)
        gw_post_ann = gw_post / max(n_post, 1)
        gw_change = gw_post_ann - gw_pre_ann  # Positive = increase

        if coal_pre_ann > 0 or gw_pre_ann > 0:  # Only meaningful products
            gap_data.append({
                'hs6': hs6,
                'coal_reduction': -coal_change,  # Make positive for reduction
                'gw_increase': gw_change,
                'coal_pre': coal_pre_ann,
                'gw_pre': gw_pre_ann
            })

    gap_df = pd.DataFrame(gap_data)
    prt(f'Products with data: {len(gap_df)}')

    # Filter to products with meaningful trade
    gap_meaningful = gap_df[(gap_df['coal_pre'] > 100) | (gap_df['gw_pre'] > 100)]  # >$100K
    prt(f'Products with meaningful trade (>$100K): {len(gap_meaningful)}')

    # Correlation
    from scipy import stats
    if len(gap_meaningful) > 10:
        corr, pval_corr = stats.pearsonr(gap_meaningful['coal_reduction'], gap_meaningful['gw_increase'])
        # Also rank correlation (more robust)
        spearman_corr, spearman_p = stats.spearmanr(gap_meaningful['coal_reduction'], gap_meaningful['gw_increase'])

        prt(f'\n  Pearson correlation: r={corr:.4f}, p={pval_corr:.4f}{stars(pval_corr)}')
        prt(f'  Spearman correlation: ρ={spearman_corr:.4f}, p={spearman_p:.4f}{stars(spearman_p)}')

        # Interpretation
        if corr > 0 and pval_corr < 0.05:
            prt('  → POSITIVE significant correlation: supports CIRCUMVENTION hypothesis')
            prt('  (Products with larger coalition reductions see larger gateway increases)')
        elif corr <= 0 or pval_corr >= 0.05:
            prt('  → Weak/no correlation: consistent with DEMAND SUBSTITUTION')

        results.append({
            'test': 'Supply Gap Correlation (Pearson)',
            'estimate': corr, 'pval': pval_corr, 'n_products': len(gap_meaningful),
            'interpretation': 'circumvention' if (corr > 0 and pval_corr < 0.05) else 'ambiguous'
        })
        results.append({
            'test': 'Supply Gap Correlation (Spearman)',
            'estimate': spearman_corr, 'pval': spearman_p, 'n_products': len(gap_meaningful),
            'interpretation': 'circumvention' if (spearman_corr > 0 and spearman_p < 0.05) else 'ambiguous'
        })

        # Regression: gw_increase = α + β × coal_reduction + ε (WLS weighted by trade volume)
        from sklearn.linear_model import LinearRegression
        X = gap_meaningful[['coal_reduction']].values
        y = gap_meaningful['gw_increase'].values
        weights = np.sqrt(gap_meaningful['coal_pre'].values + gap_meaningful['gw_pre'].values)

        reg = LinearRegression().fit(X, y, sample_weight=weights)
        beta_reg = reg.coef_[0]
        prt(f'\n  WLS regression (trade-weighted): β={beta_reg:.4f}')
        prt(f'  Interpretation: for every $1 reduction in coalition exports,')
        prt(f'  gateway exports increase by ${beta_reg:.2f}')

        results.append({
            'test': 'Supply Gap WLS Regression',
            'estimate': beta_reg, 'pval': np.nan, 'n_products': len(gap_meaningful),
            'interpretation': f'${beta_reg:.2f} gateway increase per $1 coalition reduction'
        })

    # =========================================================
    # Test 2: Exposure-Interaction Triple-DID
    # =========================================================
    prt('\n\n--- Test 2: Exposure-Interaction Triple-DID ---')
    prt('High vs Low exposure × Sanctioned × Post')

    third = to_rus[to_rus['is_third'] == 1].copy()

    # Singleton removal
    grp = third.groupby('partner_hs6')['t'].nunique()
    multi = grp[grp > 1].index
    third = third[third['partner_hs6'].isin(multi)].copy()

    # High/Low exposure split (median)
    third_partners = third.groupby('partner')['exposure'].first()
    median_exp = third_partners.median()
    prt(f'Median exposure: {median_exp:.4f}')
    third['high_exposure'] = (third['exposure'] > median_exp).astype(int)

    # Treatment: sanctioned × post × exposure interaction
    third['sanc_post'] = (third['sanctioned'].fillna(0).astype(int) * third['post']).astype(int)
    third['sanc_post_hi'] = (third['sanc_post'] * third['high_exposure']).astype(int)
    third['sanc_post_lo'] = (third['sanc_post'] * (1 - third['high_exposure'])).astype(int)

    # Spec A: Separate high vs low exposure rerouting
    prt('\n  Spec A: Separate High vs Low exposure effects')
    try:
        t0 = time.time()
        m = pf.feols('ln_v ~ sanc_post_hi + sanc_post_lo | partner_hs6 + partner_year + hs2_year',
                      data=third, vcov={'CRV1': 'partner_hs6'})

        b_hi = m.coef()['sanc_post_hi']
        se_hi = m.se()['sanc_post_hi']
        p_hi = m.pvalue()['sanc_post_hi']
        pct_hi = (np.exp(b_hi) - 1) * 100

        b_lo = m.coef()['sanc_post_lo']
        se_lo = m.se()['sanc_post_lo']
        p_lo = m.pvalue()['sanc_post_lo']
        pct_lo = (np.exp(b_lo) - 1) * 100

        elapsed = time.time() - t0
        prt(f'    High exposure: β={b_hi:.4f} ({se_hi:.4f}){stars(p_hi)}, %={pct_hi:.1f}%')
        prt(f'    Low exposure:  β={b_lo:.4f} ({se_lo:.4f}){stars(p_lo)}, %={pct_lo:.1f}%')
        prt(f'    Ratio (high/low): {pct_hi/pct_lo:.1f}x' if pct_lo != 0 else '    Low = 0')

        results.append({
            'test': 'Rerouting: High Exposure', 'estimate': b_hi,
            'pval': p_hi, 'pct': pct_hi, 'nobs': get_nobs(m)
        })
        results.append({
            'test': 'Rerouting: Low Exposure', 'estimate': b_lo,
            'pval': p_lo, 'pct': pct_lo, 'nobs': get_nobs(m)
        })

        # If high >> low → supports circumvention (needs infrastructure)
        if pct_hi > 0 and pct_lo <= 0:
            prt('  → High-exposure positive, low-exposure zero/negative: STRONG circumvention evidence')
        elif pct_hi > pct_lo * 2 and pct_hi > 0:
            prt('  → High much larger than low: supports circumvention concentration')
        else:
            prt('  → Both similar: consistent with diffuse demand substitution')

    except Exception as e:
        prt(f'  ERROR: {e}')

    # =========================================================
    # Test 3: Non-Sanctioned Placebo for Gateways
    # =========================================================
    prt('\n\n--- Test 3: Gateway Placebo (Non-Sanctioned Products) ---')
    prt('Do gateways increase ALL exports to Russia, or only sanctioned?')

    gw_data = to_rus[to_rus['is_gateway'] == 1].copy()
    grp_gw = gw_data.groupby('partner_hs6')['t'].nunique()
    multi_gw = grp_gw[grp_gw > 1].index
    gw_data = gw_data[gw_data['partner_hs6'].isin(multi_gw)].copy()

    gw_data['sanc_post'] = (gw_data['sanctioned'].fillna(0).astype(int) * gw_data['post']).astype(int)
    # Also need: post × non-sanctioned (placebo)
    gw_data['nonsanc_post'] = ((1 - gw_data['sanctioned'].fillna(0).astype(int)) * gw_data['post']).astype(int)

    prt(f'  Gateway sample: {len(gw_data):,} rows, {gw_data["partner"].nunique()} countries')

    # Spec: DiD within gateways — sanctioned vs non-sanctioned products × post
    prt('\n  Spec: Sanctioned × Post within gateways')
    try:
        t0 = time.time()
        m = pf.feols('ln_v ~ sanc_post | partner_hs6 + partner_year + hs2_year',
                      data=gw_data, vcov={'CRV1': 'partner_hs6'})
        b, se, p = m.coef()['sanc_post'], m.se()['sanc_post'], m.pvalue()['sanc_post']
        pct = (np.exp(b) - 1) * 100
        prt(f'    Sanctioned × Post: β={b:.4f} ({se:.4f}){stars(p)}, %={pct:.1f}%')

        results.append({
            'test': 'Gateway Sanctioned DiD', 'estimate': b,
            'pval': p, 'pct': pct, 'nobs': get_nobs(m)
        })
    except Exception as e:
        prt(f'  ERROR: {e}')

    # Compare: What about non-gateway third countries?
    prt('\n  Comparison: Non-gateway third countries')
    nongw_data = to_rus[(to_rus['is_third'] == 1) & (to_rus['is_gateway'] == 0)].copy()
    grp_ngw = nongw_data.groupby('partner_hs6')['t'].nunique()
    multi_ngw = grp_ngw[grp_ngw > 1].index
    nongw_data = nongw_data[nongw_data['partner_hs6'].isin(multi_ngw)].copy()
    nongw_data['sanc_post'] = (nongw_data['sanctioned'].fillna(0).astype(int) * nongw_data['post']).astype(int)

    try:
        m = pf.feols('ln_v ~ sanc_post | partner_hs6 + partner_year + hs2_year',
                      data=nongw_data, vcov={'CRV1': 'partner_hs6'})
        b, se, p = m.coef()['sanc_post'], m.se()['sanc_post'], m.pvalue()['sanc_post']
        pct = (np.exp(b) - 1) * 100
        prt(f'    Non-gateway Sanctioned × Post: β={b:.4f} ({se:.4f}){stars(p)}, %={pct:.1f}%')

        results.append({
            'test': 'Non-Gateway Sanctioned DiD', 'estimate': b,
            'pval': p, 'pct': pct, 'nobs': get_nobs(m)
        })
    except Exception as e:
        prt(f'  ERROR: {e}')

    # --- Save results ---
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'b7_rerouting_identification.csv', index=False)
    prt(f'\nSaved: {RESULTS_DIR}/b7_rerouting_identification.csv')

    # --- Summary ---
    prt('\n' + '='*60)
    prt('B-7.2 Summary: Circumvention vs Demand Substitution')
    prt('='*60)
    prt('If these hold → CIRCUMVENTION dominates:')
    prt('  (1) Positive supply-gap correlation (coalition reduction → gateway increase)')
    prt('  (2) High-exposure >> Low-exposure rerouting')
    prt('  (3) Gateway sanctioned >> Gateway non-sanctioned increase')
    prt('If none holds → DEMAND SUBSTITUTION dominates')

    return df


if __name__ == '__main__':
    print('='*60)
    print('B-7: Analytical Resolution of v6 Review Concerns')
    print('='*60)

    b71_balanced_panel_ppml()
    b72_rerouting_identification()

    print('\n' + '='*60)
    print('B-7 Analysis Complete')
    print('='*60)
