#!/usr/bin/env python3
"""
03_rerouting_analysis.py
Third-country rerouting analysis

Secondary estimand:
  Change in sanctioned HS6 exports from non-coalition third countries to Russia

Analysis:
  1. Third country → Russia: sanctioned HS6 increase (DiD)
  2. Coalition → third country: sanctioned HS6 increase (indirect route)
  3. Exposure index + top rerouting countries
  4. Net leakage estimation (direct + rerouting effects)

Input:
  - data/processed/baci_russia_panel.parquet

Output:
  - output/results/b4_rerouting.csv
  - output/results/b4_exposure_index.csv
  - output/figures/b4_rerouting_top10.png
"""

import pandas as pd
import numpy as np
import pyfixest as pf
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

PANEL_PATH = Path('data/processed/baci_russia_panel.parquet')
RESULTS_DIR = Path('output/results')
FIGURES_DIR = Path('output/figures')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Coalition definitions
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


def load_panel():
    """전체 패널 로딩"""
    prt('=== Loading Panel ===')
    panel = pd.read_parquet(PANEL_PATH)
    prt(f'Full panel: {len(panel):,} rows')

    # from_russia: 러시아에서 수출 (j에서 보면 partner가 수입자)
    # to_russia: 러시아로 수입 (partner가 수출자)

    panel['is_coalition'] = panel['partner'].isin(TIER1).astype(int)
    panel['is_third'] = ((~panel['partner'].isin(TIER1)) &
                          (panel['partner'] != RUSSIA)).astype(int)
    panel['post'] = (panel['t'] >= 2022).astype(int)
    panel['ln_v'] = np.log(panel['v'] + 1)

    return panel


def analyze_third_to_russia(panel):
    """B-4.2: 제3국→러시아 제재 HS6 증가 여부"""
    prt('\n' + '='*60)
    prt('B-4.2: Third Country → Russia (Sanctioned HS6)')
    prt('='*60)

    # to_russia 방향에서 비동맹 파트너만
    to_rus = panel[(panel['direction'] == 'to_russia') &
                   (panel['is_third'] == 1)].copy()

    prt(f'Third → Russia sample: {len(to_rus):,} rows')
    prt(f'Sanctioned at t: {to_rus["hs6_sanctioned_at_t"].sum():,}')

    # 싱글턴 제거
    grp = to_rus.groupby('partner_hs6')['t'].nunique()
    multi = grp[grp > 1].index
    to_rus = to_rus[to_rus['partner_hs6'].isin(multi)].copy()
    prt(f'After singletons: {len(to_rus):,} rows')

    results = []

    # DiD: sanctioned_at_t (already staggered)
    # 여기서는 제3국의 경우, treated = hs6_sanctioned_at_t
    # 즉 제재 발효 후 제재 HS6를 제3국이 러시아에 수출하는 양 변화
    prt('\n--- OLS: Third→Russia, sanctioned_at_t effect ---')
    try:
        t0 = time.time()
        m = pf.feols(
            'ln_v ~ hs6_sanctioned_at_t | partner_hs6 + hs6_year + partner_year',
            data=to_rus, vcov={'CRV1': 'partner_hs6'}
        )
        beta = m.coef()['hs6_sanctioned_at_t']
        se = m.se()['hs6_sanctioned_at_t']
        pval = m.pvalue()['hs6_sanctioned_at_t']
        pct = (np.exp(beta) - 1) * 100
        nobs = get_nobs(m)
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        prt(f'  β = {beta:.4f} ({se:.4f}){sig}, % = {pct:.1f}%, N = {nobs:,} ({time.time()-t0:.0f}s)')
        results.append({
            'spec': 'Third→Russia | sanctioned_at_t', 'beta': beta, 'se': se,
            'pval': pval, 'pct_effect': pct, 'nobs': nobs
        })
    except Exception as e:
        prt(f'  ERROR: {e}')

    # Category별 우회
    prt('\n--- Category-level rerouting ---')
    for cat in ['dual_use', 'industrial_cap', 'luxury', 'military_tech']:
        sub = to_rus[(to_rus['category'] == cat) |
                     (to_rus['category'].isin(['non_sanctioned', 'not_in_eu_list']))].copy()
        sub['treated_cat'] = ((sub['category'] == cat) &
                              (sub['hs6_sanctioned_at_t'] == 1)).astype(int)
        try:
            m = pf.feols('ln_v ~ treated_cat | partner_hs6 + hs6_year + partner_year',
                        data=sub, vcov={'CRV1': 'partner_hs6'})
            beta = m.coef()['treated_cat']
            se = m.se()['treated_cat']
            pval = m.pvalue()['treated_cat']
            pct = (np.exp(beta) - 1) * 100
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            prt(f'  {cat:20s}: β={beta:.4f} ({se:.4f}){sig}, % = {pct:.1f}%')
            results.append({
                'spec': f'Third→Russia | {cat}', 'beta': beta, 'se': se,
                'pval': pval, 'pct_effect': pct
            })
        except Exception as e:
            prt(f'  {cat}: ERROR: {e}')

    return results


def compute_exposure_index(panel):
    """B-4.3: Exposure Index 산출"""
    prt('\n' + '='*60)
    prt('B-4.3: Exposure Index')
    prt('='*60)

    # 사전기간: 2018-2021
    pre = panel[(panel['t'] >= 2018) & (panel['t'] <= 2021)].copy()

    # 제재 HS6 식별: sanction_year가 있고 sanctioned=1인 코드
    sanctioned_hs6 = set(panel[panel['sanctioned'] == 1]['hs6'].unique())
    prt(f'Sanctioned HS6 codes: {len(sanctioned_hs6):,}')

    # (1) Coalition → c 중 제재품 비중 (to_russia 아닌, 일반 무역)
    # BACI에서는 i=exporter, j=importer
    # Coalition이 수출자(i), 제3국이 수입자(j) — direction 무관, 전체 BACI 필요
    # 하지만 우리 패널은 러시아 관련만 → proxy 사용:
    # to_russia 방향의 무역 패턴으로 제3국 특성 추정

    # (2) c → Russia 중 제재품 비중
    to_rus_pre = pre[(pre['direction'] == 'to_russia') &
                     (pre['is_third'] == 1)]

    # 국가별 제재/비제재 품목 무역
    country_trade = to_rus_pre.groupby('partner').agg(
        total_v=('v', 'sum'),
        total_rows=('v', 'count')
    )

    sanctioned_trade = to_rus_pre[to_rus_pre['hs6'].isin(sanctioned_hs6)].groupby('partner').agg(
        sanc_v=('v', 'sum'),
        sanc_rows=('v', 'count')
    )

    exposure = country_trade.join(sanctioned_trade, how='left').fillna(0)
    exposure['sanc_share'] = exposure['sanc_v'] / exposure['total_v']
    exposure['sanc_share'] = exposure['sanc_share'].fillna(0)

    # Post-period 변화율 (simple growth)
    post_to_rus = panel[(panel['direction'] == 'to_russia') &
                        (panel['is_third'] == 1) &
                        (panel['t'] >= 2022)]

    post_trade = post_to_rus.groupby('partner')['v'].sum().rename('post_v')
    pre_trade = to_rus_pre.groupby('partner')['v'].sum().rename('pre_v')

    exposure = exposure.join(post_trade).join(pre_trade)
    exposure['growth'] = (exposure['post_v'] / (exposure['pre_v'] / 4 * 3)) - 1  # normalize to 3 years
    exposure['growth'] = exposure['growth'].replace([np.inf, -np.inf], np.nan)

    # Exposure index (simplified: sanc_share as proxy)
    exposure['exposure_index'] = exposure['sanc_share']

    # 국가 코드 매핑
    try:
        cc = pd.read_csv('data/raw/country_codes_V202601.csv')
        cc_map = dict(zip(cc['country_code'], cc['country_iso3']))
        exposure['iso3'] = exposure.index.map(cc_map)
    except:
        exposure['iso3'] = exposure.index.astype(str)

    # 정렬 (exposure 기준)
    exposure = exposure.sort_values('exposure_index', ascending=False)

    prt('\n--- Top 20 Exposure Countries ---')
    top20 = exposure.head(20)
    for _, row in top20.iterrows():
        growth_str = f'{row["growth"]*100:.0f}%' if pd.notna(row['growth']) else 'N/A'
        prt(f'  {row["iso3"]:>5s}: exposure={row["exposure_index"]:.3f}, '
            f'sanc_share={row["sanc_share"]:.3f}, '
            f'pre_v={row["pre_v"]/1e3:.0f}M, post_v={row["post_v"]/1e3:.0f}M, '
            f'growth={growth_str}')

    # 고정 후보 8국 확인
    fixed_candidates = {156: 'CHN', 699: 'IND', 792: 'TUR', 784: 'ARE',
                        51: 'ARM', 268: 'GEO', 398: 'KAZ', 417: 'KGZ'}
    prt('\n--- Fixed Candidates (8) ---')
    for code, iso in fixed_candidates.items():
        if code in exposure.index:
            row = exposure.loc[code]
            growth_str = f'{row["growth"]*100:.0f}%' if pd.notna(row['growth']) else 'N/A'
            prt(f'  {iso}: exposure={row["exposure_index"]:.3f}, growth={growth_str}, '
                f'pre_v={row["pre_v"]/1e3:.0f}M')
        else:
            prt(f'  {iso}: not in data')

    # 저장
    exposure.to_csv(RESULTS_DIR / 'b4_exposure_index.csv')
    prt(f'\nSaved: {RESULTS_DIR}/b4_exposure_index.csv')

    return exposure


def compute_net_leakage(panel, direct_beta=-0.4726, rerouting_results=None):
    """순누출 효과 추정"""
    prt('\n' + '='*60)
    prt('NET LEAKAGE EFFECT')
    prt('='*60)

    # 직접효과: coalition → Russia 감소
    to_rus = panel[(panel['direction'] == 'to_russia')]
    coalition_pre = to_rus[(to_rus['is_coalition'] == 1) & (to_rus['t'] < 2022)]
    sanctioned_pre_v = coalition_pre[coalition_pre['hs6'].isin(
        set(panel[panel['sanctioned'] == 1]['hs6'].unique()))]['v'].sum()
    direct_reduction = sanctioned_pre_v * (1 - np.exp(direct_beta))

    # 제3국 우회: third → Russia 증가
    third_pre = to_rus[(to_rus['is_third'] == 1) & (to_rus['t'] < 2022)]
    third_post = to_rus[(to_rus['is_third'] == 1) & (to_rus['t'] >= 2022)]

    sanctioned_hs6 = set(panel[panel['sanctioned'] == 1]['hs6'].unique())
    third_pre_sanc = third_pre[third_pre['hs6'].isin(sanctioned_hs6)]['v'].sum() / 7 * 3  # annualize to 3 post years
    third_post_sanc = third_post[third_post['hs6'].isin(sanctioned_hs6)]['v'].sum()
    third_increase = third_post_sanc - third_pre_sanc

    leakage_ratio = third_increase / direct_reduction * 100 if direct_reduction > 0 else np.nan

    prt(f'\n  Direct reduction (coalition, PPML β={direct_beta:.3f}):')
    prt(f'    Pre-sanction trade in sanctioned HS6: ${sanctioned_pre_v/1e3:.0f}M')
    prt(f'    Estimated reduction: ${direct_reduction/1e3:.0f}M')
    prt(f'\n  Third-country increase:')
    prt(f'    Pre annualized: ${third_pre_sanc/1e3:.0f}M')
    prt(f'    Post total: ${third_post_sanc/1e3:.0f}M')
    prt(f'    Increase: ${third_increase/1e3:.0f}M')
    prt(f'\n  Leakage ratio: {leakage_ratio:.1f}%')

    return {
        'direct_reduction_M': direct_reduction / 1e3,
        'third_increase_M': third_increase / 1e3,
        'leakage_ratio_pct': leakage_ratio
    }


def plot_rerouting(exposure):
    """Top 10 우회국 시각화"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    top10 = exposure.head(10).copy()
    top10 = top10[top10['growth'].notna()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Exposure index
    ax1.barh(range(len(top10)), top10['exposure_index'], color='#2166ac', alpha=0.7)
    ax1.set_yticks(range(len(top10)))
    ax1.set_yticklabels(top10['iso3'])
    ax1.set_xlabel('Exposure Index')
    ax1.set_title('A. Sanctions Exposure Index\n(pre-period sanctioned HS6 share)', fontsize=12)
    ax1.invert_yaxis()

    # Right: Trade growth
    colors = ['#b2182b' if g > 0 else '#2166ac' for g in top10['growth']]
    ax2.barh(range(len(top10)), top10['growth'] * 100, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(top10)))
    ax2.set_yticklabels(top10['iso3'])
    ax2.set_xlabel('Trade Growth (%)')
    ax2.set_title('B. Third→Russia Trade Growth\n(post-2022 vs pre-2022)', fontsize=12)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'b4_rerouting_top10.png', dpi=150)
    prt(f'Saved: {FIGURES_DIR}/b4_rerouting_top10.png')
    plt.close()


if __name__ == '__main__':
    t_start = time.time()

    panel = load_panel()

    # B-4.2: Third → Russia
    rerouting_results = analyze_third_to_russia(panel)

    # B-4.3: Exposure Index
    exposure = compute_exposure_index(panel)

    # Net Leakage
    leakage = compute_net_leakage(panel)

    # Plot
    plot_rerouting(exposure)

    # Save results
    pd.DataFrame(rerouting_results).to_csv(RESULTS_DIR / 'b4_rerouting.csv', index=False)
    prt(f'Saved: {RESULTS_DIR}/b4_rerouting.csv')

    prt('\n' + '='*60)
    prt('B-4 SUMMARY')
    prt('='*60)
    for r in rerouting_results:
        sig = '***' if r['pval'] < 0.001 else '**' if r['pval'] < 0.01 else '*' if r['pval'] < 0.05 else ''
        prt(f"  {r['spec']:40s}: β={r['beta']:.4f} ({r['se']:.4f}){sig}  → {r['pct_effect']:.1f}%")
    prt(f"\n  Net leakage ratio: {leakage['leakage_ratio_pct']:.1f}%")

    prt(f'\n=== Total: {time.time()-t_start:.1f}s ===')
