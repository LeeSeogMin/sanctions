#!/usr/bin/env python3
"""Persist reported aggregates that earlier scripts printed but never stored.

Recomputes descriptive quantities that appear in the paper and writes them to
results/ as CSV, so every reported figure has a file behind it. Uses OLS and
descriptive statistics only; PPML is not re-run here.

Each output is compared against the expected value and any mismatch is printed.

Outputs
  b18_ekmn_concordance_by_category.csv  concordance by product category
  b6_leakage_levels.csv                 offset-ratio dollar levels and bootstrap
  b7_gateway_growth.csv                 coalition-to-gateway growth by country
  b4_rerouting_by_category.csv          category-level third-country estimates
"""

import pandas as pd
import numpy as np
import pyfixest as pf
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

PANEL_PATH = Path('phase4_analysis/processed/baci_russia_panel.parquet')
MASTER_PATH = Path('phase4_analysis/processed/sanctions_hs6_master.csv')
CONCORDANCE_CODES = Path('phase4_analysis/results/b18_ekmn_concordance_codes.csv')
CHUPILKIN_DTA = Path('phase3_data/raw/EU_sanctions_HS6.dta')
BACI_DIR = Path('phase3_data/BACI_HS92_V202601')
RESULTS_DIR = Path('phase4_analysis/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Coalition / gateway definitions (동일하게 10b/09에서 복제)
EU27 = {40, 56, 100, 191, 196, 203, 208, 233, 246, 251, 276, 300,
        348, 372, 380, 428, 440, 442, 470, 528, 616, 620, 642,
        703, 705, 724, 752}
TIER1_NON_EU = {826, 842, 124, 392, 579, 757}
TIER1 = EU27 | TIER1_NON_EU
RUSSIA = 643
GATEWAY = {784, 51, 417, 792, 156, 860, 398, 762, 268, 699}
COUNTRY_NAMES = {784: 'UAE', 51: 'Armenia', 417: 'Kyrgyzstan', 792: 'Turkey',
                 156: 'China', 860: 'Uzbekistan', 398: 'Kazakhstan',
                 762: 'Tajikistan', 268: 'Georgia', 699: 'India'}


def prt(msg):
    print(msg, flush=True)


def check(label, got, expected, tol=0.15):
    """기대값 대조 출력. tol은 절대 허용오차(% 포인트 또는 $B)."""
    ok = abs(got - expected) <= tol
    flag = 'MATCH' if ok else 'MISMATCH'
    prt(f'  [{flag}] {label}: got={got:.1f}, expected={expected:.1f} (|Δ|={abs(got-expected):.2f})')
    return ok


# ===========================================================================
# 1. EKMN 카테고리별 콘코던스
# ===========================================================================
def ekmn_concordance_by_category():
    prt('\n' + '=' * 64)
    prt('1. EKMN concordance by category')
    prt('=' * 64)

    codes = pd.read_csv(CONCORDANCE_CODES, dtype={'hs6': str})
    codes['hs6'] = codes['hs6'].str.zfill(6)

    # 카테고리 출처: Chupilkin 처치코드의 다중 카테고리 지시 컬럼 (18_ekmn_concordance.py와 동일 처치코드 기준).
    # NOTE: 한 코드는 여러 카테고리에 속할 수 있어, 각 카테고리에 대해 해당 코드를 모두 집계한다.
    #       master CSV의 first-match single category로 join하면 luxury가 33.7%로 어긋나
    #       the expected 41.9% is not reproduced. Multi-category treatment codes are
#       therefore the basis of the reported figure.
    chu = pd.read_stata(CHUPILKIN_DTA)
    chu['Code'] = chu['Code'].astype(str).str.zfill(6)
    chu_s = chu[chu['EU_sanction'] == 1].copy()
    CATS = ['dual_use', 'industrial_cap', 'luxury', 'military_tech']
    cat_map = chu_s.set_index('Code')[CATS]

    j = codes.set_index('hs6').join(cat_map)
    chu_codes = j[j['chupilkin'] == 1]

    expected = {'dual_use': 89.2, 'industrial_cap': 99.9, 'luxury': 41.9, 'military_tech': 93.9}
    rows = []
    for c in CATS:
        sub = chu_codes[chu_codes[c] == 1]
        n = len(sub)
        n_both = int(sub['both'].sum())
        share = n_both / n * 100 if n else np.nan
        check(c, share, expected[c])
        rows.append({'category': c, 'n_chupilkin_treated': n, 'n_in_ekmn': n_both,
                     'concordance_pct': round(share, 1), 'expected_pct': expected[c]})

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / 'b18_ekmn_concordance_by_category.csv'
    df.to_csv(out, index=False)
    prt(f'  Saved: {out}')
    return df


# ===========================================================================
# 2. 누출비 구성 달러 레벨 + 부트스트랩 median
# ===========================================================================
def leakage_levels(panel):
    prt('\n' + '=' * 64)
    prt('2. Leakage component dollar levels + bootstrap median')
    prt('=' * 64)

    panel = panel.copy()
    panel['is_third'] = ((~panel['partner'].isin(TIER1)) &
                         (panel['partner'] != RUSSIA)).astype(int)
    to_rus = panel[panel['direction'] == 'to_russia'].copy()
    sanctioned_mask = to_rus['sanctioned'] == 1

    coal = to_rus[to_rus['coalition_tier1'] == 1]
    coal_sanc = coal[sanctioned_mask[coal.index]]
    n_pre = coal_sanc[coal_sanc['t'] < 2022]['t'].nunique()
    n_post = coal_sanc[coal_sanc['t'] >= 2022]['t'].nunique()

    coal_pre = coal_sanc[coal_sanc['t'] < 2022]['v'].sum() / max(n_pre, 1)
    coal_post = coal_sanc[coal_sanc['t'] >= 2022]['v'].sum() / max(n_post, 1)
    direct_reduction = coal_pre - coal_post

    third = to_rus[to_rus['is_third'] == 1]
    third_sanc = third[sanctioned_mask[third.index]]
    third_pre = third_sanc[third_sanc['t'] < 2022]['v'].sum() / max(n_pre, 1)
    third_post = third_sanc[third_sanc['t'] >= 2022]['v'].sum() / max(n_post, 1)
    rerouting_increase = third_post - third_pre

    # 단위: 패널 v는 천 달러(thousand USD). /1e6 → billion USD.
    prt('\n  Dollar levels (annualized, billion USD):')
    check('coalition pre',  coal_pre / 1e6, 75.4)
    check('coalition post', coal_post / 1e6, 28.5)
    check('third pre',      third_pre / 1e6, 53.9)
    check('third post',     third_post / 1e6, 84.0)
    check('direct reduction', direct_reduction / 1e6, 46.9, tol=0.3)
    check('rerouting increase', rerouting_increase / 1e6, 30.1, tol=0.3)

    # ----- 부트스트랩: 회귀 재추정 없이 파트너 리샘플만 → 저비용, median 산출 -----
    prt('\n  Bootstrap (500 iter, partner-level resample — NO regression):')
    partners_coal = coal_sanc['partner'].unique()
    partners_third = third_sanc['partner'].unique()
    n_boot = 500
    boot_ratios = []
    np.random.seed(42)
    for _ in range(n_boot):
        bc = np.random.choice(partners_coal, len(partners_coal), replace=True)
        bt = np.random.choice(partners_third, len(partners_third), replace=True)
        c_sub = coal_sanc[coal_sanc['partner'].isin(bc)]
        c_pre = c_sub[c_sub['t'] < 2022]['v'].sum() / max(n_pre, 1)
        c_post = c_sub[c_sub['t'] >= 2022]['v'].sum() / max(n_post, 1)
        c_red = c_pre - c_post
        t_sub = third_sanc[third_sanc['partner'].isin(bt)]
        t_pre = t_sub[t_sub['t'] < 2022]['v'].sum() / max(n_pre, 1)
        t_post = t_sub[t_sub['t'] >= 2022]['v'].sum() / max(n_post, 1)
        t_inc = t_post - t_pre
        if c_red > 0:
            boot_ratios.append(t_inc / c_red)
    boot_ratios = np.array(boot_ratios)
    boot_median = np.median(boot_ratios) * 100
    boot_mean = np.mean(boot_ratios) * 100
    leakage_ratio = rerouting_increase / direct_reduction * 100
    check('bootstrap median', boot_median, 59.2, tol=2.0)
    prt(f'  (point leakage ratio={leakage_ratio:.1f}%, bootstrap mean={boot_mean:.1f}%)')

    df = pd.DataFrame([{
        'coalition_pre_B': round(coal_pre / 1e6, 1),
        'coalition_post_B': round(coal_post / 1e6, 1),
        'direct_reduction_B': round(direct_reduction / 1e6, 1),
        'third_pre_B': round(third_pre / 1e6, 1),
        'third_post_B': round(third_post / 1e6, 1),
        'rerouting_increase_B': round(rerouting_increase / 1e6, 1),
        'leakage_ratio_pct': round(leakage_ratio, 1),
        'bootstrap_median_pct': round(boot_median, 1),
        'bootstrap_mean_pct': round(boot_mean, 1),
        'n_pre_years': n_pre, 'n_post_years': n_post, 'n_boot': n_boot,
    }])
    out = RESULTS_DIR / 'b6_leakage_levels.csv'
    df.to_csv(out, index=False)
    prt(f'  Saved: {out}')
    return df


# ===========================================================================
# 3. Gateway 국가별 증가율 (raw BACI, descriptive)
# ===========================================================================
def gateway_growth():
    prt('\n' + '=' * 64)
    prt('3. Coalition->Gateway growth (raw BACI, descriptive)')
    prt('=' * 64)

    sanctions = pd.read_csv(MASTER_PATH)
    sanctions['hs6'] = sanctions['hs6'].astype(str).str.zfill(6)
    sanctioned_set = set(sanctions[sanctions['sanctioned'] == 1]['hs6'])

    frames = []
    for year in range(2015, 2025):
        fpath = BACI_DIR / f'BACI_HS92_Y{year}_V202601.csv'
        if not fpath.exists():
            prt(f'  {year}: not found')
            continue
        t0 = time.time()
        df = pd.read_csv(fpath, dtype={'k': str},
                         usecols=['t', 'i', 'j', 'k', 'v'])
        df_cg = df[(df['i'].isin(TIER1)) & (df['j'].isin(GATEWAY))].copy()
        df_cg['hs6'] = df_cg['k'].str.zfill(6)
        agg = df_cg.groupby(['j', 'hs6'], as_index=False)['v'].sum()
        agg['t'] = year
        frames.append(agg)
        prt(f'  {year}: {len(agg):,} rows ({time.time()-t0:.0f}s)')
        del df, df_cg

    cg = pd.concat(frames, ignore_index=True)
    cg['sanctioned'] = cg['hs6'].isin(sanctioned_set).astype(int)
    n_pre = cg[cg['t'] < 2022]['t'].nunique()
    n_post = cg[cg['t'] >= 2022]['t'].nunique()

    expected = {51: 157, 417: 320, 268: 148, 792: 55}
    rows = []
    prt('\n  By gateway country (sanctioned products, annualized $B):')
    for gw in sorted(GATEWAY):
        sub = cg[cg['j'] == gw]
        ss = sub[sub['sanctioned'] == 1]
        pre = ss[ss['t'] < 2022]['v'].sum() / max(n_pre, 1) / 1e6
        post = ss[ss['t'] >= 2022]['v'].sum() / max(n_post, 1) / 1e6
        pct = (post - pre) / pre * 100 if pre > 0.01 else np.nan
        name = COUNTRY_NAMES.get(gw, str(gw))
        line = f'    {name:12s}: ${pre:8.2f}B -> ${post:8.2f}B ({pct:+.1f}%)'
        if gw in expected:
            line += f'  [expected {expected[gw]:+d}%]'
        prt(line)
        rows.append({'gateway_code': gw, 'gateway': name,
                     'pre2022_annual_B': round(pre, 2),
                     'post2022_annual_B': round(post, 2),
                     'growth_pct': round(pct, 1) if pd.notna(pct) else np.nan,
                     'expected_pct': expected.get(gw, np.nan)})

    # 집계: 제재품 / 비제재품 전체 gateway 증가율
    sc = cg[cg['sanctioned'] == 1]
    sc_pre = sc[sc['t'] < 2022]['v'].sum() / max(n_pre, 1) / 1e6
    sc_post = sc[sc['t'] >= 2022]['v'].sum() / max(n_post, 1) / 1e6
    sc_pct = (sc_post - sc_pre) / sc_pre * 100
    ns = cg[cg['sanctioned'] == 0]
    ns_pre = ns[ns['t'] < 2022]['v'].sum() / max(n_pre, 1) / 1e6
    ns_post = ns[ns['t'] >= 2022]['v'].sum() / max(n_post, 1) / 1e6
    ns_pct = (ns_post - ns_pre) / ns_pre * 100

    prt('\n  Aggregate across all gateways:')
    check('sanctioned growth', sc_pct, 28.8, tol=1.0)
    check('non-sanctioned growth', ns_pct, 10.6, tol=1.0)
    rows.append({'gateway_code': -1, 'gateway': 'ALL_sanctioned',
                 'pre2022_annual_B': round(sc_pre, 2), 'post2022_annual_B': round(sc_post, 2),
                 'growth_pct': round(sc_pct, 1), 'expected_pct': 28.8})
    rows.append({'gateway_code': -2, 'gateway': 'ALL_nonsanctioned',
                 'pre2022_annual_B': round(ns_pre, 2), 'post2022_annual_B': round(ns_post, 2),
                 'growth_pct': round(ns_pct, 1), 'expected_pct': 10.6})

    # 국가별 검증
    prt('\n  Country-level check vs expected:')
    for gw, exp in expected.items():
        r = next(x for x in rows if x['gateway_code'] == gw)
        check(COUNTRY_NAMES[gw], r['growth_pct'], float(exp), tol=3.0)

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / 'b7_gateway_growth.csv'
    df.to_csv(out, index=False)
    prt(f'  Saved: {out}')
    return df


# ===========================================================================
# 4. 카테고리별 rerouting % (feols 재추정 — 07의 카테고리 loop)
# ===========================================================================
def rerouting_by_category(panel):
    prt('\n' + '=' * 64)
    prt('4. Rerouting by category (feols, source of 08_figures_tables.py:126)')
    prt('=' * 64)

    panel = panel.copy()
    panel['is_third'] = ((~panel['partner'].isin(TIER1)) &
                         (panel['partner'] != RUSSIA)).astype(int)
    panel['ln_v'] = np.log(panel['v'] + 1)
    panel['post'] = (panel['t'] >= 2022).astype(int)

    to_rus = panel[(panel['direction'] == 'to_russia') &
                   (panel['is_third'] == 1)].copy()
    # singleton 제거 (07과 동일)
    grp = to_rus.groupby('partner_hs6')['t'].nunique()
    multi = grp[grp > 1].index
    to_rus = to_rus[to_rus['partner_hs6'].isin(multi)].copy()

    # 사양 메모: 07_rerouting_analysis.py:130의 명세는
    #   'ln_v ~ treated_cat | partner_hs6 + hs6_year + partner_year' 였으나,
    # 제3국 표본에서 hs6_sanctioned_at_t는 주어진 hs6×year 셀 내에서 모든 파트너에 대해
    # 상수이므로 hs6_year FE에 완전 흡수되어 현 패널·pyfixest에서 collinear로 실패한다.
    # The category-level third-country percentages are reproduced by the identifiable
    #   'ln_v ~ sanc_post | partner_hs6 + partner_year' (시불변 코호트 지시 × post,
    # 09_review_response.py의 rerouting 설계와 일관 — hs6_year FE 미사용)에서 재현된다.
    expected = {'dual_use': 28.2, 'industrial_cap': 47.7,
                'luxury': -17.1, 'military_tech': 27.0}
    rows = []
    for cat in ['dual_use', 'industrial_cap', 'luxury', 'military_tech']:
        sub = to_rus[(to_rus['category'] == cat) |
                     (to_rus['category'].isin(['non_sanctioned', 'not_in_eu_list']))].copy()
        sub['sanc_post'] = ((sub['category'] == cat) & (sub['post'] == 1)).astype(int)
        t0 = time.time()
        m = pf.feols('ln_v ~ sanc_post | partner_hs6 + partner_year',
                     data=sub, vcov={'CRV1': 'partner_hs6'})
        beta = m.coef()['sanc_post']
        se = m.se()['sanc_post']
        pval = m.pvalue()['sanc_post']
        pct = (np.exp(beta) - 1) * 100
        check(cat, pct, expected[cat], tol=2.0)
        prt(f'    ({time.time()-t0:.0f}s) beta={beta:.4f} se={se:.4f} p={pval:.4g}')
        rows.append({'category': cat, 'beta': round(beta, 4), 'se': round(se, 4),
                     'pval': pval, 'rerouting_pct': round(pct, 1),
                     'expected_pct': expected[cat],
                     'spec': 'sanc_post | partner_hs6 + partner_year'})

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / 'b4_rerouting_by_category.csv'
    df.to_csv(out, index=False)
    prt(f'  Saved: {out}')
    return df


if __name__ == '__main__':
    t_start = time.time()
    prt('=' * 64)
    prt('21_save_reported_aggregates.py')
    prt('=' * 64)

    panel = pd.read_parquet(PANEL_PATH)
    prt(f'Panel loaded: {len(panel):,} rows')

    ekmn_concordance_by_category()
    leakage_levels(panel)
    rerouting_by_category(panel)
    gateway_growth()   # raw BACI I/O — 마지막에 (가장 무거움)

    prt('\n' + '=' * 64)
    prt(f'Done. Total: {time.time()-t_start:.0f}s')
    prt('=' * 64)
