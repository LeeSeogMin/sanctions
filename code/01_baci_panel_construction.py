#!/usr/bin/env python3
"""
01_baci_panel_construction.py
BACI HS6 data loading + EU Sanctions HS6 merge → analysis panel construction

Input:
  - data/baci/BACI_HS92_Y{year}_V202601.csv (2015-2024)
  - data/raw/EU_sanctions_HS6.dta (Chupilkin et al.)
  - data/baci/country_codes_V202601.csv

Output:
  - data/processed/baci_russia_panel.parquet
  - data/processed/sanctions_hs6_master.csv

Note: Requires BACI HS92 v202601 (~8.2 GB) downloaded from CEPII.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

# ── 경로 설정 ──
BACI_DIR = Path('data/baci')
OUTPUT_DIR = Path('data/processed')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUSSIA = 643
YEARS = range(2015, 2025)

# ── B-0.1 Coalition 티어 정의 ──
EU27 = [40, 56, 100, 191, 196, 203, 208, 233, 246, 251, 276, 300,
        348, 372, 380, 428, 440, 442, 470, 528, 616, 620, 642,
        703, 705, 724, 752]

TIER1_NON_EU = {
    826: 'GBR',   # UK
    842: 'USA',   # US
    124: 'CAN',   # Canada
    392: 'JPN',   # Japan
    579: 'NOR',   # Norway
    757: 'CHE',   # Switzerland
}

TIER2_EXTRA = {
    36: 'AUS',    # Australia
    554: 'NZL',   # New Zealand
    352: 'ISL',   # Iceland
    410: 'KOR',   # South Korea
    702: 'SGP',   # Singapore
}

TIER1 = set(EU27) | set(TIER1_NON_EU.keys())  # 33 countries
TIER2 = TIER1 | set(TIER2_EXTRA.keys())        # 38 countries


def load_sanctions_hs6():
    """EU sanctions HS6 마스터 테이블 생성"""
    print('=== Loading EU Sanctions HS6 ===')
    eu = pd.read_stata('data/raw/EU_sanctions_HS6.dta')

    # 핵심 변수 정리
    sanctions = eu[['Code', 'Date', 'EU_sanction', 'partially_exempt',
                     'dual_use', 'industrial_cap', 'luxury', 'military_tech',
                     'aviation', 'firearms', 'oil_exploration', 'oil_refining']].copy()

    sanctions.rename(columns={'Code': 'hs6', 'Date': 'sanction_date',
                               'EU_sanction': 'sanctioned'}, inplace=True)

    # 제재 발효 연도 추출
    sanctions['sanction_year'] = np.nan
    mask = sanctions['sanctioned'] == 1
    sanctions.loc[mask, 'sanction_year'] = pd.to_datetime(
        sanctions.loc[mask, 'sanction_date']).dt.year.astype(float).values

    # 카테고리 통합
    cat_cols = ['dual_use', 'industrial_cap', 'luxury', 'military_tech',
                'aviation', 'firearms', 'oil_exploration', 'oil_refining']
    def get_category(row):
        for c in cat_cols:
            if row[c] == 1:
                return c
        return 'non_sanctioned'
    sanctions['category'] = sanctions.apply(get_category, axis=1)

    # HS6 코드를 문자열 6자리로 통일
    sanctions['hs6'] = sanctions['hs6'].astype(str).str.strip().str.zfill(6)

    print(f'Total HS6 codes: {len(sanctions)}')
    print(f'Sanctioned: {sanctions["sanctioned"].sum():.0f}')
    print(f'Non-sanctioned: {sanctions["sanctioned"].isna().sum():.0f}')
    print(f'Sanction year distribution:')
    print(sanctions[sanctions['sanctioned']==1]['sanction_year'].value_counts().sort_index().to_string())

    # 저장
    sanctions.to_csv(OUTPUT_DIR / 'sanctions_hs6_master.csv', index=False)
    print(f'Saved: {OUTPUT_DIR}/sanctions_hs6_master.csv')

    return sanctions


def load_baci_russia(sanctions_df):
    """BACI에서 러시아 관련 무역 데이터 로딩 + 제재 변수 병합"""
    print('\n=== Loading BACI Russia Trade (2015-2024) ===')

    all_dfs = []
    t0 = time.time()

    for year in YEARS:
        fpath = BACI_DIR / f'BACI_HS92_Y{year}_V202601.csv'
        df = pd.read_csv(fpath)

        # 러시아 관련만 필터 (수출 to Russia OR 수입 from Russia)
        rus = df[(df['i'] == RUSSIA) | (df['j'] == RUSSIA)].copy()

        # 방향 변수 추가
        rus['direction'] = np.where(rus['j'] == RUSSIA, 'to_russia', 'from_russia')
        rus['partner'] = np.where(rus['j'] == RUSSIA, rus['i'], rus['j'])

        # HS6 코드 문자열 변환
        rus['hs6'] = rus['k'].astype(str).str.zfill(6)

        all_dfs.append(rus)
        print(f'  {year}: {len(rus):>8,} rows ({time.time()-t0:.1f}s)')

    panel = pd.concat(all_dfs, ignore_index=True)
    print(f'\nTotal Russia panel: {len(panel):,} rows')

    # ── 제재 변수 병합 ──
    print('\n=== Merging Sanctions Classification ===')
    sanctions_merge = sanctions_df[['hs6', 'sanctioned', 'sanction_year',
                                     'category', 'partially_exempt']].copy()

    panel = panel.merge(sanctions_merge, on='hs6', how='left')

    # BACI에 있지만 EU sanctions 리스트에 없는 코드 → non-sanctioned 처리
    panel['sanctioned'] = panel['sanctioned'].fillna(0).astype(int)
    panel['category'] = panel['category'].fillna('not_in_eu_list')
    panel['partially_exempt'] = panel['partially_exempt'].fillna(0).astype(int)

    # ── 처치 변수 생성 (staggered) ──
    # treated = 1 if (HS6 sanctioned) AND (year >= sanction_year) AND (partner in coalition)
    panel['hs6_sanctioned_at_t'] = 0
    mask = (panel['sanctioned'] == 1) & (panel['t'] >= panel['sanction_year'])
    panel.loc[mask, 'hs6_sanctioned_at_t'] = 1

    # Coalition 변수
    panel['coalition_tier1'] = panel['partner'].isin(TIER1).astype(int)
    panel['coalition_tier2'] = panel['partner'].isin(TIER2).astype(int)

    # Triple-DID 처치: sanctioned_at_t × coalition
    panel['treated_t1'] = panel['hs6_sanctioned_at_t'] * panel['coalition_tier1']
    panel['treated_t2'] = panel['hs6_sanctioned_at_t'] * panel['coalition_tier2']

    # ── 고정효과 변수 ──
    panel['partner_hs6'] = panel['partner'].astype(str) + '_' + panel['hs6']
    panel['hs6_year'] = panel['hs6'] + '_' + panel['t'].astype(str)
    panel['partner_year'] = panel['partner'].astype(str) + '_' + panel['t'].astype(str)

    print(f'\n=== Panel Summary ===')
    print(f'Rows: {len(panel):,}')
    print(f'Years: {panel["t"].min()}-{panel["t"].max()}')
    print(f'Unique partners: {panel["partner"].nunique()}')
    print(f'Unique HS6: {panel["hs6"].nunique()}')
    print(f'Direction split:')
    print(panel['direction'].value_counts().to_string())
    print(f'\nSanctioned at t: {panel["hs6_sanctioned_at_t"].sum():,} / {len(panel):,} ({panel["hs6_sanctioned_at_t"].mean()*100:.1f}%)')
    print(f'Treated (Tier 1): {panel["treated_t1"].sum():,} ({panel["treated_t1"].mean()*100:.1f}%)')
    print(f'Treated (Tier 2): {panel["treated_t2"].sum():,} ({panel["treated_t2"].mean()*100:.1f}%)')

    # Direction별 요약
    for d in ['to_russia', 'from_russia']:
        sub = panel[panel['direction'] == d]
        print(f'\n--- {d} ---')
        print(f'  Rows: {len(sub):,}')
        print(f'  Sanctioned at t: {sub["hs6_sanctioned_at_t"].sum():,} ({sub["hs6_sanctioned_at_t"].mean()*100:.1f}%)')
        print(f'  Treated T1: {sub["treated_t1"].sum():,}')
        # Year-wise trade value
        yearly = sub.groupby('t')['v'].sum()
        print(f'  Trade value (top 3 years): {yearly.nlargest(3).to_dict()}')

    # ── 저장 (parquet for efficiency) ──
    output_path = OUTPUT_DIR / 'baci_russia_panel.parquet'
    panel.to_parquet(output_path, index=False)
    print(f'\nSaved: {output_path} ({output_path.stat().st_size/1e6:.1f} MB)')

    return panel


def validate_panel(panel):
    """패널 검증"""
    print('\n=== Validation ===')

    # 1. Pre/Post 분포
    pre = panel[panel['t'] < 2022]
    post = panel[panel['t'] >= 2022]
    print(f'Pre-2022: {len(pre):,} rows')
    print(f'Post-2022: {len(post):,} rows')

    # 2. to_russia (imports to Russia) 방향만 — 제재 직접효과 분석용
    imports = panel[panel['direction'] == 'to_russia']
    print(f'\nImports to Russia (main analysis direction):')
    print(f'  Total: {len(imports):,}')
    print(f'  Coalition Tier 1 partners: {imports["coalition_tier1"].sum():,} ({imports["coalition_tier1"].mean()*100:.1f}%)')

    # 3. Staggered timing check
    sanctioned = panel[panel['sanctioned'] == 1]
    print(f'\nStaggered treatment check (sanctioned HS6 only):')
    for y in sorted(sanctioned['sanction_year'].dropna().unique()):
        n_codes = sanctioned[sanctioned['sanction_year']==y]['hs6'].nunique()
        print(f'  Sanction year {int(y)}: {n_codes} HS6 codes')

    # 4. Category distribution in panel
    imports_sanc = imports[imports['hs6_sanctioned_at_t'] == 1]
    print(f'\nCategory distribution (imports, sanctioned at t):')
    print(imports_sanc['category'].value_counts().to_string())

    # 5. Trade value by sanctioned status over time
    print(f'\nTrade value (to_russia, billions USD):')
    for y in range(2015, 2025):
        yr = imports[imports['t'] == y]
        v_sanc = yr[yr['hs6_sanctioned_at_t']==1]['v'].sum() / 1e3
        v_non = yr[yr['hs6_sanctioned_at_t']==0]['v'].sum() / 1e3
        v_total = yr['v'].sum() / 1e3
        print(f'  {y}: sanc=${v_sanc:.1f}B, non=${v_non:.1f}B, total=${v_total:.1f}B')


if __name__ == '__main__':
    t_start = time.time()

    # Step 1: Load and prepare sanctions HS6 master
    sanctions_df = load_sanctions_hs6()

    # Step 2: Load BACI + merge sanctions
    panel = load_baci_russia(sanctions_df)

    # Step 3: Validate
    validate_panel(panel)

    print(f'\n=== Done ({time.time()-t_start:.1f}s) ===')
