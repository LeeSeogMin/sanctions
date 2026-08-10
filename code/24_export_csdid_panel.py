#!/usr/bin/env python3
"""Export the panel for the Callaway-Sant'Anna estimator.

No estimation happens here. This script reshapes the panel into the form the R
`did` package expects and writes it to CSV; 25_csdid.R estimates from it. (The R
build in use has no arrow package and cannot read the parquet directly.)

Why the sample is restricted to coalition exporters
  The benchmark is a triple difference in which HS6 x year fixed effects absorb
  product-time shocks and non-coalition exporters serve as a third comparison.
  The group-time estimator is a 2x2 framework and cannot carry that structure,
  so the sample keeps coalition exporters only and compares sanctioned products
  against non-sanctioned ones. This is one leg of the triple difference and
  therefore does not measure the same quantity as the benchmark.

  Setting SAMPLE='both' also exports a version including non-coalition
  exporters; there the control pool mixes coalition-unsanctioned with
  non-coalition observations and the interpretation changes.

Panel construction
  sample     exporters to Russia
  unit       partner x HS6 pair, expanded to all years 2015-2024 with
             unobserved flows set to zero (`did` requires a balanced panel)
  outcome    y = ln(v + 1), so zeros enter as zeros
  cohort g   the product's restriction year (2016 / 2022 / 2023); never-treated
             products take g = 0
  id         partner x HS6 encoded as an integer, as `did` requires

Cohort sizes in HS6 codes: 17 in 2016, 1,760 in 2022, 90 in 2023, 2,782 never
treated. The 2016 cohort has a single pre-period year, so its pre-trend cannot
be tested.

A truncated alternative is also exported: products in the 2016 and 2023 cohorts
are dropped and the window is cut to 2017-2022, leaving a single cohort so that
staggered timing does not arise. Cutting the window literally leaves only one
post-treatment year, so a second version keeps the same products and extends to
2024. Both are exported.

Outputs
  processed/csdid_panel_coalition.csv
  processed/csdid_panel_truncated.csv
  processed/csdid_panel_all.csv   (when SAMPLE='both')
  results/b24_csdid_export_log.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time

PANEL_PATH = Path('phase4_analysis/processed/baci_russia_panel.parquet')
OUT_DIR = Path('phase4_analysis/processed')
RESULTS_DIR = Path('phase4_analysis/results')
LOG_PATH = RESULTS_DIR / 'b24_csdid_export_log.txt'

YEARS = np.arange(2015, 2025)
TRUNC_YEARS = np.arange(2017, 2023)      # literal truncation window: 2017-2022
TRUNC_YEARS_EXT = np.arange(2017, 2025)  # 같은 제품, 창만 2024까지
TRUNC_DROP_COHORTS = (2016.0, 2023.0)    # 이 코호트 제품은 절단 표본에서 제외

# 절단 표본을 왜 둘로 내보내는가
#   Once the 2016 and 2023 cohort products are dropped from the sample, there is
#   no longer a reason to cut the year window at 2022. Cutting it literally at
#   2017-2022 leaves a single post-treatment year, so no dynamics are visible.
#   Both versions are therefore exported: (a) the literal window and (b) the same
#   products with the window extended to 2024.

# 'coalition' = 연합국만 (주 사양) / 'both' = 비연합국 포함 사양도 함께 내보냄
SAMPLE = 'both'

_log_lines = []


def prt(msg):
    print(msg, flush=True)
    _log_lines.append(str(msg))


def flush_log():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text('\n'.join(_log_lines), encoding='utf-8')


def product_cohort(panel):
    """HS6 -> 제재 시행연도(g). 비제재는 0.

    관측 표본이 아니라 제재 마스터(sanction_year)에서 만든다.
    Same reason as script 22: 관측 표본
    기반으로 만들면 거래가 없던 제품-연도가 통제군으로 오분류된다.
    """
    # 죽는다. 같은 HS6에 다른 sanction_year가 있어도 첫 값을 조용히 채택했다.
    # 중복 제거 *전에* 충돌을 검사한다.
    pairs = panel[['hs6', 'sanction_year']].drop_duplicates()
    conflict = pairs.groupby('hs6')['sanction_year'].nunique(dropna=False)
    bad_hs6 = conflict[conflict > 1]
    if len(bad_hs6):
        sample = ', '.join(str(h) for h in bad_hs6.index[:5])
        raise ValueError(
            f'HS6 {len(bad_hs6)}개에 서로 다른 sanction_year가 있다 (예: {sample})')
    pc = pairs.copy()
    pc['g'] = pc['sanction_year'].fillna(0).astype(int)
    bad = sorted(set(pc['g'].unique()) - {0, 2016, 2022, 2023})
    if bad:
        raise ValueError(f'예상 밖 코호트: {bad}')
    return pc[['hs6', 'g']]


def build_balanced(to_rus, cohort, years, label):
    """관측된 partner-HS6 쌍을 전 연도로 펼치고 0을 채운다."""
    prt(f'\n--- {label} ---')
    sub = to_rus[to_rus['t'].isin(years)]
    # coalition_tier1)로 유일하게 매핑되는지 먼저 확인한다.
    pairs = sub[['partner', 'hs6', 'partner_hs6', 'coalition_tier1']].drop_duplicates()
    dup = pairs['partner_hs6'].duplicated().sum()
    if dup:
        raise ValueError(f'partner_hs6 {dup}개가 서로 다른 partner/hs6/연합국 값에 매핑된다')
    prt(f'  쌍 {len(pairs):,} x 연도 {len(years)} -> {len(pairs) * len(years):,} 칸')

    flow = sub[['partner_hs6', 't', 'v']]
    if flow.duplicated(['partner_hs6', 't']).any():
        raise ValueError('관측 표본에 partner_hs6-연도 중복이 있다')

    bal = pairs.merge(pd.DataFrame({'t': years}), how='cross')
    bal = bal.merge(flow, on=['partner_hs6', 't'], how='left', validate='one_to_one')
    bal['v'] = bal['v'].fillna(0.0)
    bal = bal.merge(cohort, on='hs6', how='left', validate='many_to_one')
    if bal['g'].isna().any():
        raise ValueError(f"{bal['g'].isna().sum():,}칸에 코호트가 없다")

    # 비연합국 단위는 제재 대상이 아니므로 never-treated로 둔다
    bal.loc[bal['coalition_tier1'] == 0, 'g'] = 0
    bal['g'] = bal['g'].astype(int)

    bal['y'] = np.log(bal['v'] + 1.0)
    bal['id'] = pd.factorize(bal['partner_hs6'])[0] + 1

    zero_share = (bal['v'] == 0).mean() * 100
    prt(f'  행 {len(bal):,}, 0 비중 {zero_share:.1f}%')
    for g, n in sorted(bal.groupby('g')['id'].nunique().items()):
        tag = 'never-treated' if g == 0 else f'{g} 코호트'
        prt(f'    {tag:16s}: 단위 {n:,}개')
    return bal


def write(bal, path, cols=('id', 't', 'y', 'g', 'coalition_tier1', 'hs6', 'partner')):
    """hs6·partner를 버리면 HS6 수준 클러스터 표준오차도
    영향도 분석도 할 수 없다. 처치는 HS6 수준인데 단위는 partner×HS6이므로
    두 열을 보존한다. did는 추가 열을 무시하므로 25번 결과는 바뀌지 않는다."""
    out = bal[list(cols)].sort_values(['id', 't'])
    # did는 균형 패널을 요구한다 — 단위마다 연도 수가 같은지 확인
    per_unit = out.groupby('id')['t'].size()
    if per_unit.nunique() != 1:
        raise ValueError(f'균형 패널이 아니다: 단위당 연도 수 {sorted(per_unit.unique())[:5]}...')
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    prt(f'  -> {path}  ({len(out):,}행, 단위당 {per_unit.iloc[0]}개 연도)')


def main():
    t0 = time.time()
    prt('=' * 74)
    prt('A-3 준비: Callaway-Sant\'Anna 패널 내보내기 (EJPE 2차 revision)')
    prt('=' * 74)

    panel = pd.read_parquet(
        PANEL_PATH,
        columns=['t', 'v', 'direction', 'partner', 'hs6', 'partner_hs6',
                 'sanctioned', 'sanction_year', 'coalition_tier1'])
    prt(f'패널: {len(panel):,}행')

    cohort = product_cohort(panel)
    counts = cohort['g'].value_counts().sort_index()
    prt('HS6 코드 수: ' + ', '.join(
        f'{"비제재" if g == 0 else g}={n:,}' for g, n in counts.items()))

    to_rus = panel[panel['direction'] == 'to_russia'].copy()
    prt(f'-> 러시아: {len(to_rus):,}행 '
        f'(연합국 {int((to_rus["coalition_tier1"] == 1).sum()):,})')

    # (1) 주 사양 — 연합국만, 전 기간
    coal = to_rus[to_rus['coalition_tier1'] == 1]
    bal_coal = build_balanced(coal, cohort, YEARS, '주 사양: 연합국 -> 러시아, 2015-2024')
    write(bal_coal, OUT_DIR / 'csdid_panel_coalition.csv')

    # (2) 절단 표본 — 연합국만, 2016/2023 코호트 제품 제외, 2017-2022
    keep_hs6 = set(cohort.loc[~cohort['g'].isin([int(c) for c in TRUNC_DROP_COHORTS]), 'hs6'])
    coal_tr = coal[coal['hs6'].isin(keep_hs6)]
    prt(f'\n절단 표본용 제품: {len(keep_hs6):,}개 HS6 '
        f'(2016/2023 코호트 {len(cohort) - len(keep_hs6)}개 제외)')
    bal_tr = build_balanced(coal_tr, cohort, TRUNC_YEARS,
                            'Truncated sample (literal window): 2017-2022, one post-treatment year')
    write(bal_tr, OUT_DIR / 'csdid_panel_truncated.csv')

    bal_tr_ext = build_balanced(coal_tr, cohort, TRUNC_YEARS_EXT,
                                '절단 표본(창 연장): 2017-2024, 처치후 3년')
    write(bal_tr_ext, OUT_DIR / 'csdid_panel_truncated_ext.csv')

    # (3) 보조 사양 — 비연합국 포함
    if SAMPLE == 'both':
        bal_all = build_balanced(to_rus, cohort, YEARS,
                                 '보조 사양: 전 수출국 -> 러시아, 2015-2024')
        write(bal_all, OUT_DIR / 'csdid_panel_all.csv')

    prt(f'\n총 {(time.time() - t0) / 60:.1f}분')
    prt('다음: Rscript phase4_analysis/code/25_csdid.R')
    flush_log()


if __name__ == '__main__':
    try:
        main()
    finally:
        flush_log()
