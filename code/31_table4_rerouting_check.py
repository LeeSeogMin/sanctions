"""Reproduction check for the category-level third-country estimates.

The category-level third-country figures come from a specification with no
product-group control: ln_v ~ sanctioned x post | partner x HS6 + partner x
year. The main supplier-replacement specification adds HS2 x year fixed
effects, and the paper already reports that removing that control raises the
aggregate estimate. The category figures therefore rest on the weaker
specification, and this script establishes by how much.

The originally intended HS6 x year control cannot be used here: in the
third-country sample treatment is constant within an HS6 x year cell and is
absorbed completely.

  R0  reproduce the reported specification (gate)
  R1  re-estimate adding HS2 x year fixed effects
  R2  re-estimate excluding China, which the other diagnostics identify as the
      single influential partner

Outputs: results/b31_table4_check.csv, results/b31_table4_log.txt
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

PANEL_PATH = Path('phase4_analysis/processed/baci_russia_panel.parquet')
RESULTS_DIR = Path('phase4_analysis/results')
LOG_PATH = RESULTS_DIR / 'b31_table4_log.txt'
CSV_PATH = RESULTS_DIR / 'b31_table4_check.csv'

RUSSIA = 643
CHINA = 156
EXPECT_VERSION = '0.40.1'

# Table 4에 실린 값 (21_save_reported_aggregates.py:308)
REPORTED = {'industrial_cap': 47.7, 'dual_use': 28.2,
            'military_tech': 27.0, 'luxury': -17.1}
TOL_PP = 2.0          # 재현 허용 오차 (21번이 쓰던 값과 동일)
CONTROL_CATS = ['non_sanctioned', 'not_in_eu_list']

_log = []


def prt(msg=''):
    print(msg, flush=True)
    _log.append(msg)


def flush_log():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text('\n'.join(_log) + '\n')


def fit(sub, fe, label):
    t0 = time.time()
    m = pf.feols(f'ln_v ~ sanc_post | {fe}', data=sub,
                 vcov={'CRV1': 'partner_hs6'})
    b = m.coef()['sanc_post']
    se = m.se()['sanc_post']
    p = m.pvalue()['sanc_post']
    pct = (np.exp(b) - 1) * 100
    stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    prt(f'    {label:<26} beta={b:+.4f} ({se:.4f}){stars:<3} '
        f'pct={pct:+.1f}%  N={m._N:,}  [{time.time() - t0:.0f}s]')
    return dict(beta=round(b, 4), se=round(se, 4), pval=p,
                pct=round(pct, 1), n=int(m._N))


def main():
    t_start = time.time()
    prt('=' * 74)
    prt('중단 기준 5번 확인: Table 4 카테고리 우회값 재현')
    prt('=' * 74)
    prt(f'pyfixest {pf.__version__} (기대 {EXPECT_VERSION})')
    if pf.__version__ != EXPECT_VERSION:
        prt('  ⚠️ 버전이 다르다 — 재현 게이트가 의미를 잃을 수 있다')

    if not PANEL_PATH.exists():
        prt(f'입력 없음: {PANEL_PATH}')
        return 1

    panel = pd.read_parquet(
        PANEL_PATH,
        columns=['t', 'v', 'direction', 'partner', 'hs6', 'partner_hs6',
                 'partner_year', 'category', 'coalition_tier1'])
    prt(f'패널 {len(panel):,}행')
    need = ['partner_hs6', 'partner_year', 'category']
    missing = [c for c in need if c not in panel.columns]
    if missing:
        prt(f'  패널에 열이 없다: {missing}')
        return 1

    # 21번과 동일한 표본 구성
    panel['is_third'] = ((panel['coalition_tier1'] == 0) &
                         (panel['partner'] != RUSSIA)).astype(int)
    panel['ln_v'] = np.log(panel['v'] + 1)
    panel['post'] = (panel['t'] >= 2022).astype(int)
    panel['hs2_year'] = (panel['hs6'].astype(str).str.zfill(6).str[:2]
                         + '_' + panel['t'].astype(str))

    to_rus = panel[(panel['direction'] == 'to_russia') &
                   (panel['is_third'] == 1)].copy()
    grp = to_rus.groupby('partner_hs6')['t'].nunique()
    to_rus = to_rus[to_rus['partner_hs6'].isin(grp[grp > 1].index)].copy()
    prt(f'제3국 -> 러시아 (singleton 제거 후) {len(to_rus):,}행, '
        f'파트너 {to_rus["partner"].nunique()}개')

    rows, failures = [], []
    for cat, expected in REPORTED.items():
        prt('')
        prt('-' * 74)
        prt(f'[{cat}]  Table 4 보고값 {expected:+.1f}%')
        prt('-' * 74)
        sub = to_rus[(to_rus['category'] == cat) |
                     (to_rus['category'].isin(CONTROL_CATS))].copy()
        sub['sanc_post'] = ((sub['category'] == cat) &
                            (sub['post'] == 1)).astype(int)
        if sub['sanc_post'].sum() == 0:
            prt('  처치 행이 없다 — 건너뜀')
            failures.append(f'{cat}: 처치 행 0')
            continue

        r0 = fit(sub, 'partner_hs6 + partner_year', 'R0 보고 사양')
        gap = abs(r0['pct'] - expected)
        ok = gap <= TOL_PP
        prt(f'    재현 게이트: |{r0["pct"]:+.1f} - {expected:+.1f}| = {gap:.2f}%p '
            f'(허용 {TOL_PP}) {"[OK]" if ok else "[FAIL]"}')
        if not ok:
            failures.append(f'{cat}: 재현 실패 ({r0["pct"]:+.1f} vs {expected:+.1f})')

        r1 = fit(sub, 'partner_hs6 + partner_year + hs2_year',
                 'R1 +hs2_year (주 사양)')
        sub_nc = sub[sub['partner'] != CHINA]
        r2 = fit(sub_nc, 'partner_hs6 + partner_year', 'R2 중국 제외')

        prt(f'    제품군 통제 효과: {r0["pct"]:+.1f}% -> {r1["pct"]:+.1f}% '
            f'({r1["pct"] - r0["pct"]:+.1f}%p)')
        prt(f'    중국 제외 효과:   {r0["pct"]:+.1f}% -> {r2["pct"]:+.1f}% '
            f'({r2["pct"] - r0["pct"]:+.1f}%p)')

        for name, r in [('R0_reported_spec', r0), ('R1_plus_hs2year', r1),
                        ('R2_excl_china', r2)]:
            rows.append(dict(category=cat, variant=name, reported_pct=expected,
                             **r))

    if rows:
        pd.DataFrame(rows).to_csv(CSV_PATH, index=False)

    prt('')
    prt('=' * 74)
    prt('SUMMARY')
    prt('=' * 74)
    df = pd.DataFrame(rows)
    for cat in REPORTED:
        s = df[df['category'] == cat]
        if s.empty:
            continue
        g = {r['variant']: r['pct'] for _, r in s.iterrows()}
        prt(f'  {cat:<16} 보고 {REPORTED[cat]:+6.1f}%  |  '
            f'R0 {g.get("R0_reported_spec", float("nan")):+6.1f}%  '
            f'R1(+hs2_year) {g.get("R1_plus_hs2year", float("nan")):+6.1f}%  '
            f'R2(중국제외) {g.get("R2_excl_china", float("nan")):+6.1f}%')

    prt('')
    if failures:
        prt(f'  재현 실패 {len(failures)}건:')
        for f in failures:
            prt(f'    - {f}')
        prt('  -> 중단 기준 5번에 걸린다. Table 4 우회 열을 삭제해야 한다.')
    else:
        prt('  네 카테고리 모두 재현됐다 -> 중단 기준 5번에는 걸리지 않는다.')
        prt('  다만 R1/R2가 크게 움직이면 "재현은 되나 명세가 약하다"를 밝혀야 한다:')
        prt('    · R1은 본문 주 사양과 같은 제품군 통제를 넣은 값이다.')
        prt('      Table 4가 R0(제품군 통제 없음)를 싣는 것은 본문 +9.6%와 불일치다.')
        prt('    · R2는 A-5·A-7에서 확인된 중국 단일 의존이 카테고리별로도')
        prt('      나타나는지 본다.')

    prt('')
    prt(f'총 {(time.time() - t_start) / 60:.1f}분')
    prt(f'저장: {CSV_PATH}')
    return 1 if failures else 0


if __name__ == '__main__':
    # 예외가 나면 code가 정의되지 않아 로그에 실패가 남지 않았다.
    # 예외를 잡아 로그에 적고 nonzero로 끝낸다.
    code = 1
    try:
        code = main()
    except Exception as exc:                      # noqa: BLE001
        import traceback
        prt('')
        prt('=' * 74)
        prt(f'예외로 중단: {type(exc).__name__}: {exc}')
        prt('=' * 74)
        prt(traceback.format_exc())
        code = 1
    finally:
        flush_log()
    raise SystemExit(code)
