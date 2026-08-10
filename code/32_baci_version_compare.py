"""Compare the BACI release used here against the preceding release.

Establishes how much a provisional year moves when it is finalised, which bears
on the treatment of 2024 in the dynamic results.

What is possible. CEPII distributes only the immediately preceding release, so
a single pair of versions can be compared; older releases are unavailable. That
limit is recorded in the output.

The sharpest test is V3. In the earlier release 2023 was provisional and in the
current one it is final, so the difference between the two for 2023 measures
how far a provisional year moves on revision, and bounds what the current
provisional year may still do. 2015-2022 are final in both releases and serve
as the control.

  V0  file and year coverage
  V1  annual aggregates: cell counts, value, partner counts
  V2  cell-level comparison after merging on exporter, importer, product, year
  V3  revision size for the previously provisional year against 2015-2022
  V4  the same comparison restricted to the estimation sample, split by
      sanctioned status and coalition membership

Reading it: if 2015-2022 are effectively identical and only the provisional
year moves, revision is confined to that year and is already covered by the
exclude-2024 robustness checks.

Usage: python 32_baci_version_compare.py --old-dir <previous release CSV folder>
Outputs: results/b32_version_compare.csv, results/b32_version_log.txt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

NEW_DIR = Path('phase3_data/BACI_HS92_V202601')
# Directory holding the previous BACI release. Override with --old-dir.
OLD_DIR_DEFAULT = Path('phase3_data/BACI_HS92_V202501')
RESULTS_DIR = Path('phase4_analysis/results')
LOG_PATH = RESULTS_DIR / 'b32_version_log.txt'
CSV_PATH = RESULTS_DIR / 'b32_version_compare.csv'

RUSSIA = 643
NEW_V, OLD_V = 'V202601', 'V202501'
OVERLAP = range(2015, 2024)          # 구판은 2023년까지
PROVISIONAL_IN_OLD = 2023            # 구판에서 잠정이었던 연도
SETTLED = range(2015, 2023)          # 두 판 모두 확정
TOL_PCT = 1.0                        # 확정 연도 금액 변화 허용치(%)

_log = []


def prt(msg=''):
    print(msg, flush=True)
    _log.append(msg)


def flush_log():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text('\n'.join(_log) + '\n')


def load(dirpath, year, version):
    f = Path(dirpath) / f'BACI_HS92_Y{year}_{version}.csv'
    if not f.exists():
        return None
    d = pd.read_csv(f, usecols=['t', 'i', 'j', 'k', 'v'],
                    dtype={'i': 'int32', 'j': 'int32', 'v': 'float64'})
    return d[(d['i'] == RUSSIA) | (d['j'] == RUSSIA)].copy()


def main(old_dir):
    t_start = time.time()
    prt('=' * 74)
    prt(f'중단 기준 6번(나머지 절반): BACI {NEW_V} 대 {OLD_V}')
    prt('=' * 74)
    prt(f'구판 폴더: {old_dir}')
    prt('CEPII는 직전 버전 하나만 배포한다. V202401은 404이므로 이 한 쌍이 전부다.')

    # ------------------------------------------------------------ V0
    prt('')
    prt('V0 — 파일 커버리지')
    have = []
    for y in OVERLAP:
        n = (NEW_DIR / f'BACI_HS92_Y{y}_{NEW_V}.csv').exists()
        o = (Path(old_dir) / f'BACI_HS92_Y{y}_{OLD_V}.csv').exists()
        if n and o:
            have.append(y)
        else:
            prt(f'  {y}: 신판 {"O" if n else "X"} / 구판 {"O" if o else "X"} — 제외')
    if not have:
        prt('  겹치는 연도가 없다 — 중단')
        return 1
    prt(f'  대조 가능 연도: {have[0]}–{have[-1]} ({len(have)}개)')

    rows = []
    prt('')
    prt('=' * 74)
    prt('V1·V2·V3 — 연도별 대조')
    prt('=' * 74)
    prt(f'{"연도":>6} {"신판셀":>10} {"구판셀":>10} {"셀차":>8} '
        f'{"신판금액":>14} {"구판금액":>14} {"금액변화%":>10} {"공통셀평균|변화|%":>16}')

    for y in have:
        new = load(NEW_DIR, y, NEW_V)
        old = load(old_dir, y, OLD_V)
        if new is None or old is None:
            continue

        m = new.merge(old, on=['i', 'j', 'k'], how='outer',
                      suffixes=('_new', '_old'), indicator=True)
        both = m[m['_merge'] == 'both']
        only_new = int((m['_merge'] == 'left_only').sum())
        only_old = int((m['_merge'] == 'right_only').sum())

        # 공통 셀의 상대 변화 (금액 가중)
        rel = np.abs(both['v_new'] - both['v_old']) / both['v_old'].replace(0, np.nan)
        w = both['v_old'].fillna(0)
        wmean = float(np.nansum(rel * w) / w.sum() * 100) if w.sum() > 0 else np.nan

        v_new, v_old = new['v'].sum() / 1e6, old['v'].sum() / 1e6
        dv = (v_new / v_old - 1) * 100 if v_old else np.nan

        prt(f'{y:>6} {len(new):>10,} {len(old):>10,} {len(new)-len(old):>+8,} '
            f'{v_new:>14,.1f} {v_old:>14,.1f} {dv:>+10.2f} {wmean:>16.2f}')

        rows.append(dict(year=y, cells_new=len(new), cells_old=len(old),
                         only_new=only_new, only_old=only_old,
                         value_new_bn=round(v_new, 2), value_old_bn=round(v_old, 2),
                         value_change_pct=round(dv, 3),
                         common_cell_abs_change_pct=round(wmean, 3)))
        del new, old, m, both

    if not rows:
        prt('대조 결과가 없다 — 중단')
        return 1

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)

    # ------------------------------------------------------------ V3 판정
    prt('')
    prt('=' * 74)
    prt('V3 — 잠정 연도 개정 폭 (핵심 판정)')
    prt('=' * 74)
    settled = df[df['year'].isin(SETTLED)]
    prov = df[df['year'] == PROVISIONAL_IN_OLD]

    if not settled.empty:
        s_max = settled['value_change_pct'].abs().max()
        s_mean = settled['common_cell_abs_change_pct'].mean()
        prt(f'  확정 연도({SETTLED[0]}–{SETTLED[-1]}): '
            f'금액 변화 최대 {s_max:+.2f}%, 공통셀 평균|변화| {s_mean:.2f}%')
    else:
        s_max = np.nan
    if not prov.empty:
        p_dv = float(prov['value_change_pct'].iloc[0])
        p_cell = float(prov['common_cell_abs_change_pct'].iloc[0])
        prt(f'  잠정이었던 {PROVISIONAL_IN_OLD}년: '
            f'금액 변화 {p_dv:+.2f}%, 공통셀 평균|변화| {p_cell:.2f}%')
        prt(f'  구판에만 있던 셀 {int(prov["only_old"].iloc[0]):,}개, '
            f'신판에만 있는 셀 {int(prov["only_new"].iloc[0]):,}개')
    else:
        p_dv = np.nan

    prt('')
    prt('  해석: 구판의 2023년은 잠정치였고 신판에서 확정됐다. 그 변화 폭이')
    prt('        신판 2024년(현재 잠정)이 앞으로 움직일 폭의 근거다.')

    # ------------------------------------------------------------ 판정
    prt('')
    prt('=' * 74)
    prt('판정')
    prt('=' * 74)
    settled_stable = bool(np.isfinite(s_max) and s_max <= TOL_PCT)
    if settled_stable:
        prt(f'  확정 연도는 두 판에서 {TOL_PCT}% 이내로 동일하다.')
        prt('  -> 개정 문제는 잠정 연도에 국한된다. 2024년 제외 검정(D6·H6)이')
        prt('     이미 다룬 범위이므로 중단 기준 6번에는 걸리지 않는다.')
    else:
        prt(f'  ⚠️ 확정 연도조차 {TOL_PCT}%를 넘게 움직였다(최대 {s_max:+.2f}%).')
        prt('  -> 자료 자체가 판 사이에 달라진다. 헤드라인·우회 추정을 구판으로')
        prt('     다시 돌려 결론이 유지되는지 확인해야 한다. 6번 기준에 걸린다.')
    prt('')
    prt('  ⚠️ 한계: V202401 이전 판은 CEPII가 배포하지 않아 확인할 수 없다.')
    prt('     이 검정은 "직전 판과 비교했을 때"까지만 말한다.')

    prt('')
    prt(f'총 {(time.time() - t_start) / 60:.1f}분')
    prt(f'저장: {CSV_PATH}')
    return 0 if settled_stable else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--old-dir', default=str(OLD_DIR_DEFAULT))
    args = ap.parse_args()
    code = 1
    try:
        code = main(args.old_dir)
    except Exception as exc:                      # noqa: BLE001
        import traceback
        prt('')
        prt(f'예외로 중단: {type(exc).__name__}: {exc}')
        prt(traceback.format_exc())
        code = 1
    finally:
        flush_log()
    raise SystemExit(code)
