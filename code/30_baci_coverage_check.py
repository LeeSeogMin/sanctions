"""Post-2022 coverage of Russia-related flows in BACI.

Supplies the data-coverage limitation reported in Section 6.4. If Russia
stopped publishing customs statistics, post-2022 Russia-facing flows rest on
partner reports alone. The supplier-replacement sample is built from
non-coalition exports to Russia, so if the weaker reporters among those
countries drop out, the sample changes at the same time as treatment.

What the script establishes, taking the three together
  1  whether the worldwide count of reporting exporters changed over the same
     period, which separates a Russia-specific problem from a BACI-wide one
  2  whether trade value fell alongside the partner count; a falling count with
     stable value indicates missing observations rather than ceased trade
  3  whether the departing partners are coalition or non-coalition members,
     which fixes the direction of any bias

What it cannot establish
  Whether Russia stopped publishing. The BACI release carries no reporter
  information and the distributed files do not record which side reported a
  given value. The script can show that the observed patterns are consistent
  with a reporting break, and no more.

Output: results/b30_coverage_log.txt
"""

import time
from pathlib import Path

import pandas as pd

BACI_DIR = Path('phase3_data/BACI_HS92_V202601')
PANEL = Path('phase4_analysis/processed/baci_russia_panel.parquet')
LOG_PATH = Path('phase4_analysis/results/b30_coverage_log.txt')
RUSSIA = 643
YEARS = range(2015, 2025)

_log = []


def prt(msg=''):
    print(msg, flush=True)
    _log.append(msg)


def flush_log():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text('\n'.join(_log) + '\n')


def main():
    t0 = time.time()
    prt('=' * 74)
    prt('A.8 확인: 2022년 이후 러시아 관련 BACI 커버리지')
    prt('=' * 74)

    # ---------------------------------------------------------- ①② 원자료
    prt('')
    prt('① 전세계 보고국 수 대조 + ② 파트너 수 대 금액')
    prt('   전세계 수출국 수가 불변인데 러시아 파트너만 줄면 러시아 자료 문제다.')
    prt('')
    prt(f'{"연도":>6} {"대러수출국":>10} {"대러수출액(십억$)":>18} '
        f'{"러수출상대국":>12} {"러수출액(십억$)":>17} {"전세계수출국":>12}')

    rows = []
    for y in YEARS:
        f = BACI_DIR / f'BACI_HS92_Y{y}_V202601.csv'
        if not f.exists():
            prt(f'  {y}: 파일 없음 — 건너뜀')
            continue
        d = pd.read_csv(f, usecols=['i', 'j', 'v'],
                        dtype={'i': 'int32', 'j': 'int32'})
        to_ru = d[d['j'] == RUSSIA]
        fr_ru = d[d['i'] == RUSSIA]
        r = dict(t=y,
                 to_partners=to_ru['i'].nunique(),
                 to_value_bn=to_ru['v'].sum() / 1e6,
                 fr_partners=fr_ru['j'].nunique(),
                 fr_value_bn=fr_ru['v'].sum() / 1e6,
                 world_exporters=d['i'].nunique())
        rows.append(r)
        prt(f'{y:>6} {r["to_partners"]:>10,} {r["to_value_bn"]:>18,.1f} '
            f'{r["fr_partners"]:>12,} {r["fr_value_bn"]:>17,.1f} '
            f'{r["world_exporters"]:>12,}')
        del d

    cov = pd.DataFrame(rows)
    if cov.empty:
        prt('원자료를 하나도 읽지 못했다 — 중단')
        return 1

    world_const = cov['world_exporters'].nunique() == 1
    prt('')
    prt(f'  전세계 수출국 수: {"모든 연도 동일 (" + str(cov["world_exporters"].iloc[0]) + "개)" if world_const else "연도별로 변동 — 아래 판정 보류"}')

    pre = cov[cov['t'].between(2019, 2021)]
    post = cov[cov['t'].between(2022, 2024)]
    if len(pre) and len(post):
        prt(f'  대러수출국  {pre["to_partners"].mean():.0f} -> {post["to_partners"].mean():.0f}'
            f'  ({100 * (post["to_partners"].mean() / pre["to_partners"].mean() - 1):+.0f}%)')
        prt(f'  러시아수출 상대국  {pre["fr_partners"].mean():.0f} -> {post["fr_partners"].mean():.0f}'
            f'  ({100 * (post["fr_partners"].mean() / pre["fr_partners"].mean() - 1):+.0f}%)')
        prt(f'  러시아 수출액  {pre["fr_value_bn"].mean():,.1f} -> {post["fr_value_bn"].mean():,.1f} 십억$'
            f'  ({100 * (post["fr_value_bn"].mean() / pre["fr_value_bn"].mean() - 1):+.0f}%)')
        prt('  -> 상대국 수는 줄고 금액은 유지·증가하면 무역 중단이 아니라 관측 누락이다.')

    # ---------------------------------------------------------- ③ 누가 사라졌나
    prt('')
    prt('=' * 74)
    prt('③ 사라진 파트너는 연합국인가 비연합국인가')
    prt('=' * 74)
    if not PANEL.exists():
        prt(f'  패널 없음: {PANEL} — ③ 건너뜀')
    else:
        d = pd.read_parquet(PANEL,
                            columns=['t', 'v', 'direction', 'partner',
                                     'coalition_tier1'])
        s = d[d['direction'] == 'to_russia']
        pre_p = set(s[s['t'].between(2019, 2021)]['partner'].unique())
        post_p = set(s[s['t'].between(2022, 2024)]['partner'].unique())
        gone = pre_p - post_p
        coal = s.drop_duplicates('partner').set_index('partner')['coalition_tier1'].to_dict()
        n_coal = sum(1 for p in gone if coal.get(p) == 1)

        prt(f'  2019-2021에 있다가 2022-2024에 완전히 사라진 파트너: {len(gone)}개')
        prt(f'    연합국 {n_coal}개 / 비연합국 {len(gone) - n_coal}개')

        v21 = s[s['t'] == 2021]
        share = 100 * v21[v21['partner'].isin(gone)]['v'].sum() / v21['v'].sum()
        prt(f'  사라진 파트너의 2021년 대러 수출액 비중: {share:.2f}%')
        prt('  -> 수는 많고 금액은 작다. 금액 기반 지표(누출 비율)보다')
        prt('     개수·외연 마진 지표가 직접 영향을 받는다.')

        byg = s.groupby(['t', 'coalition_tier1'])['partner'].nunique().unstack()
        byg.columns = ['비연합국' if c == 0 else '연합국' for c in byg.columns]
        prt('')
        prt('  연도별 파트너 수:')
        for t, r in byg.iterrows():
            prt(f'    {t}  비연합국 {int(r["비연합국"]):>4}  연합국 {int(r["연합국"]):>3}')

    prt('')
    prt('=' * 74)
    prt('판정 지침')
    prt('=' * 74)
    prt('  전세계 수출국 수가 불변 + 금액은 유지·증가 + 양방향 대칭이면,')
    prt('  러시아 측 보고 중단과 정합적이다. 다만 그 외부 사실 자체는 이 자료로')
    prt('  확인할 수 없다 — BACI 최종 파일은 어느 쪽이 신고했는지 기록하지 않는다.')
    prt('  Report as consistent-with; do not state it as established.')
    prt('')
    prt(f'총 {(time.time() - t0) / 60:.1f}분')
    prt(f'저장: {LOG_PATH}')
    return 0


if __name__ == '__main__':
    try:
        code = main()
    finally:
        flush_log()
    raise SystemExit(code)
