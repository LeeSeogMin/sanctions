#!/usr/bin/env python3
"""Upstream coalition-to-gateway test, Equation (5), by OLS and PPML.

Supplies the gateway rows of Table 6 and Online Appendix I. The sample follows
the earlier gateway construction, and every specification is estimated twice,
once by OLS and once by PPML.

  G1  OLS  coalition -> gateway
  G3  OLS  coalition -> non-gateway, placebo
  G5  OLS  pooled gateway differential, without importer x year fixed effects
  ---- all three must fall inside tolerance before anything below runs ----
  G2  PPML coalition -> gateway
  G4  PPML coalition -> non-gateway
  G6  PPML pooled differential, without importer x year
  G7  OLS  pooled differential, with importer x year   <- Equation (5)
  G9  OLS  as G7, staggered treatment
  G8  PPML as G7
  G10 PPML as G9

The importer x year fixed effects matter. Without them, gateway countries'
general post-2022 import growth loads onto the sanctioned-product interaction;
G5 and G6 are retained only to show the size of that effect. G7 to G10 are the
specifications reported in the paper.

Reproduction gate: the three OLS rows are estimated first and compared with
their expected values. If any is outside tolerance the script exits without
running PPML, so that a plausible-looking coefficient from the wrong sample
cannot reach the results.

BACI is read year by year and cached to parquet with a metadata sidecar; the
cache rebuilds automatically when the sample definition changes (--rebuild
forces it).

Outputs
  results/b26_gateway_ppml.csv, results/b26_gateway_log.txt
  processed/gateway_panel.parquet (+ .meta.json)
"""

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

BACI_DIR = Path('phase3_data/BACI_HS92_V202601')
SANCTIONS_MASTER = Path('phase4_analysis/processed/sanctions_hs6_master.csv')
PROCESSED_DIR = Path('phase4_analysis/processed')
RESULTS_DIR = Path('phase4_analysis/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = PROCESSED_DIR / 'gateway_panel.parquet'
CACHE_META = PROCESSED_DIR / 'gateway_panel.meta.json'
LOG_PATH = RESULTS_DIR / 'b26_gateway_log.txt'
CSV_PATH = RESULTS_DIR / 'b26_gateway_ppml.csv'

CACHE_VERSION = 3  # 표본 구성 로직을 바꾸면 올린다 (v3: sanc_stag_gw 추가)

EU27 = {40, 56, 100, 191, 196, 203, 208, 233, 246, 251, 276, 300,
        348, 372, 380, 428, 440, 442, 470, 528, 616, 620, 642,
        703, 705, 724, 752}
TIER1_NON_EU = {826, 842, 124, 392, 579, 757}
TIER1 = EU27 | TIER1_NON_EU
RUSSIA = 643

# 관문국 10개 (B-4 exposure 분석). India는 699 (2026-06-08 코드 교정 반영)
GATEWAY = {784, 51, 417, 792, 156, 860, 398, 762, 268, 699}

YEARS = list(range(2015, 2025))
VCOV_KEY = 'exporter_importer_hs6'
REQUIRED_COLS = ['v', 'ln_v', 'sanc_post', 'sanc_post_gw', 'sanc_stag', 'sanc_stag_gw',
                 'is_gateway', 't',
                 'exporter_importer_hs6', 'exporter_year', 'importer_year', 'hs6_year']

# 재현 허용 오차 (퍼센트포인트)
REPLICATION_TOL = 0.5

FEPOIS_KWARGS = dict(iwls_tol=1e-8, iwls_maxiter=100,
                     fixef_tol=1e-8, fixef_maxiter=100_000,
                     separation_check=['fe'])

_log_lines = []


def prt(msg):
    print(msg, flush=True)
    _log_lines.append(str(msg))


def flush_log():
    LOG_PATH.write_text('\n'.join(_log_lines), encoding='utf-8')


def checkpoint(results):
    pd.DataFrame(results).to_csv(CSV_PATH, index=False)
    flush_log()


def get_nobs(m):
    for attr in ('nobs', '_N'):
        try:
            return getattr(m, attr)
        except AttributeError:
            continue
    try:
        return len(m._Y)
    except AttributeError:
        return -1


def stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return ''


def supported_fepois_kwargs():
    """이 pyfixest 버전이 실제로 받는 인자만 남긴다.

    TypeError를 잡아 폴백하면 fepois 내부의 진짜 TypeError까지 삼키게 되므로,
    호출 전에 시그니처를 확인한다.
    """
    try:
        params = inspect.signature(pf.fepois).parameters
    except (TypeError, ValueError):
        prt('    (fepois 시그니처를 읽지 못함 -> 기본값 사용)')
        return {}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(FEPOIS_KWARGS)
    ok = {k: v for k, v in FEPOIS_KWARGS.items() if k in params}
    dropped = sorted(set(FEPOIS_KWARGS) - set(ok))
    if dropped:
        prt(f'    (이 pyfixest 버전이 받지 않는 인자 제외: {dropped})')
    return ok


def cache_signature():
    return {
        'cache_version': CACHE_VERSION,
        'tier1': sorted(TIER1),
        'gateway': sorted(GATEWAY),
        'years': YEARS,
        'sanctions_master': str(SANCTIONS_MASTER),
        'baci_dir': str(BACI_DIR),
    }


def build_panel():
    """BACI에서 연합→관문국 / 연합→비관문국 흐름을 만든다 (10b와 동일 절차)."""
    prt('=== Building gateway panel from BACI ===')

    missing = [y for y in YEARS
               if not (BACI_DIR / f'BACI_HS92_Y{y}_V202601.csv').exists()]
    if missing:
        raise FileNotFoundError(
            f'BACI years missing: {missing}. 표본이 달라지므로 중단한다.')

    sanctions = pd.read_csv(SANCTIONS_MASTER)
    sanctions['hs6'] = sanctions['hs6'].astype(str).str.zfill(6)
    sanctioned_set = set(sanctions.loc[sanctions['sanctioned'] == 1, 'hs6'])
    sanc_year_map = (sanctions.loc[sanctions['sanctioned'] == 1]
                     .set_index('hs6')['sanction_year'].to_dict())
    prt(f'  sanctioned HS6 codes: {len(sanctioned_set):,}')

    # 비관문국 위약군: 2019년 연합 수출액 기준 상위 10개 (10b:72-80과 동일)
    prt('  identifying top-10 non-gateway destinations (2019)...')
    df19 = pd.read_csv(BACI_DIR / 'BACI_HS92_Y2019_V202601.csv', dtype={'k': str})
    coal19 = df19[df19['i'].isin(TIER1)]
    targets = set(coal19['j'].unique()) - TIER1 - GATEWAY - {RUSSIA}
    non_gateway = set(coal19[coal19['j'].isin(targets)]
                      .groupby('j')['v'].sum().sort_values(ascending=False)
                      .head(10).index)
    prt(f'  non-gateway placebo set: {sorted(non_gateway)}')
    del df19, coal19
    flush_log()

    frames = []
    for year in YEARS:
        t0 = time.time()
        df = pd.read_csv(BACI_DIR / f'BACI_HS92_Y{year}_V202601.csv', dtype={'k': str})
        df = df[df['i'].isin(TIER1)]
        for dest, flag in ((GATEWAY, 1), (non_gateway, 0)):
            sub = df[df['j'].isin(dest)].copy()
            sub['hs6'] = sub['k'].str.zfill(6)
            agg = sub.groupby(['i', 'j', 'hs6'], as_index=False).agg(v=('v', 'sum'))
            agg['t'] = np.int16(year)
            agg['is_gateway'] = np.int8(flag)
            agg.rename(columns={'i': 'exporter', 'j': 'importer'}, inplace=True)
            frames.append(agg)
        prt(f'  {year}: done ({time.time()-t0:.0f}s)')
        flush_log()
        del df

    panel = pd.concat(frames, ignore_index=True)
    del frames

    panel['sanctioned'] = panel['hs6'].isin(sanctioned_set).astype('int8')
    # 10b:121-124와 동일한 처리: 결측을 0으로 채우고 sanction_year > 0을 요구
    sanction_year = panel['hs6'].map(sanc_year_map).fillna(0).astype('int32')
    panel['sanc_stag'] = (panel['sanctioned'].eq(1)
                          & sanction_year.gt(0)
                          & panel['t'].ge(sanction_year)).astype('int8')
    panel['sanc_post'] = (panel['sanctioned'] * (panel['t'] >= 2022)).astype('int8')
    panel['sanc_post_gw'] = (panel['sanc_post'] * panel['is_gateway']).astype('int8')
    # staggered 대응항. sanc_stag는 v2까지 만들어 놓고 한 번도 쓰지 않았다
    # This is where staggered treatment timing enters; used by G9/G10.
    panel['sanc_stag_gw'] = (panel['sanc_stag'] * panel['is_gateway']).astype('int8')
    panel['ln_v'] = np.log(panel['v'] + 1)

    # 고정효과 키는 메모리 절약을 위해 정수 코드로. pyfixest는 FE를 내부적으로 정수로
    # 인코딩하므로 연속형으로 오인될 위험은 없다.
    ei_hs6 = (panel['exporter'].astype(str) + '_' + panel['importer'].astype(str)
              + '_' + panel['hs6'])
    panel['exporter_importer_hs6'] = pd.factorize(ei_hs6)[0].astype('int32')
    del ei_hs6
    panel['exporter_year'] = pd.factorize(
        panel['exporter'].astype(str) + '_' + panel['t'].astype(str))[0].astype('int32')
    panel['importer_year'] = pd.factorize(
        panel['importer'].astype(str) + '_' + panel['t'].astype(str))[0].astype('int32')
    panel['hs6_year'] = pd.factorize(
        panel['hs6'] + '_' + panel['t'].astype(str))[0].astype('int32')

    panel = panel.drop(columns=['hs6'])
    prt(f'  panel: {len(panel):,} rows '
        f'(gateway {int((panel["is_gateway"]==1).sum()):,} / '
        f'non-gateway {int((panel["is_gateway"]==0).sum()):,})')

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(CACHE_PATH, index=False)
    CACHE_META.write_text(json.dumps(cache_signature(), indent=2), encoding='utf-8')
    prt(f'  cached -> {CACHE_PATH}')
    return panel


def load_panel(rebuild=False):
    if rebuild:
        prt('=== --rebuild: ignoring cache ===')
        return build_panel()
    if not (CACHE_PATH.exists() and CACHE_META.exists()):
        return build_panel()
    try:
        meta = json.loads(CACHE_META.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        prt('=== cache metadata unreadable -> rebuilding ===')
        return build_panel()
    if meta != cache_signature():
        prt('=== cache built under different settings -> rebuilding ===')
        return build_panel()
    prt(f'=== Loading cached panel: {CACHE_PATH} ===')
    panel = pd.read_parquet(CACHE_PATH)
    missing = [c for c in REQUIRED_COLS if c not in panel.columns]
    if missing:
        prt(f'=== cached panel missing columns {missing} -> rebuilding ===')
        return build_panel()
    prt(f'  {len(panel):,} rows')
    return panel


def drop_singletons(d):
    grp = d.groupby('exporter_importer_hs6')['t'].nunique()
    return d[d['exporter_importer_hs6'].isin(grp[grp > 1].index)].copy()


def fit(fml, data, estimator, fepois_kwargs):
    if estimator == 'OLS':
        return pf.feols(fml, data=data, vcov={'CRV1': VCOV_KEY})
    return pf.fepois(fml, data=data, vcov={'CRV1': VCOV_KEY}, **fepois_kwargs)


def run_spec(data, label, estimator, treat, fe, results, fepois_kwargs, expect=None):
    y = 'v' if estimator == 'PPML' else 'ln_v'
    fml = f'{y} ~ {treat} | {fe}'
    prt(f'\n--- {label} ---')
    prt(f'    {estimator}: {fml}')
    t0 = time.time()
    required = list(dict.fromkeys([y, treat, VCOV_KEY]
                                  + [c.strip() for c in fe.split('+')]))
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(f'{label}: 필요한 열이 없다 -> {missing}')
    try:
        m = fit(fml, data[required], estimator, fepois_kwargs)
        b, se, p = m.coef()[treat], m.se()[treat], m.pvalue()[treat]
        pct = (np.exp(b) - 1) * 100
        n, el = get_nobs(m), time.time() - t0
        row = {'spec': label, 'estimator': estimator, 'treatment': treat, 'fe': fe,
               'beta': b, 'se': se, 'pval': p, 'pct': pct, 'nobs': n,
               'seconds': round(el, 1), 'expected_pct': expect}
        note = ''
        if expect is not None:
            gap = abs(pct - expect)
            row['replication_gap_pp'] = round(gap, 3)
            row['replication_ok'] = bool(gap <= REPLICATION_TOL)
            note = (f'  [expected ~{expect:+.1f}%, gap {gap:.2f}pp '
                    f'{"OK" if gap <= REPLICATION_TOL else "FAIL"}]')
        prt(f'    beta={b:+.4f} ({se:.4f}){stars(p)}  pct={pct:+.1f}%{note}  '
            f'N={n:,}  [{el:.0f}s]')
        results.append(row)
    except Exception as e:
        prt(f'    ERROR: {type(e).__name__}: {e}')
        results.append({'spec': label, 'estimator': estimator, 'treatment': treat,
                        'fe': fe, 'nobs': -1, 'seconds': round(time.time() - t0, 1),
                        'error': f'{type(e).__name__}: {e}', 'expected_pct': expect,
                        'replication_ok': False if expect is not None else None})
    checkpoint(results)
    return results[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rebuild', action='store_true',
                    help='캐시를 무시하고 BACI에서 표본을 다시 만든다')
    args = ap.parse_args()

    t_start = time.time()
    prt('=' * 74)
    prt('A-4: Coalition -> Gateway transshipment test, Eq.(5) by PPML')
    prt('=' * 74)
    flush_log()

    panel = load_panel(rebuild=args.rebuild)
    fepois_kwargs = supported_fepois_kwargs()
    results = []

    FE_PAIR = 'exporter_importer_hs6 + exporter_year + importer_year'
    FE_POOL = 'exporter_importer_hs6 + hs6_year + exporter_year'
    # G7~G10용. FE_POOL에 importer_year를 더한 것이 식(5)의 올바른 명세다.
    FE_POOL_FULL = 'exporter_importer_hs6 + hs6_year + exporter_year + importer_year'

    samples = {}
    samples['gw'] = drop_singletons(panel[panel['is_gateway'] == 1])
    samples['ngw'] = drop_singletons(panel[panel['is_gateway'] == 0])
    samples['pool'] = drop_singletons(panel)
    del panel
    for k, v in samples.items():
        prt(f'\nsample {k}: {len(v):,} rows after singleton removal')

    # ---- 1단계: OLS 재현 게이트 ----
    prt('\n' + '=' * 74)
    prt('STAGE 1 — OLS replication gate')
    prt('=' * 74)
    gate = [
        run_spec(samples['gw'], 'G1: OLS  coalition->gateway  (replication)',
                 'OLS', 'sanc_post', FE_PAIR, results, fepois_kwargs, expect=10.0),
        run_spec(samples['ngw'], 'G3: OLS  coalition->non-gateway placebo (replication)',
                 'OLS', 'sanc_post', FE_PAIR, results, fepois_kwargs, expect=6.2),
        run_spec(samples['pool'], 'G5: OLS  pooled gateway differential (replication)',
                 'OLS', 'sanc_post_gw', FE_POOL, results, fepois_kwargs, expect=22.6),
    ]
    failed = [r for r in gate if not r.get('replication_ok')]
    if failed:
        prt('\n' + '!' * 74)
        prt('REPLICATION GATE FAILED — PPML 사양을 실행하지 않는다.')
        for r in failed:
            prt(f"  {r['spec']}: expected {r.get('expected_pct')}, "
                f"got {r.get('pct', 'ERROR')}")
        prt('표본 구성이 기존과 다르다. 원인을 찾기 전에는 PPML 결과를 쓰면 안 된다.')
        prt('!' * 74)
        checkpoint(results)
        raise SystemExit(1)
    prt('\nReplication gate PASSED — proceeding to PPML.')

    # ---- 2단계: PPML ----
    prt('\n' + '=' * 74)
    prt('STAGE 2 — PPML')
    prt('=' * 74)
    run_spec(samples['gw'], 'G2: PPML coalition->gateway  (new)',
             'PPML', 'sanc_post', FE_PAIR, results, fepois_kwargs)
    del samples['gw']
    run_spec(samples['ngw'], 'G4: PPML coalition->non-gateway placebo (new)',
             'PPML', 'sanc_post', FE_PAIR, results, fepois_kwargs)
    del samples['ngw']
    run_spec(samples['pool'], 'G6: PPML pooled gateway differential (new)',
             'PPML', 'sanc_post_gw', FE_POOL, results, fepois_kwargs)

    # ---- 3단계: 식(5)를 제대로 명세한다 (importer_year FE 추가) ----
    prt('\n' + '=' * 74)
    prt('STAGE 3 — 식(5)의 올바른 명세 (+ importer_year FE)')
    prt('=' * 74)
    prt('  왜 필요한가. G5/G6의 FE에는 수입국-연도 항이 없다. 그래서')
    prt('  "2022년 이후 관문국행 연합 수출이 품목 불문 늘었다"(G_j x P_t)가 통제되지')
    prt('  않고, 처치변수 sanc_post_gw = S_k P_t G_j 가 그 안에 완전히 포개져 있다.')
    prt('  hs6_year는 제품-연도 충격을 흡수하지만 목적지-연도 충격은 흡수하지 못한다.')
    prt('')
    prt('  근거: 같은 양을 재는 추정치 넷 중 셋이 +2.7~3.6%에 모이고 G5만 +22.6%다.')
    prt('    G1-G3 = +0.0350 (+3.6%)   [FE_PAIR, importer_year 있음]')
    prt('    G2-G4 = +0.0269 (+2.7%)   [FE_PAIR, importer_year 있음]')
    prt('    G6    = +0.0341 (+3.5%)   [FE_POOL, importer_year 없음, PPML]')
    prt('    G5    = +0.2039 (+22.6%)  [FE_POOL, importer_year 없음, OLS] <- 이상치')
    prt('')
    prt('  식별: sanc_post_gw는 hs6_year 셀 안에서 수입국 간에, importer_year 셀 안에서')
    prt('  품목 간에 변한다. 따라서 4-way FE에서도 식별된다. 이건 진단이 아니라')
    prt('  식(5) 자체를 옳게 쓴 것이다.')
    prt('')
    prt('  Reading: G7 near +3.5% means the omitted fixed effect was the cause.')
    prt('        G7이 +22% 부근이면 원인은 FE가 아니라 등가중 대 수준 적합의 차이다')
    prt('        (서술이 완전히 달라진다).')

    # 싼 것부터. OLS 두 개가 먼저 나와야 판정을 시작할 수 있다.
    run_spec(samples['pool'], 'G7: OLS  pooled + importer_year FE (correct Eq.5)',
             'OLS', 'sanc_post_gw', FE_POOL_FULL, results, fepois_kwargs)
    run_spec(samples['pool'], 'G9: OLS  pooled + importer_year FE, staggered treatment',
             'OLS', 'sanc_stag_gw', FE_POOL_FULL, results, fepois_kwargs)
    run_spec(samples['pool'], 'G8: PPML pooled + importer_year FE (correct Eq.5)',
             'PPML', 'sanc_post_gw', FE_POOL_FULL, results, fepois_kwargs)
    run_spec(samples['pool'], 'G10: PPML pooled + importer_year FE, staggered treatment',
             'PPML', 'sanc_stag_gw', FE_POOL_FULL, results, fepois_kwargs)

    prt('\n' + '=' * 74)
    prt('SUMMARY')
    prt('=' * 74)
    for r in results:
        if r.get('nobs', -1) > 0:
            exp = r.get('expected_pct')
            tag = f"  (expected ~{exp:+.1f}%)" if exp is not None else ''
            prt(f"  {r['spec']:<56} {r['pct']:+7.1f}%{tag}")
        else:
            prt(f"  {r['spec']:<56} FAILED")
    prt('\nG5/G6 해석 주의: importer_year FE가 없으므로 관문국별 연도 충격은 통제되지')
    prt('  않는다. 계수는 "제품-연도·수출자-연도 효과를 통제한 비관문국 대비 추가 변화"이지')
    prt('  관문국의 인과효과나 환적의 확정적 증거가 아니다.')
    prt('  ** The same caveat applies to OLS G5 (+22.6%): it omits importer-by-year FE. **')
    prt('')
    prt('G7~G10이 식(5)의 올바른 명세다. 보고할 때는 이쪽을 쓰고, G5/G6은 누락 FE의')
    prt('  효과를 보이는 비교 행으로만 싣는다.')
    prt('G9/G10 주의: staggered 처치는 2016년 크림 코드 17개의 시점을 바로잡지만')
    prt('  코드 수가 2,304 중 17개라 계수 자체는 크게 움직이지 않을 수 있다.')
    prt('  The point is consistency with staggered treatment timing, not effect size.')

    # 판정 보조 — 사람이 로그만 보고 결론을 내리지 않도록 대조표를 찍는다
    by_spec = {r['spec'][:3].rstrip(':'): r for r in results if r.get('nobs', -1) > 0}
    g5, g6, g7, g8 = (by_spec.get(k) for k in ('G5', 'G6', 'G7', 'G8'))
    if g7:
        prt('\n--- 판정 보조 ---')
        prt(f"  G5 (importer_year 없음, OLS)  : {g5['beta']:+.4f}  {g5['pct']:+.1f}%"
            if g5 else '  G5 없음')
        prt(f"  G7 (importer_year 있음, OLS)  : {g7['beta']:+.4f}  {g7['pct']:+.1f}%")
        if g6:
            prt(f"  G6 (importer_year 없음, PPML) : {g6['beta']:+.4f}  {g6['pct']:+.1f}%")
        if g8:
            prt(f"  G8 (importer_year 있음, PPML) : {g8['beta']:+.4f}  {g8['pct']:+.1f}%")
        if g5:
            shrink = (1 - abs(g7['beta']) / abs(g5['beta'])) * 100 if g5['beta'] else float('nan')
            prt(f"  OLS에서 importer_year를 넣었을 때 계수 축소: {shrink:.0f}%")
            if abs(g7['pct']) < 8.0:
                prt('  -> G7이 8% 아래다. 누락 FE가 +22.6%의 원인이라는 판정과 부합한다.')
            elif abs(g7['pct']) > 15.0:
                prt('  -> G7이 15% 위다. 누락 FE만으로는 설명되지 않는다.')
                prt('     원인을 등가중 대 수준 적합의 차이에서 다시 찾아야 한다.')
            else:
                prt('  -> G7이 중간 구간이다. 단정하지 말고 G8과 함께 읽을 것.')
    prt(f'\nTotal: {(time.time()-t_start)/60:.1f} min')
    prt(f'Saved: {CSV_PATH}')


if __name__ == '__main__':
    try:
        main()
    finally:
        flush_log()
