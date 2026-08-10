#!/usr/bin/env python3
"""Influence and inference diagnostics for the benchmark differential.

Supplies the leave-one-country-out results reported in Section 4.1 and the
sensitivity discussion in Section 6.4.

Why these are needed. The bridge decomposition in script 23 splits the
benchmark into a coalition-side coefficient and a non-coalition-side
coefficient, and more than half of the contrast comes from the non-coalition
side. Because the benchmark is a contrast against non-coalition exporters, a
single large non-coalition partner can move it, and the same diagnostics that
apply to the supplier-replacement estimate apply here.

  H0  reproduction gates for both the bridge and benchmark specifications;
      the script exits if either fails
  H1  leave-one-country-out across six major non-coalition partners, reporting
      the coalition side, the non-coalition side and the contrast separately
  H2  excluding the top 1% and 5% of pre-sanctions pairs
  H3  clustering at HS6 rather than partner x HS6
  H4  the decisive exclusion, re-run under the benchmark fixed effects
  H5  HS6 clustering under the benchmark fixed effects
  H6  excluding 2024, whose BACI values are provisional
  H7  five-country leave-one-out under the companion fixed-effect structure

Cost. The benchmark specification with HS6 x year fixed effects takes roughly
half an hour per fit, so the exhaustive sweeps use the cheaper HS2 x year
structure, whose contrast matches the benchmark almost exactly, and only the
decisive cells are re-run under the benchmark fixed effects. Each fit is
written to CSV and to the log as it completes.

Decision rules, fixed in advance
  If the contrast stays at or below -0.30 after excluding a country, the
  benchmark does not depend on that country; if it rises above -0.20 it does.
  If the coalition-side coefficient is insensitive to exclusions, the claim is
  narrowed to the coalition-side contraction.
  If HS6 clustering pushes |t| below 1.96, the significance claim is dropped.
  If trimming the top 1% halves the contrast, dependence on a few large flows
  is stated explicitly.

Country codes are verified against the BACI country file before anything is
estimated; a missing code would mean a coalition member had been treated as a
third country throughout, so the script stops rather than continue.

Requires pyfixest 0.40.1, as in script 27.

Usage
  python 29_headline_diagnostics.py
  python 29_headline_diagnostics.py --stage=cheap
  python 29_headline_diagnostics.py --stage=costly

Outputs: results/b29_headline_diagnostics.csv, results/b29_headline_log.txt
"""

import argparse
import inspect
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

PANEL_PATH = Path('phase4_analysis/processed/baci_russia_panel.parquet')
RESULTS_DIR = Path('phase4_analysis/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = RESULTS_DIR / 'b29_headline_log.txt'
CSV_PATH = RESULTS_DIR / 'b29_headline_diagnostics.csv'

EU27 = {40, 56, 100, 191, 196, 203, 208, 233, 246, 251, 276, 300,
        348, 372, 380, 428, 440, 442, 470, 528, 616, 620, 642,
        703, 705, 724, 752}
TIER1_NON_EU = {826, 842, 124, 392, 579, 757}
TIER1 = EU27 | TIER1_NON_EU
RUSSIA = 643
PRE_YEARS = range(2015, 2022)

LOO_COUNTRIES = {156: 'China', 792: 'Turkey', 784: 'UAE',
                 51: 'Armenia', 417: 'Kyrgyzstan', 398: 'Kazakhstan'}

VCOV = {'CRV1': 'partner_hs6'}
VCOV_HS6 = {'CRV1': 'hs6'}
FE_B1 = 'partner_hs6 + partner_year + hs2_year'
FE_C2 = 'partner_hs6 + partner_year + hs6_year'

# --- 23번 로그(b23_bridge_log.txt)에서 가져온 고정 검증값 ---
EXPECTED_PYFIXEST = '0.40.1'
RAW_ROWS = 1_056_761
SAMPLE_ROWS = 1_017_770
COAL_ROWS = 569_914
NONCOAL_ROWS = 447_856
SANC_COAL_ROWS = 43_425
SANC_NONCOAL_ROWS = 45_736

B1_EXPECTED_N = 1_017_601
B1_EXPECTED_BETA1 = -0.2119        # sanc_coal
B1_EXPECTED_BETA2 = +0.2416        # sanc_noncoal
B1_EXPECTED_CONTRAST = -0.4535

C2_EXPECTED_N = 1_015_978
C2_EXPECTED_BETA = -0.4538
C2_EXPECTED_PCT = -36.5

# --- b29_headline_log_costly.txt의 H4 출력에서 가져온 재현 검증값 ---
# H4: C2 excluding China  beta=-0.1296 (0.0904)  N=978,128
C2_CHINA_EXPECTED_BETA = -0.1296
C2_CHINA_EXPECTED_N = 978_128

GATE_TOL_BETA = 0.005
GATE_TOL_PP = 0.5

# 사전에 못박은 판정 임계값 (위 docstring 참조)
CHINA_ROBUST_CONTRAST = -0.30      # 이보다 음수면 중국 비의존
CHINA_BROKEN_CONTRAST = -0.20      # 이보다 위면 중국 의존

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
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def pct_of(beta):
    return (np.exp(beta) - 1) * 100


def check_country_codes():
    """Tier 1 국가코드가 BACI에 실재하는지 확인한다. 실패하면 즉시 중단.

    왜 필요한가. 이 목록은 India를 356(ISO)에서 699(BACI)로 한 번 고친 적이 있다.
    그런데 목록 안에 251(FRA)·842(USA)·579(NOR)·757(CHE)처럼 Comtrade 확장코드가
    섞여 있는데 **380(ITA)만 순수 ISO 코드**다. Comtrade 계열에서 이탈리아는 381이다.
    380이 BACI에 없다면 이탈리아는 지금까지 모든 추정에서 연합국이 아니라
    **제3국**으로 들어가 있었다는 뜻이고, 헤드라인과 우회 계수가 둘 다 틀린다.

    1분이면 끝나는 검사이므로 비싼 추정 앞에 무조건 세운다.
    """
    prt('\n=== Tier 1 국가코드 실재 확인 ===')
    cc_path = None
    for cand in (Path('phase3_data/BACI_HS92_V202601/country_codes_V202601.csv'),
                 Path('phase3_data/raw/country_codes.csv')):
        if cand.exists():
            cc_path = cand
            break
    if cc_path is None:
        prt('  [FAIL] BACI country_codes 파일을 찾지 못했다. 코드 검증 없이는')
        prt('         진단을 진행하지 않는다. --skip-code-check로 우회할 수 있으나')
        prt('         In that case the diagnostics are not usable.')
        flush_log()
        sys.exit(1)

    cc = pd.read_csv(cc_path)
    code_col = next((c for c in ('country_code', 'country_id', 'i', 'k')
                     if c in cc.columns), None)
    if code_col is None:
        raise KeyError(f'{cc_path}에서 국가코드 열을 찾지 못했다: {list(cc.columns)}')
    present = set(pd.to_numeric(cc[code_col], errors='coerce').dropna().astype(int))

    missing = sorted(TIER1 - present)
    prt(f'  대조 파일: {cc_path}')
    prt(f'  Tier 1 코드 {len(TIER1)}개 중 BACI에 없는 코드: {missing if missing else "없음"}')
    if missing:
        name_col = next((c for c in ('country_iso3', 'iso_3digit_alpha', 'country_name')
                         if c in cc.columns), None)
        prt('  [FAIL] 위 코드는 BACI에 존재하지 않는다. 해당 국가는 지금까지')
        prt('         연합국이 아니라 제3국으로 집계되어 왔다는 뜻이다.')
        if name_col:
            for probe in (380, 381, 250, 251, 840, 842, 356, 699):
                row = cc[pd.to_numeric(cc[code_col], errors='coerce') == probe]
                if len(row):
                    prt(f'         참고 {probe} = {row.iloc[0][name_col]}')
        prt('         목록을 고친 뒤 모든 추정을 다시 돌려야 한다.')
        flush_log()
        sys.exit(1)
    prt('  [OK ] Tier 1 코드 전부 BACI에 존재한다.')


def check_version(allow_mismatch):
    ver = getattr(pf, '__version__', 'unknown')
    prt(f'pyfixest {ver} (기대 {EXPECTED_PYFIXEST})')
    if ver != EXPECTED_PYFIXEST and not allow_mismatch:
        prt(f'  [FAIL] 버전이 다르면 재현 게이트가 의미를 잃는다. '
            f'pip install pyfixest=={EXPECTED_PYFIXEST}')
        flush_log()
        sys.exit(1)


def supported_fepois_kwargs():
    """이 pyfixest 버전이 실제로 받는 인자만 남긴다 (22·23·27과 동일 절차).

    재현 게이트를 돌리는 스크립트이므로 시그니처를 못 읽으면 조용히 기본값으로
    떨어지지 않고 실패시킨다.
    """
    try:
        params = inspect.signature(pf.fepois).parameters
    except (TypeError, ValueError) as e:
        prt(f'    [FAIL] fepois 시그니처를 읽지 못했다 ({e}). '
            f'재현 게이트가 있는 스크립트에서는 기본값 폴백을 허용하지 않는다.')
        flush_log()
        sys.exit(1)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(FEPOIS_KWARGS)
    ok = {k: v for k, v in FEPOIS_KWARGS.items() if k in params}
    dropped = sorted(set(FEPOIS_KWARGS) - set(ok))
    if dropped:
        prt(f'    (이 pyfixest 버전이 받지 않는 인자 제외: {dropped})')
    return ok


def load_to_russia():
    """23번 load_to_russia와 동일한 표본. hs6·t·v를 추가로 보존한다.

    hs6  -> HS6 클러스터 추론(H3·H5)
    t·v  -> 사전 규모 분위(H2)
    """
    prt('=== Loading panel ===')
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f'패널이 없다: {PANEL_PATH}')
    panel = pd.read_parquet(PANEL_PATH)
    to_rus = panel[(panel['direction'] == 'to_russia')
                   & (panel['partner'] != RUSSIA)].copy()
    del panel
    prt(f'To-Russia (raw): {len(to_rus):,}')

    grp = to_rus.groupby('partner_hs6')['t'].nunique()
    to_rus = to_rus[to_rus['partner_hs6'].isin(grp[grp > 1].index)].copy()
    prt(f'To-Russia (after singletons): {len(to_rus):,}')

    to_rus['is_coalition'] = to_rus['partner'].isin(TIER1).astype('int8')
    to_rus['sanc_stag'] = to_rus['hs6_sanctioned_at_t'].astype('int8')
    to_rus['sanc_coal'] = (to_rus['sanc_stag'] * to_rus['is_coalition']).astype('int8')
    to_rus['sanc_noncoal'] = (to_rus['sanc_stag']
                              * (1 - to_rus['is_coalition'])).astype('int8')
    to_rus['hs2_year'] = to_rus['hs6'].str[:2] + '_' + to_rus['t'].astype(str)

    n_coal = int(to_rus['is_coalition'].sum())
    prt(f'  coalition rows: {n_coal:,} / non-coalition: {len(to_rus)-n_coal:,}')
    prt(f"  sanc x coal = 1: {to_rus['sanc_coal'].sum():,}, "
        f"sanc x noncoal = 1: {to_rus['sanc_noncoal'].sum():,}")

    # --- 표본 규격 게이트 (계수만 맞고 표본이 다른 경우를 막는다) ---
    checks = [
        ('raw rows', len(to_rus) if False else RAW_ROWS, RAW_ROWS),  # 표시용
        ('sample rows', len(to_rus), SAMPLE_ROWS),
        ('coalition rows', n_coal, COAL_ROWS),
        ('non-coalition rows', len(to_rus) - n_coal, NONCOAL_ROWS),
        ('sanc x coal rows', int(to_rus['sanc_coal'].sum()), SANC_COAL_ROWS),
        ('sanc x noncoal rows', int(to_rus['sanc_noncoal'].sum()), SANC_NONCOAL_ROWS),
    ]
    failed = []
    for name, got, want in checks[1:]:
        ok = (got == want)
        prt(f'  [{"OK " if ok else "FAIL"}] {name}: {got:,} (기대 {want:,})')
        if not ok:
            failed.append(name)
    if failed:
        prt(f'\n[GATE FAILED] 표본 규격 불일치: {failed}')
        prt('표본이 23번과 다르다. 진단을 진행하지 않는다.')
        flush_log()
        sys.exit(1)

    keep = ['v', 't', 'hs6', 'partner', 'sanc_stag', 'sanc_coal', 'sanc_noncoal',
            'partner_hs6', 'partner_year', 'hs2_year', 'hs6_year']
    missing = [c for c in keep if c not in to_rus.columns]
    if missing:
        raise KeyError(f'필요한 열이 없다 -> {missing}')
    return to_rus[keep].copy()


def fit_spec(data, label, terms, fe, vcov, results, kwargs, note=''):
    """PPML 하나를 적합하고 계수를 기록한다. 대비(contrast)까지 계산한다."""
    fml = f"v ~ {' + '.join(terms)} | {fe}"
    prt(f'\n--- {label} ---')
    prt(f'    PPML: {fml}')
    prt(f'    vcov: {vcov}')
    flush_log()                       # 긴 적합 전에 진행 상태를 남긴다
    t0 = time.time()
    row = {'spec': label, 'formula': fml, 'vcov': str(vcov), 'note': note}
    try:
        fe_cols = [c for c in ('partner_hs6', 'partner_year',
                               'hs2_year', 'hs6_year') if c in fe]
        cl_cols = [v for v in vcov.values()]
        cols = list(dict.fromkeys(['v'] + terms + fe_cols + cl_cols))
        absent = [c for c in cols if c not in data.columns]
        if absent:
            raise KeyError(f'필요한 열이 없다 -> {absent}')
        m = pf.fepois(fml, data=data[cols], vcov=vcov, **kwargs)
        el, n = time.time() - t0, get_nobs(m)
        row.update({'nobs': n, 'seconds': round(el, 1)})
        for term in terms:
            b, se, p = m.coef()[term], m.se()[term], m.pvalue()[term]
            prt(f'    {term:<13} beta={b:+.4f} ({se:.4f}){stars(p)}  '
                f'pct={pct_of(b):+.1f}%')
            row[f'{term}_beta'], row[f'{term}_se'] = b, se
            row[f'{term}_p'], row[f'{term}_pct'] = p, pct_of(b)
        if 'sanc_coal' in terms and 'sanc_noncoal' in terms:
            contrast = m.coef()['sanc_coal'] - m.coef()['sanc_noncoal']
            share = (abs(m.coef()['sanc_noncoal']) / abs(contrast) * 100
                     if contrast != 0 else np.nan)
            prt(f'    contrast(b1-b2) = {contrast:+.4f}  pct={pct_of(contrast):+.1f}%'
                f'   비연합국 기여 {share:.1f}%')
            row['contrast_beta'] = contrast
            row['contrast_pct'] = pct_of(contrast)
            row['noncoal_share_of_contrast'] = share
        prt(f'    N={n:,}  [{el:.0f}s]')
    except Exception as e:
        prt(f'    ERROR: {type(e).__name__}: {e}')
        row.update({'nobs': -1, 'seconds': round(time.time() - t0, 1),
                    'error': f'{type(e).__name__}: {e}'})
    results.append(row)
    checkpoint(results)
    return row


def gate(row, label, checks):
    """기대값 대조. 어긋나면 즉시 종료한다."""
    prt(f'    게이트 ({label}):')
    bad = []
    for name, got, want, tol in checks:
        if got is None:
            bad.append(f'{name}=없음')
            prt(f'      [FAIL] {name}: 값이 없다')
            continue
        diff = abs(got - want)
        ok = diff <= tol
        prt(f'      [{"OK " if ok else "FAIL"}] {name}: {got:+.4f} '
            f'(기대 {want:+.4f}, 차 {diff:.4f}, 허용 {tol})')
        if not ok:
            bad.append(name)
    if bad:
        prt(f'\n[GATE FAILED] {label} 재현 실패: {bad}')
        prt('Reproduction failed; aborting so that unusable diagnostics are not produced.')
        flush_log()
        sys.exit(1)
    prt('    Gate PASSED.')


def pair_pre_size(data):
    """쌍(partner x HS6) 수준 사전 평균 흐름. 27번 D1·D2와 같은 정의."""
    pre = data[data['t'].isin(PRE_YEARS)]
    size = pre.groupby('partner_hs6')['v'].mean().rename('pre_size')
    prt(f'  사전 관측이 있는 쌍: {len(size):,}')
    no_pre = data.loc[~data['partner_hs6'].isin(size.index), 'partner_hs6'].nunique()
    n_rows_no_pre = int((~data['partner_hs6'].isin(size.index)).sum())
    prt(f'  사전 관측이 없는 쌍: {no_pre:,} ({n_rows_no_pre:,}행) '
        f'— 상위 절단 사양에서 함께 제외된다')
    return size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage',
                    choices=['all', 'cheap', 'costly', 'c2loo', 'c2loo-kz'],
                    default='all')
    ap.add_argument('--fixef-maxiter', type=int, default=None,
                    help='demeaning 반복 상한을 올린다. 허용오차(fixef_tol)는 '
                         '건드리지 않으므로 사양 변경이 아니다.')
    ap.add_argument('--allow-version-mismatch', action='store_true')
    ap.add_argument('--skip-code-check', action='store_true',
                    help='skip the country-code check (not recommended: the run is then unusable)')
    args = ap.parse_args()

    global LOG_PATH, CSV_PATH, FEPOIS_KWARGS
    if args.stage == 'c2loo':
        # 기존 b29 산출물을 덮지 않도록 별도 파일에 쓴다.
        LOG_PATH = RESULTS_DIR / 'b29_c2_loo_log.txt'
        CSV_PATH = RESULTS_DIR / 'b29_c2_loo.csv'
    elif args.stage == 'c2loo-kz':
        LOG_PATH = RESULTS_DIR / 'b29_c2_loo_kz_log.txt'
        CSV_PATH = RESULTS_DIR / 'b29_c2_loo_kz.csv'
    if args.fixef_maxiter:
        FEPOIS_KWARGS = dict(FEPOIS_KWARGS, fixef_maxiter=args.fixef_maxiter)

    results = []
    t_start = time.time()
    try:
        prt('=' * 74)
        prt('A-7: 헤드라인 직접효과(-36.5%)의 영향력·추론 진단')
        prt('=' * 74)
        prt('배경: bridge 분해상 대비 -0.4535의 53.3%가 비연합국 상승(beta2 +0.2416)에서')
        prt('      나오고, 그 beta2는 A-5에서 중국 의존으로 판정됐다. 헤드라인이 같은')
        prt('      취약성을 물려받았는지 확인한다.')
        prt(f'단계: {args.stage}')
        check_version(args.allow_version_mismatch)
        if args.skip_code_check:
            prt('\n⚠️  Skipping the country-code check. This run is not usable.')
        else:
            check_country_codes()
        kwargs = supported_fepois_kwargs()
        data = load_to_russia()

        do_cheap = args.stage in ('all', 'cheap')
        do_costly = args.stage in ('all', 'costly')

        # ===== H7kz: 카자흐스탄만 재시도 (demeaning 미수렴 대응) =====
        if args.stage == 'c2loo-kz':
            prt('\n' + '=' * 74)
            prt('H7kz — C2 카자흐스탄 제외 재시도')
            prt('=' * 74)
            prt(f'  fixef_maxiter = {FEPOIS_KWARGS["fixef_maxiter"]:,}'
                f' (허용오차 fixef_tol = {FEPOIS_KWARGS["fixef_tol"]:g}, 원본과 동일)')
            prt('  100,000 반복에서 "Demeaning failed"로 끝난 사양이다.')
            prt(f'  기준선 C2: pct {C2_EXPECTED_PCT}%')
            sub = data[data['partner'] != 398]
            prt(f'  Kazakhstan(398) 제외:'
                f' {len(data)-len(sub):,}행 제거 -> {len(sub):,}행')
            r = fit_spec(sub, 'H7: C2 excluding Kazakhstan',
                         ['sanc_coal'], FE_C2, VCOV, results, kwargs,
                         note=f'retry, fixef_maxiter={FEPOIS_KWARGS["fixef_maxiter"]}')
            if r.get('sanc_coal_beta') is not None:
                d = r['sanc_coal_pct'] - C2_EXPECTED_PCT
                prt(f'    기준선 대비 편차: {d:+.3f}%p')
                prt('\n  → 이 값을 b29_c2_loo.csv의 네 나라와 합쳐 상한을 정한다.')
                prt('    (bridge 사양의 카자흐스탄 편차는 1.122%p였다)')
            else:
                prt('\n  → 재시도도 실패했다. §4.1은 카자흐스탄을 companion '
                    '사양으로 보고하고 미수렴 사실을 밝힌다.')
            prt(f'\nTotal: {(time.time()-t_start)/60:.1f} min')
            prt(f'Saved: {CSV_PATH}')
            return

        # ===== H7: C2 leave-one-out — 중국 이외 5개국 =====
        # Purpose: reproduce the five-country leave-one-out under the companion
        #       우회 인용하고 있다. 헤드라인 사양(C2, HS6xyear)에서 직접 돌려
        #       그 우회를 없앤다.
        if args.stage == 'c2loo':
            prt('\n' + '=' * 74)
            prt('H7-gate — C2 중국 제외 재현 (costly 로그의 H4와 대조)')
            prt('=' * 74)
            prt('  If the gate fails, the five-country results below are not used.')
            sub = data[data['partner'] != 156]
            prt(f'  China(156) 제외: {len(data)-len(sub):,}행 제거 -> {len(sub):,}행')
            rg = fit_spec(sub, 'H7-gate: C2 excluding China (reproduction)',
                          ['sanc_coal'], FE_C2, VCOV, results, kwargs,
                          note='b29 costly H4 재현: beta -0.1296, N 978,128')
            # gate()는 실패하면 sys.exit(1)로 끝낸다. 통과하면 그대로 진행한다.
            gate(rg, 'C2-China', [
                ('sanc_coal beta', rg.get('sanc_coal_beta'),
                 C2_CHINA_EXPECTED_BETA, GATE_TOL_BETA),
                ('N', rg.get('nobs'), C2_CHINA_EXPECTED_N, 0),
            ])

            prt('\n' + '=' * 74)
            prt('H7 — C2 leave-one-out (Turkey, UAE, Armenia, Kyrgyzstan, Kazakhstan)')
            prt('=' * 74)
            prt(f'  기준선 C2: beta {C2_EXPECTED_BETA}, pct {C2_EXPECTED_PCT}%,'
                f' N {C2_EXPECTED_N:,} (23번·H0b에서 확정)')
            for code, name in LOO_COUNTRIES.items():
                if code == 156:
                    continue
                sub = data[data['partner'] != code]
                prt(f'\n  {name}({code}) 제외:'
                    f' {len(data)-len(sub):,}행 제거 -> {len(sub):,}행')
                r = fit_spec(sub, f'H7: C2 excluding {name}',
                             ['sanc_coal'], FE_C2, VCOV, results, kwargs,
                             note=f'excluded={name}, base_beta={C2_EXPECTED_BETA}')
                b = r.get('sanc_coal_beta')
                if b is not None:
                    dpp = r['sanc_coal_pct'] - C2_EXPECTED_PCT
                    prt(f'    기준선 대비 편차: {dpp:+.3f}%p')

            prt('\n' + '=' * 74)
            prt('H7 SUMMARY - reported values')
            prt('=' * 74)
            devs = []
            for row in results:
                if not row['spec'].startswith('H7: '):
                    continue
                if row.get('nobs', -1) < 0:
                    prt(f"  {row['spec']:<40s} ERROR")
                    continue
                d = row['sanc_coal_pct'] - C2_EXPECTED_PCT
                devs.append(abs(d))
                prt(f"  {row['spec']:<40s} beta={row['sanc_coal_beta']:+.4f}"
                    f" ({row['sanc_coal_se']:.4f})  pct={row['sanc_coal_pct']:+.2f}%"
                    f"  편차={d:+.3f}%p")
            if devs:
                prt(f'\n  최대 편차: {max(devs):.3f}%p'
                    f'  → §4.1 문장에 쓸 상한: at most {max(devs):.1f} percentage points')
                prt('  (bridge 사양의 기존 값은 최대 1.122%p였다 — 두 값을 비교할 것)')
            prt(f'\nTotal: {(time.time()-t_start)/60:.1f} min')
            prt(f'Saved: {CSV_PATH}')
            return

        # ================= H0a: B1 재현 게이트 =================
        if do_cheap:
            prt('\n' + '=' * 74)
            prt('H0a — B1 재현 게이트 (bridge 분해, HS2xyear FE)')
            prt('=' * 74)
            r = fit_spec(data, 'H0a: B1 baseline (bridge)',
                         ['sanc_coal', 'sanc_noncoal'], FE_B1, VCOV, results, kwargs,
                         note='23번 B1 재현')
            gate(r, 'B1', [
                ('sanc_coal beta', r.get('sanc_coal_beta'),
                 B1_EXPECTED_BETA1, GATE_TOL_BETA),
                ('sanc_noncoal beta', r.get('sanc_noncoal_beta'),
                 B1_EXPECTED_BETA2, GATE_TOL_BETA),
                ('contrast', r.get('contrast_beta'),
                 B1_EXPECTED_CONTRAST, GATE_TOL_BETA),
                ('N', r.get('nobs'), B1_EXPECTED_N, 0),
            ])
            base_contrast = r['contrast_beta']

            # ================= H1: 국가별 제외 =================
            prt('\n' + '=' * 74)
            prt('H1 — 주요 비연합국 하나씩 제외 (B1 분해)')
            prt('=' * 74)
            prt('  beta1(연합국 하락)은 제외 국가와 무관해야 정상이다. beta2(비연합국')
            prt('  상승)와 대비가 함께 움직이면 헤드라인이 그 나라에 의존한다는 뜻이다.')
            for code, name in LOO_COUNTRIES.items():
                sub = data[data['partner'] != code]
                removed = len(data) - len(sub)
                prt(f'\n  {name}({code}) 제외: {removed:,}행 제거 -> {len(sub):,}행')
                fit_spec(sub, f'H1: B1 excluding {name}',
                         ['sanc_coal', 'sanc_noncoal'], FE_B1, VCOV, results, kwargs,
                         note=f'excluded={name}, base_contrast={base_contrast:.4f}')

            # ================= H2: 사전 규모 상위 제외 =================
            prt('\n' + '=' * 74)
            prt('H2 — 사전 규모 상위 1%·5% 쌍 제외 (B1 분해)')
            prt('=' * 74)
            size = pair_pre_size(data)
            for q, tag in ((0.99, 'top 1%'), (0.95, 'top 5%')):
                thr = size.quantile(q)
                keep_pairs = size[size <= thr].index
                sub = data[data['partner_hs6'].isin(keep_pairs)]
                prt(f'\n  {tag} 제외: 임계 {thr:,.2f} 천 USD, '
                    f'{len(data)-len(sub):,}행 제거 -> {len(sub):,}행')
                fit_spec(sub, f'H2: B1 excluding {tag} of pairs by pre-size',
                         ['sanc_coal', 'sanc_noncoal'], FE_B1, VCOV, results, kwargs,
                         note=f'threshold={thr:.2f}k USD')

            # ================= H3: HS6 클러스터 =================
            prt('\n' + '=' * 74)
            prt('H3 — HS6 클러스터 추론 (B1). 점추정은 불변, 표준오차만 바뀐다')
            prt('=' * 74)
            prt('  처치는 HS6 제품 수준에서 배정되는데 기본 클러스터는 partner_hs6다.')
            prt('  같은 HS6의 여러 수출국이 공통 제품 충격을 공유하면 SE가 과소평가된다.')
            fit_spec(data, 'H3: B1 with HS6 clustering',
                     ['sanc_coal', 'sanc_noncoal'], FE_B1, VCOV_HS6, results, kwargs,
                     note='점추정은 H0a와 같아야 한다')

            # ================= H6: 2024년 제외 =================
            prt('\n' + '=' * 74)
            prt('H6 — 2024년 제외')
            prt('=' * 74)
            prt('  CEPII는 최신 연도 값이 이후 버전에서 수정될 수 있다고 밝힌다.')
            prt('  2024 제외로 분해나 대비가 크게 움직이면 동학·2차 사후연도 주장을 보류한다.')
            sub = data[data['t'] <= 2023]
            prt(f'  {len(data):,}행 -> {len(sub):,}행 (2024년 {len(data)-len(sub):,}행 제거)')
            fit_spec(sub, 'H6: B1 excluding provisional year 2024',
                     ['sanc_coal', 'sanc_noncoal'], FE_B1, VCOV, results, kwargs,
                     note='표본 2015-2023. H0a와 대조')

        # ================= H0b/H4/H5: 헤드라인 사양 =================
        if do_costly:
            prt('\n' + '=' * 74)
            prt('H0b - C2 reproduction gate (benchmark specification, HS6xyear FE)')
            prt('=' * 74)
            prt('  주의: 23번에서 1,714초(28.5분) 걸린 사양이다.')
            r = fit_spec(data, 'H0b: C2 baseline (manuscript headline)',
                         ['sanc_coal'], FE_C2, VCOV, results, kwargs,
                         note='23번 C2 재현')
            gate(r, 'C2', [
                ('sanc_coal beta', r.get('sanc_coal_beta'),
                 C2_EXPECTED_BETA, GATE_TOL_BETA),
                ('pct', r.get('sanc_coal_pct'), C2_EXPECTED_PCT, GATE_TOL_PP),
                ('N', r.get('nobs'), C2_EXPECTED_N, 0),
            ])

            prt('\n' + '=' * 74)
            prt('H4 — C2 중국 제외 (B1 대리가 헤드라인에서도 맞는지 확인)')
            prt('=' * 74)
            sub = data[data['partner'] != 156]
            prt(f'  China(156) 제외: {len(data)-len(sub):,}행 제거 -> {len(sub):,}행')
            fit_spec(sub, 'H4: C2 excluding China', ['sanc_coal'], FE_C2,
                     VCOV, results, kwargs, note='H1의 China 행과 대조할 것')

            prt('\n' + '=' * 74)
            prt('H5 — C2 HS6 클러스터 추론')
            prt('=' * 74)
            fit_spec(data, 'H5: C2 with HS6 clustering', ['sanc_coal'], FE_C2,
                     VCOV_HS6, results, kwargs,
                     note='점추정은 H0b와 같아야 한다')

        # ================= 요약 =================
        prt('\n' + '=' * 74)
        prt('SUMMARY')
        prt('=' * 74)
        for row in results:
            if row.get('nobs', -1) < 0:
                prt(f"  {row['spec']:<48s} ERROR")
                continue
            bits = []
            for term in ('sanc_coal', 'sanc_noncoal'):
                if f'{term}_beta' in row:
                    bits.append(f"{term}={row[f'{term}_beta']:+.4f}"
                                f"({row[f'{term}_se']:.4f})")
            if 'contrast_beta' in row:
                bits.append(f"contrast={row['contrast_beta']:+.4f}"
                            f" ({row['contrast_pct']:+.1f}%)")
            prt(f"  {row['spec']:<48s} {' | '.join(bits)}")

        prt('\n판정 기준 (사전에 못박은 것 — 결과를 보고 바꾸지 말 것):')
        prt(f'  · H1 중국 제외 후 대비가 {CHINA_ROBUST_CONTRAST} 이하로 유지'
            f' -> 헤드라인은 중국 비의존')
        prt(f'  · H1 중국 제외 후 대비가 {CHINA_BROKEN_CONTRAST} 위로 올라옴'
            f' -> 헤드라인도 중국 의존, 서술 전면 수정')
        prt('  · beta1(연합국 하락)이 국가 제외에 둔감하면 "연합국 하락"으로 서술을 좁힌다')
        prt('  · H3/H5의 HS6 클러스터에서 |t| < 1.96이면 유의성 주장을 접는다')
        prt('  · H2 상위 1% 제외 후 대비가 절반 아래로 줄면 대형 흐름 의존을 명시한다')
        prt('\n서술 규칙: 헤드라인 -36.5%는 "연합국 무역이 36.5% 줄었다"가 아니라')
        prt('  "연합국 대비 비연합국 격차가 36.5% 로그포인트"다. 분해를 밝히지 않고')
        prt('  전자로 쓰면 안 된다.')
        prt(f'\nTotal: {(time.time()-t_start)/60:.1f} min')
        prt(f'Saved: {CSV_PATH}')
    finally:
        checkpoint(results)


if __name__ == '__main__':
    main()
