#!/usr/bin/env python3
"""Influence and composition diagnostics for the supplier-replacement estimate.

Supplies the exclusion rows of Table 5 Panel A and the Appendix E diagnostics.
The estimate differs by a factor of more than two across estimators, +9.0% by
OLS and +22.7% by PPML. That gap is consistent with adjustment concentrated in
large flows, but it is equally consistent with a handful of influential
observations. These diagnostics separate the two.

  D0  reproduction gate for the baseline (sample size, treated counts, country
      and HS6 counts, N, coefficient, percentage)
  D1  effect by quartile of pre-sanctions flow size
  D2  excluding the top 1% and 5% of pre-sanctions flows
  D3  excluding major third countries one at a time
  D4  accounting for the observations lost in the balanced-panel variant
  D5  re-estimation on the pre-registered ten-country sample
  D6  excluding 2024, whose BACI values are provisional

D2 runs in three arms. Pairs with no pre-sanctions observation are pairs that
first traded after 2022, and dropping them together with the largest flows
conflates two very different exclusions; newly opened trade relationships are
themselves a leading rerouting candidate.
    D2a  drop new entrants as well as the top tail (earlier construction)
    D2b  trim the top tail only, like-for-like with D0   <- the reported arm
    D2c  drop new entrants only, no trimming

Sample, fixed effects and clustering match the main supplier-replacement
specification (partner x HS6, partner x year, HS2 x year).

Notes
  Quartile boundaries are cut on the pair-level distribution, not the row-level
  one, so that long-observed pairs do not dominate the cut points.
  Pre-sanctions size is the mean flow over observed years, because BACI records
  positive flows only.
  D4 refits the balanced specification rather than counting all-zero groups:
  in a balanced panel every pair is positive in at least one year by
  construction.

Requires pyfixest 0.40.1; the singleton-removal default changed in that
release, so a different version invalidates the reproduction gates.

Outputs: results/b27_rerouting_diagnostics.csv, results/b27_diagnostics_log.txt
"""

import argparse
import inspect
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

PANEL_PATH = Path('phase4_analysis/processed/baci_russia_panel.parquet')
RESULTS_DIR = Path('phase4_analysis/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = RESULTS_DIR / 'b27_diagnostics_log.txt'
CSV_PATH = RESULTS_DIR / 'b27_rerouting_diagnostics.csv'

EU27 = {40, 56, 100, 191, 196, 203, 208, 233, 246, 251, 276, 300,
        348, 372, 380, 428, 440, 442, 470, 528, 616, 620, 642,
        703, 705, 724, 752}
TIER1_NON_EU = {826, 842, 124, 392, 579, 757}
TIER1 = EU27 | TIER1_NON_EU
RUSSIA = 643
YEARS = np.arange(2015, 2025)
PRE_YEARS = range(2015, 2022)

LOO_COUNTRIES = {156: 'China', 792: 'Turkey', 784: 'UAE',
                 51: 'Armenia', 417: 'Kyrgyzstan', 398: 'Kazakhstan'}

# D5용 — 사전규칙(B0_preanalysis_rules.md §B-0.2) 표본. 고정 후보 8개국
# (CHN·IND·TUR·ARE·ARM·GEO·KAZ·KGZ)에 exposure 분석의 UZB·TJK를 더한 것으로,
# 10b·11·21·26의 GATEWAY 집합과 동일하다. India는 699 (BACI 코드).
PRERULE_COUNTRIES = {156: 'China', 699: 'India', 792: 'Turkey', 784: 'UAE',
                     51: 'Armenia', 268: 'Georgia', 398: 'Kazakhstan',
                     417: 'Kyrgyzstan', 860: 'Uzbekistan', 762: 'Tajikistan'}

VCOV = {'CRV1': 'partner_hs6'}
FE = 'partner_hs6 + partner_year + hs2_year'
FE_COLS = ('partner_hs6', 'partner_year', 'hs2_year')

# --- 22번 로그(b22_rerouting_log.txt)에서 가져온 고정 검증값 ---
EXPECTED_PYFIXEST = '0.40.1'
S4_SAMPLE_ROWS = 447_856      # singleton 제거 후 표본 행 수
S4_SANC_ROWS = 45_736         # sanc_stag = 1 인 행 수
S4_PARTNERS = 180
S4_HS6 = 4_568
S4_EXPECTED_N = 447_688       # 적합 표본 N
S4_EXPECTED_BETA = 0.2043
S4_EXPECTED_PCT = 22.7
GATE_TOL_PP = 0.5
GATE_TOL_BETA = 0.005

S5_BALANCED_CELLS = 747_110
S5_EXPECTED_N = 717_161

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

    시그니처를 읽지 못하면 조용히 기본값으로 넘어가지 않고 실패시킨다. solver 설정이
    달라지면 재현 게이트가 무의미해지기 때문이다.
    """
    params = inspect.signature(pf.fepois).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(FEPOIS_KWARGS)
    ok = {k: v for k, v in FEPOIS_KWARGS.items() if k in params}
    dropped = sorted(set(FEPOIS_KWARGS) - set(ok))
    if dropped:
        prt(f'    (이 pyfixest 버전이 받지 않는 인자 제외: {dropped})')
    return ok


def check_version(allow_mismatch):
    v = getattr(pf, '__version__', 'unknown')
    prt(f'pyfixest {v} (기대 {EXPECTED_PYFIXEST})')
    if v != EXPECTED_PYFIXEST:
        msg = (f'pyfixest {v} != {EXPECTED_PYFIXEST}. 0.40.1에서 fixef_rm 기본값이 '
               f'바뀌었으므로 22번 재현값과 대조할 수 없다.')
        if not allow_mismatch:
            raise RuntimeError(msg + ' 계속하려면 --allow-version-mismatch')
        prt(f'  ⚠️ {msg} (--allow-version-mismatch 지정됨)')


def load_sample():
    """22번 S4와 동일한 제3국 -> 러시아 표본. 구성값을 22번 로그와 대조한다."""
    prt('=== Loading panel ===')
    panel = pd.read_parquet(PANEL_PATH)
    prt(f'Panel: {len(panel):,} rows')

    is_third = (~panel['partner'].isin(TIER1)) & (panel['partner'] != RUSSIA)
    d = panel[(panel['direction'] == 'to_russia') & is_third].copy()
    prt(f'Third -> Russia (raw): {len(d):,}')

    grp = d.groupby('partner_hs6')['t'].nunique()
    d = d[d['partner_hs6'].isin(grp[grp > 1].index)].copy()
    prt(f'Third -> Russia (after singletons): {len(d):,}')

    d['sanc_stag'] = d['hs6_sanctioned_at_t'].astype('int8')
    if not set(np.unique(d['sanc_stag'])) <= {0, 1}:
        raise ValueError('sanc_stag is not binary')
    d['hs2_year'] = d['hs6'].str[:2] + '_' + d['t'].astype(str)

    # --- 표본 구성 게이트 (계수만 맞고 표본이 다른 경우를 막는다) ---
    checks = [
        ('sample rows', len(d), S4_SAMPLE_ROWS),
        ('sanc_stag rows', int(d['sanc_stag'].sum()), S4_SANC_ROWS),
        ('partners', int(d['partner'].nunique()), S4_PARTNERS),
        ('HS6 codes', int(d['hs6'].nunique()), S4_HS6),
    ]
    for name, got, want in checks:
        ok = got == want
        prt(f'  [{"OK " if ok else "FAIL"}] {name}: {got:,} (기대 {want:,})')
        if not ok:
            raise RuntimeError(f'S4 표본 불일치 — {name}: {got:,} != {want:,}')
    years = sorted(int(y) for y in d['t'].unique())
    if years != list(range(2015, 2025)):
        raise RuntimeError(f'연도 지지집합 불일치: {years}')
    prt('  [OK ] years 2015-2024')

    # 사전 기간(2015-2021)의 관측된 흐름 평균으로 쌍의 크기를 정의한다.
    # 처치 후 값을 쓰면 처치가 크기 분류로 새어 들어간다.
    # BACI는 양의 흐름만 기록하므로 이는 "관측된 연도의 평균"이지 "연평균"이 아니다.
    pre = d[d['t'].isin(PRE_YEARS)].groupby('partner_hs6')['v'].mean()
    d['pre_size'] = d['partner_hs6'].map(pre)
    n_nopre = int(d['pre_size'].isna().sum())
    prt(f'  사전 기간 관측이 없는 행: {n_nopre:,} ({n_nopre/len(d)*100:.1f}%) '
        f'— D1·D2에서 동일하게 제외')
    return panel, d


def fit(fml, data, estimator, fepois_kwargs):
    if estimator == 'OLS':
        return pf.feols(fml, data=data, vcov=VCOV)
    return pf.fepois(fml, data=data, vcov=VCOV, **fepois_kwargs)


def run(data, label, terms, results, fepois_kwargs, estimator='PPML',
        gate=False, note=''):
    y = 'v' if estimator == 'PPML' else 'ln_v'
    fml = f'{y} ~ {" + ".join(terms)} | {FE}'
    prt(f'\n--- {label} ---')
    prt(f'    {estimator}: {fml}')
    flush_log()          # 긴 추정 중 죽어도 어디까지 갔는지 남긴다
    t0 = time.time()
    required = list(dict.fromkeys([y] + list(terms) + list(FE_COLS)))
    absent = [c for c in required if c not in data.columns]
    if absent:
        raise KeyError(f'{label}: 필요한 열이 없다 -> {absent}')
    row = {'spec': label, 'estimator': estimator, 'formula': fml, 'note': note}
    try:
        m = fit(fml, data[required], estimator, fepois_kwargs)
        n, el = get_nobs(m), time.time() - t0
        row.update({'nobs': n, 'seconds': round(el, 1)})
        for term in terms:
            b, se, p = m.coef()[term], m.se()[term], m.pvalue()[term]
            pct = (np.exp(b) - 1) * 100
            prt(f'    {term:<16} beta={b:+.4f} ({se:.4f}){stars(p)}  pct={pct:+.1f}%')
            row[f'{term}_beta'], row[f'{term}_se'] = b, se
            row[f'{term}_p'], row[f'{term}_pct'] = p, pct
        if gate:
            term = terms[0]
            beta_gap = abs(row[f'{term}_beta'] - S4_EXPECTED_BETA)
            pct_gap = abs(row[f'{term}_pct'] - S4_EXPECTED_PCT)
            n_ok, beta_ok, pct_ok = (n == S4_EXPECTED_N,
                                     beta_gap <= GATE_TOL_BETA,
                                     pct_gap <= GATE_TOL_PP)
            row.update({'beta_gap': round(beta_gap, 6),
                        'pct_gap_pp': round(pct_gap, 3),
                        'replication_ok': bool(n_ok and beta_ok and pct_ok)})
            prt(f'    게이트: N {n:,} vs {S4_EXPECTED_N:,} [{"OK" if n_ok else "FAIL"}], '
                f'beta 차 {beta_gap:.4f} [{"OK" if beta_ok else "FAIL"}], '
                f'% 차 {pct_gap:.2f}pp [{"OK" if pct_ok else "FAIL"}]')
        prt(f'    N={n:,}  [{el:.0f}s]')
    except Exception as e:
        prt(f'    ERROR: {type(e).__name__}: {e}')
        row.update({'nobs': -1, 'seconds': round(time.time() - t0, 1),
                    'error': f'{type(e).__name__}: {e}',
                    'replication_ok': False if gate else None})
    results.append(row)
    checkpoint(results)
    return row


def pair_level_quartiles(d):
    """쌍 수준 분포에서 4분위를 자른다 (행 수준으로 자르면 관측 연수가 가중된다)."""
    pair = d.loc[d['pre_size'].notna(), ['partner_hs6', 'pre_size']].drop_duplicates()
    if pair.groupby('partner_hs6')['pre_size'].nunique().gt(1).any():
        raise ValueError('pre_size is not constant within partner_hs6')
    if len(pair) < 4:
        raise ValueError('유효한 partner_hs6 쌍이 4개 미만이다')

    # 동일값이 많아도 4개 그룹이 나오도록 순위 기준으로 자른다.
    # 경계에서 동일 pre_size가 갈릴 수 있으므로 경계값을 로그에 남긴다.
    pair = pair.sort_values(['pre_size', 'partner_hs6'],
                            kind='mergesort').reset_index(drop=True)
    pair['size_q'] = pd.qcut(np.arange(len(pair)), 4,
                             labels=[1, 2, 3, 4]).astype('int8')
    prt(f'  쌍 수: {len(pair):,}')
    for q in (1, 2, 3, 4):
        s = pair.loc[pair['size_q'] == q, 'pre_size']
        prt(f'    Q{q}: 쌍 {len(s):,}개, 사전 평균 무역액 '
            f'{s.min():,.2f} ~ {s.max():,.2f} (천 USD)')
    ties = pair.groupby('pre_size')['size_q'].nunique()
    n_split = int((ties > 1).sum())
    if n_split:
        prt(f'    ⚠️ 동일 pre_size가 두 분위로 갈린 값: {n_split:,}개 '
            f'(순위 기준 절단의 부작용)')
    return pair.set_index('partner_hs6')['size_q']


def fe_prune_trace(work):
    """all-zero FE 그룹과 singleton을 더 이상 없을 때까지 반복 제거하고 내역을 남긴다."""
    work = work.reset_index(drop=True)
    audit, round_no = [], 0
    while True:
        round_no += 1
        drop_now = pd.Series(False, index=work.index)
        for fe in FE_COLS:
            g = work.groupby(fe, observed=True).agg(n=('v', 'size'), vmax=('v', 'max'))
            for reason, ids in (('all_zero', g.index[g['vmax'].eq(0)]),
                                ('singleton', g.index[g['n'].eq(1)])):
                if len(ids) == 0:
                    continue
                mask = work[fe].isin(ids)
                if mask.any():
                    audit.append({'round': round_no, 'fe': fe, 'reason': reason,
                                  'groups': int(len(ids)), 'rows': int(mask.sum())})
                    drop_now |= mask
        if not drop_now.any():
            break
        work = work.loc[~drop_now].reset_index(drop=True)
    return work, pd.DataFrame(audit)


def build_balanced(panel, d):
    meta = panel[['hs6', 'sanctioned', 'sanction_year']].drop_duplicates()
    if meta.groupby('hs6').size().max() != 1:
        raise ValueError('sanction metadata not time-invariant by hs6')
    py = meta.merge(pd.DataFrame({'t': YEARS}), how='cross')
    py['sanctioned'] = py['sanctioned'].fillna(0).astype('int8')
    py['sanc_stag'] = (py['sanctioned'].eq(1)
                       & py['t'].ge(py['sanction_year'])).astype('int8')

    pairs = d[['partner', 'hs6', 'partner_hs6']].drop_duplicates()
    flow = d[['partner_hs6', 't', 'v']]
    if flow.duplicated(['partner_hs6', 't']).any():
        raise ValueError('duplicate partner_hs6-year flow')

    bal = pairs.merge(pd.DataFrame({'t': YEARS}), how='cross')
    bal = bal.merge(flow, on=['partner_hs6', 't'], how='left', validate='one_to_one')
    bal['v'] = bal['v'].fillna(0.0)
    bal = bal.merge(py[['hs6', 't', 'sanc_stag']], on=['hs6', 't'],
                    how='left', validate='many_to_one')
    if bal['sanc_stag'].isna().any():
        raise ValueError(f"{int(bal['sanc_stag'].isna().sum()):,} cells lack treatment")
    bal['sanc_stag'] = bal['sanc_stag'].astype('int8')
    bal['hs2_year'] = bal['hs6'].str[:2] + '_' + bal['t'].astype(str)
    bal['partner_year'] = bal['partner'].astype(str) + '_' + bal['t'].astype(str)

    required = ['v', 'sanc_stag', *FE_COLS]
    bad = bal[required].isna().any(axis=1) | ~np.isfinite(bal['v'].to_numpy())
    if bad.any():
        raise ValueError(f'유효하지 않은 균형패널 행: {int(bad.sum()):,}')
    return bal[required]


def d4_dropout(panel, d, fepois_kwargs, results):
    """S5에서 사라진 관측치의 원인을 실제 적합과 대조해 규명한다."""
    prt('\n' + '=' * 74)
    prt('D4 — S5 균형패널에서 사라진 관측치의 원인')
    prt('=' * 74)
    bal = build_balanced(panel, d)
    prt(f'  balanced cells: {len(bal):,} (22번 로그 {S5_BALANCED_CELLS:,})')

    prt('  S5 재적합 중...')
    flush_log()
    t0 = time.time()
    m_full = pf.fepois(f'v ~ sanc_stag | {FE}', data=bal, vcov=VCOV, **fepois_kwargs)
    n_full = get_nobs(m_full)
    prt(f'  model N = {n_full:,} (22번 로그 {S5_EXPECTED_N:,}) [{time.time()-t0:.0f}s]')

    pruned, audit = fe_prune_trace(bal)
    rule_drop = len(bal) - len(pruned)
    model_drop = len(bal) - n_full
    prt(f'  반복 FE 제거 후 행 수: {len(pruned):,}')
    prt(f'  모형이 떨어뜨린 행: {model_drop:,}')
    prt(f'  규칙으로 설명된 행: {rule_drop:,}')
    for _, r in audit.iterrows():
        prt(f"    round {int(r['round'])}: {r['fe']} {r['reason']} — "
            f"{int(r['groups']):,} groups / {int(r['rows']):,} rows")

    prt('  제거 후 표본 재적합 중...')
    flush_log()
    m_pruned = pf.fepois(f'v ~ sanc_stag | {FE}', data=pruned, vcov=VCOV, **fepois_kwargs)
    n_pruned = get_nobs(m_pruned)
    residual = len(pruned) - n_pruned
    prt(f'  제거 후 모형 N = {n_pruned:,}, 잔여 미설명 = {residual:,}')

    explained = (n_full == S5_EXPECTED_N and residual == 0
                 and rule_drop == model_drop)
    prt(f'\n  판정: {"설명됨" if explained else "미설명 — 원인을 단정하지 말 것"}')
    if not explained:
        prt('  The 29,949 rows are not explained by all-zero fixed-effect groups.')

    results.append({
        'spec': 'D4: balanced-panel dropout diagnosis', 'estimator': 'PPML',
        'nobs': n_full, 'balanced_cells': len(bal), 'pruned_rows': len(pruned),
        'model_dropped': model_drop, 'rule_explained': rule_drop,
        'residual_unexplained': residual, 'explained': explained,
        'sanc_stag_beta': m_full.coef()['sanc_stag'],
        'sanc_stag_pct': (np.exp(m_full.coef()['sanc_stag']) - 1) * 100,
        'note': '; '.join(f"r{int(r['round'])} {r['fe']} {r['reason']}: "
                          f"{int(r['rows'])} rows" for _, r in audit.iterrows())})
    checkpoint(results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--skip-d4', action='store_true',
                    help='D4를 건너뛴다 (제출 전 필수 진단이므로 최종 실행에서는 쓰지 말 것)')
    ap.add_argument('--allow-version-mismatch', action='store_true',
                    help='pyfixest 버전이 달라도 진행한다 (재현 게이트 의미 약화)')
    args = ap.parse_args()

    t_start = time.time()
    prt('=' * 74)
    prt('A-5: Rerouting influence and composition diagnostics')
    prt('=' * 74)
    check_version(args.allow_version_mismatch)
    if args.skip_d4:
        prt('⚠️ --skip-d4 지정됨: D4(실종 셀 규명)를 건너뛴다. 제출 전 필수 진단이다.')
    flush_log()

    panel, d = load_sample()
    fepois_kwargs = supported_fepois_kwargs()
    results = []

    prt('\n' + '=' * 74)
    prt('D0 — S4 기준선 재현 게이트')
    prt('=' * 74)
    gate = run(d, 'D0: S4 baseline (PPML x staggered)', ['sanc_stag'], results,
               fepois_kwargs, gate=True, note='22번 S4 재현')
    if not gate.get('replication_ok'):
        prt('\n' + '!' * 74)
        prt('GATE FAILED — 22번 S4를 재현하지 못했다. 진단을 진행하지 않는다.')
        prt('!' * 74)
        raise SystemExit(1)
    prt('\nGate PASSED.')

    prt('\n' + '=' * 74)
    prt('D1 — 사전 흐름 크기 4분위별 효과 (쌍 수준 분위)')
    prt('=' * 74)
    try:
        size_map = pair_level_quartiles(d)
        dq = d[d['partner_hs6'].isin(size_map.index)].copy()
        dq['size_q'] = dq['partner_hs6'].map(size_map).astype('int8')
        for q in (1, 2, 3, 4):
            dq[f'sanc_q{q}'] = (dq['sanc_stag'] * dq['size_q'].eq(q)).astype('int8')
            nz = int(dq[f'sanc_q{q}'].sum())
            prt(f'    sanc_q{q} = 1 인 행: {nz:,}')
            if nz == 0:
                prt(f'    ⚠️ Q{q}에 처치 변이가 없다 — 계수가 삭제될 수 있다')
        run(dq, 'D1: quartile-specific effects',
            ['sanc_q1', 'sanc_q2', 'sanc_q3', 'sanc_q4'], results, fepois_kwargs,
            note='쌍 수준 4분위. 단조 증가면 large-flow 서사 지지')
    except Exception as e:
        prt(f'  D1 SKIPPED: {type(e).__name__}: {e}')
        results.append({'spec': 'D1: quartile-specific effects', 'nobs': -1,
                        'error': f'{type(e).__name__}: {e}'})
        checkpoint(results)

    prt('\n' + '=' * 74)
    prt('D2 — 사전 크기 상위 1%·5% 제외')
    prt('=' * 74)
    prt('  ⚠️ 초판 D2의 결함 (2026-08-05 감사에서 발견, 판정 무효):')
    prt('     사전 관측이 없는 쌍(pre_size NaN)이 pair_size 색인에 없어서 상위 1%와')
    prt('     **함께** 버려졌다. NaN = 2022년 이후 처음 생긴 partner x HS6 쌍이다.')
    prt('     초판 로그: 사전 관측 없는 행 5,614개, 상위 1% 제외 시 제거 12,038행')
    prt('     -> 제거분의 47%가 "큰 쌍"이 아니라 "새로 열린 무역관계"였다.')
    prt('     그런데 신규 진입이야말로 우회의 가장 직접적 후보다. 따라서 "+22.7%는')
    prt('     상위 1%에 얹혀 있다"는 판정은 그 표본으로는 성립하지 않는다.')
    prt('     아래는 두 팔을 모두 돌려 어느 쪽이 계수를 움직였는지 가린다.')
    prt('       (a) drop-new  = 초판과 동일 (상위 절단 + 신규 진입 쌍 제거)')
    prt('       (b) keep-new  = 상위 절단만 (신규 진입 쌍은 남긴다)  <- 올바른 절단')
    prt('       (c) newonly   = 신규 진입 쌍만 제거 (절단 없음)      <- 신규 진입의 순효과')
    prt('     D0 기준선은 신규 진입 쌍을 포함하므로 (b)가 D0와 like-for-like다.')

    pair_size = d.dropna(subset=['pre_size']).groupby('partner_hs6')['pre_size'].first()
    has_pre = set(pair_size.index)
    no_pre_pairs = set(d['partner_hs6'].unique()) - has_pre
    n_rows_no_pre = int((~d['partner_hs6'].isin(has_pre)).sum())
    prt(f'\n  사전 관측 있는 쌍 {len(has_pre):,} / 없는 쌍 {len(no_pre_pairs):,} '
        f'({n_rows_no_pre:,}행)')

    # (c) 신규 진입 쌍만 제거 — 절단과 분리해서 순효과를 본다
    sub_c = d[d['partner_hs6'].isin(has_pre)].copy()
    prt(f'\n  (c) 신규 진입 쌍만 제거: {len(d) - len(sub_c):,}행 제거 -> {len(sub_c):,}행')
    run(sub_c, 'D2c: excluding new-entrant pairs only (no size truncation)',
        ['sanc_stag'], results, fepois_kwargs,
        note='pre-period 관측이 없는 쌍만 제외. 상위 절단 없음')

    for pctl, tag in ((0.99, 'top 1%'), (0.95, 'top 5%')):
        thr = pair_size.quantile(pctl)
        small = set(pair_size[pair_size <= thr].index)

        # (a) 초판 재현 — 상위 절단 + 신규 진입 제거
        sub_a = d[d['partner_hs6'].isin(small)].copy()
        prt(f'\n  (a) {tag} 제외 [drop-new, 초판 재현]: 임계 {thr:,.2f} 천 USD, '
            f'{len(d) - len(sub_a):,}행 제거 -> {len(sub_a):,}행')
        run(sub_a, f'D2a: excluding {tag} by pre-size, dropping new-entrant pairs',
            ['sanc_stag'], results, fepois_kwargs,
            note=f'threshold={thr:.2f}k USD; 초판과 동일 (비교용)')

        # (b) 올바른 절단 — 상위만 자르고 신규 진입은 남긴다
        keep_b = small | no_pre_pairs
        sub_b = d[d['partner_hs6'].isin(keep_b)].copy()
        prt(f'  (b) {tag} 제외 [keep-new, 올바른 절단]: '
            f'{len(d) - len(sub_b):,}행 제거 -> {len(sub_b):,}행')
        run(sub_b, f'D2b: excluding {tag} by pre-size, keeping new-entrant pairs',
            ['sanc_stag'], results, fepois_kwargs,
            note=f'threshold={thr:.2f}k USD; D0와 like-for-like')

    prt('\n  읽는 법:')
    prt('    (b)가 D0(+22.7%)에서 크게 내려가면 -> 대형 흐름 의존이 맞다')
    prt('    (b)는 버티는데 (a)만 내려가면 -> 초판 결론은 절단이 아니라 신규 진입')
    prt('       쌍 제거가 만든 것이다. "상위 1% 의존" 서술을 폐기한다')
    prt('    (c)만으로 크게 내려가면 -> 우회 신호가 새로 열린 무역관계에 있다는 뜻이고,')
    prt('       이는 오히려 우회 해석을 뒷받침한다. 그 경우 서술을 그렇게 바꾼다')

    prt('\n' + '=' * 74)
    prt('D3 — 주요 제3국 하나씩 제외')
    prt('=' * 74)
    for code, name in LOO_COUNTRIES.items():
        sub = d[d['partner'] != code].copy()
        dropped = len(d) - len(sub)
        if dropped == 0:
            prt(f'  {name}({code}): 표본에 없음 — 건너뜀')
            continue
        prt(f'  {name}({code}) 제외: {dropped:,}행 제거 -> {len(sub):,}행')
        run(sub, f'D3: excluding {name}', ['sanc_stag'], results, fepois_kwargs,
            note=f'비교 기준은 D0 전체표본 추정치 (partner code {code})')

    if not args.skip_d4:
        d4_dropout(panel, d, fepois_kwargs, results)

    # ---- D5·D6: pre-registered stopping rules 대응 ----
    prt('\n' + '=' * 74)
    prt('D5 — 사전규칙 표본 재현 (B0_preanalysis_rules.md §B-0.2)')
    prt('=' * 74)
    prt('  사전규칙은 제3국 표본을 "고정 후보 8개국 + exposure Top-10, 최대 12~15개국,')
    prt('  소국 제외"로 정했는데 실제 구현은 비연합국 180개국 전부였다.')
    prt('  중단 기준: 사전규칙 표본에서 우회 효과가 재현되지 않으면')
    prt('  "광범위한 우회효과" 서술을 폐기한다.')
    sub5 = d[d['partner'].isin(PRERULE_COUNTRIES)].copy()
    got = sorted(PRERULE_COUNTRIES[c] for c in sub5['partner'].unique())
    prt(f'  표본 국가 {len(got)}/10: {", ".join(got)}')
    prt(f'  {len(d):,}행 -> {len(sub5):,}행')
    run(sub5, 'D5: pre-registered gateway sample (10 countries)', ['sanc_stag'],
        results, fepois_kwargs,
        note='B-0.2 고정후보 8 + UZB·TJK. 180개국 전체표본 D0와 대조')

    prt('\n' + '=' * 74)
    prt('D6 — 2024년 제외 (BACI 최신 연도는 잠정치)')
    prt('=' * 74)
    prt('  CEPII는 최신 연도 값이 이후 버전에서 수정될 수 있다고 밝힌다.')
    prt('  중단 기준: 2024 제외로 결론이 바뀌면 동학·2차 사후연도 주장을 보류한다.')
    sub6 = d[d['t'] <= 2023].copy()
    prt(f'  {len(d):,}행 -> {len(sub6):,}행 (2024년 {len(d) - len(sub6):,}행 제거)')
    run(sub6, 'D6: excluding provisional year 2024', ['sanc_stag'],
        results, fepois_kwargs, note='표본 2015-2023. D0와 대조')

    prt('\n' + '=' * 74)
    prt('SUMMARY')
    prt('=' * 74)
    for r in results:
        if r.get('nobs', -1) <= 0:
            prt(f"  {r['spec']:<52} FAILED")
            continue
        bits = [f"{k[:-4]}={v:+.1f}%" for k, v in r.items()
                if k.endswith('_pct') and k != 'expected_pct'
                and isinstance(v, float) and v == v]
        prt(f"  {r['spec']:<52} " + (', '.join(bits) if bits else str(r.get('note', ''))))
    prt('\n판정 기준:')
    prt('  D1이 Q1<Q2<Q3<Q4로 단조 증가하고 D2b에서 상위 1% 제외 후에도 양의 유의한')
    prt('  계수가 남으면 "큰 흐름에 집중된 우회"로 서술할 수 있다.')
    prt('  ** D2는 반드시 D2b(keep-new)로 판정한다. ** D2a는 초판 재현용 비교 행이고,')
    prt('  신규 진입 쌍을 함께 버리므로 D0와 like-for-like가 아니다.')
    prt('    · D2b가 +9% 부근으로 내려감 -> large-flow 서사를 버리고 영향력 문제로 서술')
    prt('    · D2b는 버티는데 D2a만 내려감 -> "상위 1% 의존" 서술을 폐기.')
    prt('      초판 결론은 절단이 아니라 신규 진입 쌍 제거가 만든 것이다')
    prt('    · D2c(신규 진입만 제거)만으로 크게 내려감 -> 우회 신호가 새로 열린')
    prt('      무역관계에 있다는 뜻이고, 이는 오히려 우회 해석을 뒷받침한다')
    prt('  D3에서 한 나라를 빼는 것만으로 계수가 무너지면 그 나라 의존성을 명시해야 한다.')
    prt('  If D4 ends unexplained, the cause of the missing cells is not established.')
    prt('  D5(사전규칙 10개국)에서 효과가 재현되지 않으면 "광범위한 우회효과"를 폐기한다.')
    prt('  D6(2024 제외)으로 결론이 바뀌면 동학·2차 사후연도 주장을 보류한다.')
    prt(f'\nTotal: {(time.time()-t_start)/60:.1f} min')
    prt(f'Saved: {CSV_PATH}')


if __name__ == '__main__':
    try:
        main()
    finally:
        flush_log()
