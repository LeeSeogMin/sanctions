#!/usr/bin/env python3
"""Third-country supplier replacement estimated by PPML with staggered timing.

Estimates Equation (4) of the paper: non-coalition exports to Russia of
sanctioned products, with treatment switching on in each product's own
restriction year.

Two things change relative to a pooled post-2022 dummy estimated by OLS. The
estimator becomes PPML, which fits a multiplicative conditional mean in levels;
and the treatment indicator becomes staggered, matching how treatment is
defined everywhere else in the paper. Both variants are reported so the two
sources of difference can be read separately.

Fixed effects throughout: partner x HS6, partner x year, HS2 x year.

The coefficient is supplier replacement in the observed non-coalition sample.
It does not by itself identify physical rerouting; see 27 for the influence
diagnostics that qualify it.
"""

import pandas as pd
import numpy as np
import pyfixest as pf
from pathlib import Path
import inspect
import time

PANEL_PATH = Path('phase4_analysis/processed/baci_russia_panel.parquet')
RESULTS_DIR = Path('phase4_analysis/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = RESULTS_DIR / 'b22_rerouting_log.txt'
CSV_PATH = RESULTS_DIR / 'b22_rerouting_ppml_staggered.csv'

EU27 = {40, 56, 100, 191, 196, 203, 208, 233, 246, 251, 276, 300,
        348, 372, 380, 428, 440, 442, 470, 528, 616, 620, 642,
        703, 705, 724, 752}
TIER1_NON_EU = {826, 842, 124, 392, 579, 757}
TIER1 = EU27 | TIER1_NON_EU
RUSSIA = 643
YEARS = np.arange(2015, 2025)
VCOV = {'CRV1': 'partner_hs6'}
FE = 'partner_hs6 + partner_year + hs2_year'

# 수렴·분리 진단을 명시적으로 켠다. pyfixest 버전이 인자를 받지 않으면 기본값으로 폴백.
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
    """사양 하나 끝날 때마다 저장. 중간에 죽어도 앞 결과는 남는다."""
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


def fit(fml, data, estimator, fepois_kwargs):
    if estimator == 'OLS':
        return pf.feols(fml, data=data, vcov=VCOV)
    return pf.fepois(fml, data=data, vcov=VCOV, **fepois_kwargs)


def load_third_country_sample():
    prt('=== Loading panel ===')
    panel = pd.read_parquet(PANEL_PATH)
    prt(f'Panel: {len(panel):,} rows')

    is_third = (~panel['partner'].isin(TIER1)) & (panel['partner'] != RUSSIA)
    to_rus = panel[(panel['direction'] == 'to_russia') & is_third].copy()
    prt(f'Third -> Russia (raw): {len(to_rus):,}')

    # Singleton removal (09_review_response.py와 동일 절차)
    grp = to_rus.groupby('partner_hs6')['t'].nunique()
    to_rus = to_rus[to_rus['partner_hs6'].isin(grp[grp > 1].index)].copy()
    prt(f'Third -> Russia (after singletons): {len(to_rus):,}')

    to_rus['post'] = (to_rus['t'] >= 2022).astype('int8')
    to_rus['ln_v'] = np.log(to_rus['v'] + 1)
    to_rus['sanc_post'] = (to_rus['sanctioned'].fillna(0).astype(int) * to_rus['post']).astype('int8')
    to_rus['sanc_stag'] = to_rus['hs6_sanctioned_at_t'].astype('int8')
    to_rus['hs2_year'] = to_rus['hs6'].str[:2] + '_' + to_rus['t'].astype(str)

    prt(f"  sanc_post = 1 : {to_rus['sanc_post'].sum():,} ({to_rus['sanc_post'].mean()*100:.1f}%)")
    prt(f"  sanc_stag = 1 : {to_rus['sanc_stag'].sum():,} ({to_rus['sanc_stag'].mean()*100:.1f}%)")
    prt(f"  partners: {to_rus['partner'].nunique()}, HS6: {to_rus['hs6'].nunique():,}, "
        f"years: {sorted(to_rus['t'].unique())}")
    return panel, to_rus


def product_year_treatment(panel):
    """제재 마스터에서 (hs6, t) 처치를 재생성한다. 관측표본에 의존하지 않는다."""
    meta = panel[['hs6', 'sanctioned', 'sanction_year']].drop_duplicates()
    dup = meta.groupby('hs6').size()
    if dup.max() != 1:
        bad = dup[dup > 1].index.tolist()[:5]
        raise ValueError(f'sanction metadata not time-invariant by hs6 (예: {bad})')

    py = meta.merge(pd.DataFrame({'t': YEARS}), how='cross')
    py['sanctioned'] = py['sanctioned'].fillna(0).astype('int8')
    py['sanc_stag'] = (py['sanctioned'].eq(1) & py['t'].ge(py['sanction_year'])).astype('int8')
    py['sanc_post'] = (py['sanctioned'].eq(1) & py['t'].ge(2022)).astype('int8')
    prt(f'  product-year treatment grid: {len(py):,} rows '
        f'(sanc_stag=1: {py["sanc_stag"].sum():,}, sanc_post=1: {py["sanc_post"].sum():,})')
    return py[['hs6', 't', 'sanctioned', 'sanc_stag', 'sanc_post']]


def build_balanced(panel, to_rus):
    """관측된 partner-HS6 쌍을 전 연도로 확장하고 결측 흐름을 0으로 채운다."""
    prt('\n=== Building balanced panel (zeros filled) ===')
    py = product_year_treatment(panel)

    pairs = to_rus[['partner', 'hs6', 'partner_hs6']].drop_duplicates()
    prt(f'  pairs: {len(pairs):,} x years: {len(YEARS)} -> {len(pairs)*len(YEARS):,} cells')

    flow = to_rus[['partner_hs6', 't', 'v']]
    if flow.duplicated(['partner_hs6', 't']).any():
        raise ValueError('duplicate partner_hs6-year flow in observed sample')

    bal = pairs.merge(pd.DataFrame({'t': YEARS}), how='cross')
    bal = bal.merge(flow, on=['partner_hs6', 't'], how='left', validate='one_to_one')
    bal['v'] = bal['v'].fillna(0.0)
    bal = bal.merge(py, on=['hs6', 't'], how='left', validate='many_to_one')

    missing = bal['sanc_stag'].isna().sum()
    if missing:
        raise ValueError(f'{missing:,} balanced cells lack product-year treatment')
    bal['sanc_stag'] = bal['sanc_stag'].astype('int8')

    bal['hs2_year'] = bal['hs6'].str[:2] + '_' + bal['t'].astype(str)
    bal['partner_year'] = bal['partner'].astype(str) + '_' + bal['t'].astype(str)

    zero_share = (bal['v'] == 0).mean() * 100
    prt(f'  balanced rows: {len(bal):,}, zeros: {zero_share:.1f}%, '
        f'sanc_stag=1: {bal["sanc_stag"].sum():,}')
    # 추정에 쓰는 열만 남긴다 (메모리 피크 완화)
    return bal[['v', 'sanc_stag', 'partner_hs6', 'partner_year', 'hs2_year']].copy()


def run_spec(data, label, estimator, treat_var, results, fepois_kwargs):
    y = 'v' if estimator == 'PPML' else 'ln_v'
    fml = f'{y} ~ {treat_var} | {FE}'
    prt(f'\n--- {label} ---')
    prt(f'    {estimator}: {fml}')
    t0 = time.time()
    try:
        required = [y, treat_var, 'partner_hs6', 'partner_year', 'hs2_year']
        absent = [c for c in required if c not in data.columns]
        if absent:
            raise KeyError(f'필요한 열이 없다 -> {absent}')
        m = fit(fml, data[required], estimator, fepois_kwargs)
        b, se, p = m.coef()[treat_var], m.se()[treat_var], m.pvalue()[treat_var]
        pct = (np.exp(b) - 1) * 100
        n, el = get_nobs(m), time.time() - t0
        prt(f'    beta={b:+.4f} ({se:.4f}){stars(p)}  pct={pct:+.1f}%  N={n:,}  [{el:.0f}s]')
        results.append({'spec': label, 'estimator': estimator, 'treatment': treat_var,
                        'beta': b, 'se': se, 'pval': p, 'pct': pct, 'nobs': n,
                        'seconds': round(el, 1)})
    except Exception as e:
        prt(f'    ERROR: {type(e).__name__}: {e}')
        results.append({'spec': label, 'estimator': estimator, 'treatment': treat_var,
                        'beta': np.nan, 'se': np.nan, 'pval': np.nan, 'pct': np.nan,
                        'nobs': -1, 'seconds': round(time.time() - t0, 1), 'error': str(e)})
    checkpoint(results)


def main():
    t_start = time.time()
    prt('=' * 74)
    prt('A-1: Rerouting Eq.(3) — PPML x staggered grid (EJPE 2nd revision)')
    prt('=' * 74)

    panel, to_rus = load_third_country_sample()
    fepois_kwargs = supported_fepois_kwargs()
    results = []

    run_spec(to_rus, 'S1: OLS  x post-dummy  (current preferred, replication)', 'OLS', 'sanc_post', results, fepois_kwargs)
    run_spec(to_rus, 'S2: OLS  x staggered   (current robustness, replication)', 'OLS', 'sanc_stag', results, fepois_kwargs)
    run_spec(to_rus, 'S3: PPML x post-dummy  (new)', 'PPML', 'sanc_post', results, fepois_kwargs)
    run_spec(to_rus, 'S4: PPML x staggered   (new preferred candidate)', 'PPML', 'sanc_stag', results, fepois_kwargs)

    try:
        bal = build_balanced(panel, to_rus)
        del panel, to_rus
        run_spec(bal, 'S5: PPML x staggered, balanced panel with zeros', 'PPML', 'sanc_stag', results, fepois_kwargs)
    except Exception as e:
        prt(f'\nS5 SKIPPED: {type(e).__name__}: {e}')
        results.append({'spec': 'S5: PPML x staggered, balanced panel with zeros',
                        'estimator': 'PPML', 'treatment': 'sanc_stag', 'nobs': -1,
                        'error': f'{type(e).__name__}: {e}'})
        checkpoint(results)

    prt('\n' + '=' * 74)
    prt('SUMMARY')
    prt('=' * 74)
    for r in results:
        if r.get('nobs', -1) > 0 and r.get('pct') == r.get('pct'):
            prt(f"  {r['spec']:<58} {r['pct']:+7.1f}%  (beta={r['beta']:+.4f}, N={r['nobs']:,})")
        else:
            prt(f"  {r['spec']:<58} FAILED")
    prt('\n비교 기준: S1은 기존 +9.6%, S2는 기존 +9.0%를 재현해야 한다.')
    prt(f'Total: {(time.time()-t_start)/60:.1f} min')
    prt(f'Saved: {CSV_PATH}')
    flush_log()


if __name__ == '__main__':
    try:
        main()
    finally:
        flush_log()
