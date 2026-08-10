#!/usr/bin/env python3
"""Coalition and non-coalition responses estimated in one Russia-facing PPML.

Supplies the decomposition behind the influence diagnostics in Section 4.1.
Elsewhere the coalition and non-coalition margins are estimated on separate
samples; here both enter a single regression so the benchmark contrast can be
split into its two sides.

Sample: all exporters to Russia, roughly one million rows after singleton
removal.

Specifications
  B1   v ~ sanc_coal + sanc_noncoal | partner x HS6 + partner x year + HS2 x year
       coalition-side and non-coalition-side coefficients
  B1r  v ~ sanc_stag + sanc_diff | same fixed effects
       a reparameterisation of B1 that delivers the standard error of the
       contrast directly
  C2   v ~ sanc_coal | partner x HS6 + partner x year + HS6 x year
       the benchmark specification, on this sample

Identification note: under HS6 x year fixed effects the sanctioned indicator is
absorbed, so the two levels are not separately identified. Estimating them
jointly requires relaxing the product-time control to HS2 x year. That is the
cost of the decomposition, so B1 and B1r are reported as a bridge
specification rather than compared directly with the benchmark.

The non-coalition coefficient is supplier replacement, which includes
third-country production and demand substitution; it is not rerouting.

Outputs: results/b23_bridge_ppml.csv, results/b23_bridge_log.txt
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
LOG_PATH = RESULTS_DIR / 'b23_bridge_log.txt'
CSV_PATH = RESULTS_DIR / 'b23_bridge_ppml.csv'

EU27 = {40, 56, 100, 191, 196, 203, 208, 233, 246, 251, 276, 300,
        348, 372, 380, 428, 440, 442, 470, 528, 616, 620, 642,
        703, 705, 724, 752}
TIER1_NON_EU = {826, 842, 124, 392, 579, 757}
TIER1 = EU27 | TIER1_NON_EU
RUSSIA = 643
VCOV = {'CRV1': 'partner_hs6'}

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


def fit(fml, data, fepois_kwargs):
    return pf.fepois(fml, data=data, vcov=VCOV, **fepois_kwargs)


def load_to_russia():
    prt('=== Loading panel ===')
    panel = pd.read_parquet(PANEL_PATH)
    to_rus = panel[(panel['direction'] == 'to_russia') & (panel['partner'] != RUSSIA)].copy()
    del panel
    prt(f'To-Russia (raw): {len(to_rus):,}')

    grp = to_rus.groupby('partner_hs6')['t'].nunique()
    to_rus = to_rus[to_rus['partner_hs6'].isin(grp[grp > 1].index)].copy()
    prt(f'To-Russia (after singletons): {len(to_rus):,}')

    to_rus['is_coalition'] = to_rus['partner'].isin(TIER1).astype('int8')
    to_rus['sanc_stag'] = to_rus['hs6_sanctioned_at_t'].astype('int8')
    to_rus['sanc_coal'] = (to_rus['sanc_stag'] * to_rus['is_coalition']).astype('int8')
    to_rus['sanc_noncoal'] = (to_rus['sanc_stag'] * (1 - to_rus['is_coalition'])).astype('int8')
    to_rus['sanc_diff'] = to_rus['sanc_coal']  # 재parameterization에서 상호작용 항
    to_rus['hs2_year'] = to_rus['hs6'].str[:2] + '_' + to_rus['t'].astype(str)

    n_coal = int(to_rus['is_coalition'].sum())
    prt(f'  coalition rows: {n_coal:,} / non-coalition: {len(to_rus)-n_coal:,}')
    prt(f"  sanc x coal = 1: {to_rus['sanc_coal'].sum():,}, "
        f"sanc x noncoal = 1: {to_rus['sanc_noncoal'].sum():,}")

    keep = ['v', 'sanc_stag', 'sanc_coal', 'sanc_noncoal', 'sanc_diff',
            'partner_hs6', 'partner_year', 'hs2_year', 'hs6_year']
    return to_rus[keep].copy()


def run(data, label, fml, terms, results, fepois_kwargs, note=''):
    prt(f'\n--- {label} ---')
    prt(f'    PPML: {fml}')
    t0 = time.time()
    try:
        used = [t for t in terms] + [c for c in ('partner_hs6', 'partner_year',
                                                 'hs2_year', 'hs6_year') if c in fml]
        cols = ['v'] + list(dict.fromkeys(used))
        absent = [c for c in cols if c not in data.columns]
        if absent:
            raise KeyError(f'필요한 열이 없다 -> {absent}')
        m = fit(fml, data[cols], fepois_kwargs)
        el, n = time.time() - t0, get_nobs(m)
        row = {'spec': label, 'formula': fml, 'nobs': n,
               'seconds': round(el, 1), 'note': note}
        for term in terms:
            b, se, p = m.coef()[term], m.se()[term], m.pvalue()[term]
            pct = (np.exp(b) - 1) * 100
            prt(f'    {term:<13} beta={b:+.4f} ({se:.4f}){stars(p)}  pct={pct:+.1f}%')
            row[f'{term}_beta'], row[f'{term}_se'] = b, se
            row[f'{term}_p'], row[f'{term}_pct'] = p, pct
        prt(f'    N={n:,}  [{el:.0f}s]')
        results.append(row)
    except Exception as e:
        prt(f'    ERROR: {type(e).__name__}: {e}')
        results.append({'spec': label, 'formula': fml, 'nobs': -1,
                        'seconds': round(time.time() - t0, 1),
                        'error': f'{type(e).__name__}: {e}', 'note': note})
    checkpoint(results)


def main():
    t_start = time.time()
    prt('=' * 74)
    prt('A-2: Bridge PPML — direct effect and third-country response jointly')
    prt('=' * 74)

    d = load_to_russia()
    fepois_kwargs = supported_fepois_kwargs()
    results = []

    run(d, 'B1: joint levels (sanc x coal, sanc x noncoal), HS2xyear FE',
        'v ~ sanc_coal + sanc_noncoal | partner_hs6 + partner_year + hs2_year',
        ['sanc_coal', 'sanc_noncoal'], results, fepois_kwargs,
        note='bridge/decomposition estimand; b2 is NOT rerouting per se')

    run(d, 'B1r: same model reparameterized (contrast with SE)',
        'v ~ sanc_stag + sanc_diff | partner_hs6 + partner_year + hs2_year',
        ['sanc_stag', 'sanc_diff'], results, fepois_kwargs,
        note='sanc_stag = b2; sanc_diff = b1 - b2 (triple-DID contrast)')

    run(d, 'C2: single coef, HS6xyear FE (manuscript headline spec, replication)',
        'v ~ sanc_coal | partner_hs6 + partner_year + hs6_year',
        ['sanc_coal'], results, fepois_kwargs,
        note='should reproduce the -36.5% headline on this sample')

    prt('\n' + '=' * 74)
    prt('SUMMARY')
    prt('=' * 74)
    for r in results:
        if r.get('nobs', -1) > 0:
            bits = [f"{k.replace('_pct',''):<13}{v:+7.1f}%"
                    for k, v in r.items() if k.endswith('_pct') and v == v]
            prt(f"  {r['spec']}\n      " + ' | '.join(bits) + f"  (N={r['nobs']:,})")
        else:
            prt(f"  {r['spec']}  FAILED")
    prt('\n확인 사항: B1의 (sanc_coal - sanc_noncoal)가 B1r의 sanc_diff와 일치해야 한다.')
    prt('           C2 must reproduce the benchmark -36.5%.')
    prt(f'Total: {(time.time()-t_start)/60:.1f} min')
    prt(f'Saved: {CSV_PATH}')
    flush_log()


if __name__ == '__main__':
    try:
        main()
    finally:
        flush_log()
