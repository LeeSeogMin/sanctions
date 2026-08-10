#!/usr/bin/env python3
"""Aggregate event study estimated by PPML with high-dimensional fixed effects.

Produces the dynamic coefficients plotted in Figure 2 and tabulated in
Online Appendix K.

Specifications (each written to CSV as soon as it completes)
  S1  PPML static, full sample
  S2  PPML event study, full sample (event-time dummies x coalition, ref = -1)
  S3  PPML event study, top-50 partners
  S4  OLS event study, for the equal-weighted comparison in Appendix K

Sample and variable construction follow the static specification: singleton
removal, event time binned at [-5, +2], reference period t-1.

Output: phase4_analysis/results/rev_ppml_*.csv
"""

import pandas as pd
import numpy as np
import pyfixest as pf
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

PANEL_PATH = Path('phase4_analysis/processed/baci_russia_panel.parquet')
RESULTS_DIR = Path('phase4_analysis/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def prt(msg):
    print(msg, flush=True)


def get_nobs(m):
    try:
        return m.nobs
    except Exception:
        try:
            return m._N
        except Exception:
            return -1


def et_col(et):
    """이벤트 타임 더미 변수명 (음수는 m, 양수·0은 p 접두)"""
    et = int(et)
    return f'et_m{abs(et)}_x_coal' if et < 0 else f'et_p{et}_x_coal'


def load_data():
    """06b_ppml_focused.py와 동일한 전처리"""
    prt('=== Loading Panel ===')
    panel = pd.read_parquet(PANEL_PATH)
    imp = panel[panel['direction'] == 'to_russia'].copy()
    prt(f'Full sample: {len(imp):,} rows')

    # 싱글턴 제거 (06b와 동일)
    grp = imp.groupby('partner_hs6')['t'].nunique()
    multi = grp[grp > 1].index
    imp = imp[imp['partner_hs6'].isin(multi)].copy()
    prt(f'After singleton removal: {len(imp):,} rows')

    # Event time (06b와 동일)
    imp['event_time'] = np.nan
    mask = imp['sanctioned'] == 1
    imp.loc[mask, 'event_time'] = imp.loc[mask, 't'] - imp.loc[mask, 'sanction_year']
    imp['et_bin'] = imp['event_time'].copy()
    imp.loc[imp['et_bin'] < -5, 'et_bin'] = -5
    imp.loc[imp['et_bin'] > 2, 'et_bin'] = 2
    imp['et_bin'] = imp['et_bin'].fillna(-99)

    # Event-time × coalition 더미 (ref = -1; 비제재 상품은 et_bin=-99 → 전부 0 = 통제군)
    # 변수명에 음수 하이픈이 들어가면 formula 파서가 뺄셈으로 해석하므로 m/p 표기 사용
    event_times = sorted([et for et in imp['et_bin'].unique() if et not in (-99, -1)])
    for et in event_times:
        imp[et_col(et)] = ((imp['et_bin'] == et) & (imp['coalition_tier1'] == 1)).astype(int)
    prt(f'Event times (excl. ref=-1): {event_times}')

    imp['ln_v'] = np.log(imp['v'] + 1)

    # Top-50 파트너 서브셋 (06b와 동일)
    top50 = imp.groupby('partner')['v'].sum().nlargest(50).index
    imp_top = imp[imp['partner'].isin(top50)].copy()
    grp = imp_top.groupby('partner_hs6')['t'].nunique()
    multi = grp[grp > 1].index
    imp_top = imp_top[imp_top['partner_hs6'].isin(multi)].copy()
    prt(f'Top-50 subset: {len(imp_top):,} rows, {imp_top["partner_hs6"].nunique():,} FE groups')

    return imp, imp_top, event_times


def extract_es(m, event_times, label):
    """이벤트 스터디 계수 추출 (ref=-1 포함)"""
    rows = []
    for et in event_times:
        col = et_col(et)
        beta = m.coef()[col]
        se = m.se()[col]
        pval = m.pvalue()[col]
        rows.append({
            'spec': label, 'event_time': int(et), 'beta': beta, 'se': se,
            'pval': pval, 'pct_effect': (np.exp(beta) - 1) * 100,
            'ci_lower': beta - 1.96 * se, 'ci_upper': beta + 1.96 * se,
            'nobs': get_nobs(m),
        })
    rows.append({'spec': label, 'event_time': -1, 'beta': 0.0, 'se': 0.0,
                 'pval': 1.0, 'pct_effect': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0,
                 'nobs': get_nobs(m)})
    return pd.DataFrame(rows).sort_values('event_time')


def print_es(es_df):
    for _, r in es_df.iterrows():
        sig = '***' if r['pval'] < 0.001 else '**' if r['pval'] < 0.01 else '*' if r['pval'] < 0.05 else ''
        prt(f"  t={int(r['event_time']):+d}: β={r['beta']:.4f} ({r['se']:.4f}){sig} [{r['pct_effect']:.1f}%]")
    pre = es_df[es_df['event_time'] < -1]
    if len(pre) > 0:
        prt(f"  Pre-trend: min p = {pre['pval'].min():.4f}, any sig (p<0.05): {(pre['pval'] < 0.05).any()}")


if __name__ == '__main__':
    t_start = time.time()
    imp, imp_top, event_times = load_data()
    et_vars = ' + '.join([et_col(et) for et in event_times])
    FE = 'partner_hs6 + hs6_year + partner_year'
    VCOV = {'CRV1': 'partner_hs6'}

    # ── S1: PPML ATE 전체 표본 (파일럿 겸 robustness) ──
    prt('\n' + '=' * 60)
    prt('S1: PPML ATE — FULL SAMPLE (pilot)')
    prt('=' * 60)
    t0 = time.time()
    if (RESULTS_DIR / 'rev_ppml_ate_full.csv').exists():
        prt('  Skip: rev_ppml_ate_full.csv 이미 존재 (이전 실행 완료분)')
    else:
        try:
            m = pf.fepois(f'v ~ treated_t1 | {FE}', data=imp, vcov=VCOV)
            beta = m.coef()['treated_t1']
            se = m.se()['treated_t1']
            pval = m.pvalue()['treated_t1']
            elapsed = time.time() - t0
            prt(f'  β = {beta:.4f} ({se:.4f}), p = {pval:.2e}, '
                f'% = {(np.exp(beta)-1)*100:.1f}%, N = {get_nobs(m):,} ({elapsed:.0f}s)')
            pd.DataFrame([{
                'spec': 'PPML ATE full sample', 'beta': beta, 'se': se, 'pval': pval,
                'pct_effect': (np.exp(beta) - 1) * 100, 'nobs': get_nobs(m),
                'elapsed_sec': elapsed,
            }]).to_csv(RESULTS_DIR / 'rev_ppml_ate_full.csv', index=False)
            prt('  Saved: rev_ppml_ate_full.csv')
        except Exception as e:
            prt(f'  S1 ERROR ({time.time()-t0:.0f}s): {e}')

    # ── S2: PPML Event Study 전체 표본 ──
    prt('\n' + '=' * 60)
    prt('S2: PPML EVENT STUDY — FULL SAMPLE')
    prt('=' * 60)
    t0 = time.time()
    try:
        m = pf.fepois(f'v ~ {et_vars} | {FE}', data=imp, vcov=VCOV)
        prt(f'  Converged in {time.time()-t0:.0f}s')
        es = extract_es(m, event_times, 'PPML full sample')
        print_es(es)
        es.to_csv(RESULTS_DIR / 'rev_ppml_event_study_full.csv', index=False)
        prt('  Saved: rev_ppml_event_study_full.csv')
    except Exception as e:
        prt(f'  S2 ERROR ({time.time()-t0:.0f}s): {e}')

    # ── S3: PPML Event Study Top-50 (본문 PPML ATE와 동일 표본) ──
    prt('\n' + '=' * 60)
    prt('S3: PPML EVENT STUDY — TOP-50 PARTNERS')
    prt('=' * 60)
    t0 = time.time()
    try:
        m = pf.fepois(f'v ~ {et_vars} | {FE}', data=imp_top, vcov=VCOV)
        prt(f'  Converged in {time.time()-t0:.0f}s')
        es = extract_es(m, event_times, 'PPML Top-50')
        print_es(es)
        es.to_csv(RESULTS_DIR / 'rev_ppml_event_study_top50.csv', index=False)
        prt('  Saved: rev_ppml_event_study_top50.csv')
    except Exception as e:
        prt(f'  S3 ERROR ({time.time()-t0:.0f}s): {e}')

    # ── S4: OLS Event Study 재현 (b3 대조 검증) ──
    prt('\n' + '=' * 60)
    prt('S4: OLS EVENT STUDY — FULL SAMPLE (replication check vs b3)')
    prt('=' * 60)
    t0 = time.time()
    try:
        m = pf.feols(f'ln_v ~ {et_vars} | {FE}', data=imp, vcov=VCOV)
        prt(f'  Done in {time.time()-t0:.0f}s')
        es = extract_es(m, event_times, 'OLS full sample (replication)')
        print_es(es)
        es.to_csv(RESULTS_DIR / 'rev_ols_event_study_check.csv', index=False)
        prt('  Saved: rev_ols_event_study_check.csv')
    except Exception as e:
        prt(f'  S4 ERROR ({time.time()-t0:.0f}s): {e}')

    prt(f'\n=== Total: {(time.time()-t_start)/60:.1f} min ===')
