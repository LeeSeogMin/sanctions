#!/usr/bin/env python3
"""Category-level event studies for the 2022 treatment cohort, by OLS and PPML.

Produces Figure 3.

Design
  sample        codes sanctioned in 2022 plus never-sanctioned controls
                (the 2016 and 2023 cohorts are excluded)
  event time    e = t - 2022, unbinned
  treatment     1[e == s] x 1[sanctioned] x 1[coalition]
  s in {-7..-2, 0..2}, reference s = -1
  fixed effects partner x HS6, HS6 x year, partner x year
  inference     clustered at partner x HS6

Dummies are included for s = -7 and -6 so that the reference period is exactly
t-1 (2021). Without them those observations join the reference category and
shift every coefficient. Their estimates are stored but the plotted range stays
-5..+2. The aggregate event study (Figure 2) uses staggered timing with binned
endpoints, which absorbs s = -7 and -6 into the lower bin, so it is unaffected.

Outputs
  results/rev_category_dynamics_ols.csv
  results/rev_category_dynamics_ppml.csv
  figures/rev_category_dynamics.png
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
FIGURES_DIR = Path('phase4_analysis/figures')

CATEGORIES = ['dual_use', 'industrial_cap', 'luxury', 'military_tech']
CONTROLS = ['non_sanctioned', 'not_in_eu_list']
FE = 'partner_hs6 + hs6_year + partner_year'
VCOV = {'CRV1': 'partner_hs6'}
EVENT_TIMES = [-7, -6, -5, -4, -3, -2, 0, 1, 2]  # v3: -7,-6 더미 추가 (순수 t=-1 기준)


def prt(msg):
    print(msg, flush=True)


def et_col(et):
    et = int(et)
    return f'et_m{abs(et)}_x_coal' if et < 0 else f'et_p{et}_x_coal'


def get_nobs(m):
    try:
        return m.nobs
    except Exception:
        return -1


def load_data():
    """b66_category_pretrends(09_review_response.py)와 동일한 2022 코호트 구성"""
    prt('=== Loading Panel (2022 cohort design, b6-consistent) ===')
    panel = pd.read_parquet(PANEL_PATH)
    to_rus = panel[panel['direction'] == 'to_russia'].copy()
    to_rus['ln_v'] = np.log(to_rus['v'] + 1)

    # 2022 코호트 + 비제재 통제만 (b6과 동일; 싱글턴 제거 없음)
    to_rus['et'] = to_rus['t'] - 2022
    mask_2022 = to_rus['sanction_year'] == 2022
    df = to_rus[mask_2022 | (to_rus['sanctioned'].fillna(0) == 0)].copy()
    prt(f'2022-cohort sample: {len(df):,} rows (of {len(to_rus):,})')

    for e in EVENT_TIMES:
        df[et_col(e)] = ((df['et'] == e) &
                         (df['sanctioned'] == 1) &
                         (df['coalition_tier1'] == 1)).astype(int)
    return df


def extract_es(m, spec, category):
    rows = []
    for et in EVENT_TIMES:
        col = et_col(et)
        beta = m.coef()[col]
        se = m.se()[col]
        rows.append({
            'category': category, 'spec': spec, 'event_time': int(et),
            'beta': beta, 'se': se, 'pval': m.pvalue()[col],
            'pct_effect': (np.exp(beta) - 1) * 100,
            'ci_lower': beta - 1.96 * se, 'ci_upper': beta + 1.96 * se,
            'nobs': get_nobs(m),
        })
    rows.append({'category': category, 'spec': spec, 'event_time': -1,
                 'beta': 0.0, 'se': 0.0, 'pval': 1.0, 'pct_effect': 0.0,
                 'ci_lower': 0.0, 'ci_upper': 0.0, 'nobs': get_nobs(m)})
    return rows


def check_vs_b6(ols_df):
    """정보용 대조: 구 b6 설계(기준군 {2015,2016,2021}) 대비 계수 이동 폭 확인.
    v3는 기준군을 순수 t=-1로 교정했으므로 차이가 나는 것이 정상 — 크기만 보고."""
    b6_path = RESULTS_DIR / 'b6_category_pretrends.csv'
    if not b6_path.exists():
        prt('  (b6 파일 없음 — 대조 생략)')
        return
    b6 = pd.read_csv(b6_path)
    prt('\n=== 정보: v3(순수 t=-1 기준) vs 구 b6 설계 (beta 이동) ===')
    for _, r in ols_df[(ols_df['event_time'] != -1) & (ols_df['event_time'] >= -5)].iterrows():
        m = b6[(b6['category'] == r['category']) & (b6['event_time'] == r['event_time'])]
        if len(m) == 1:
            d = r['beta'] - m.iloc[0]['beta']
            if abs(d) > 0.02:
                prt(f"  {r['category']:16s} t={int(r['event_time']):+d}: Δβ={d:+.4f}")


def make_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    frames = []
    for f in ['rev_category_dynamics_ols.csv', 'rev_category_dynamics_ppml.csv']:
        p = RESULTS_DIR / f
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)

    titles = {'industrial_cap': 'Industrial capacity', 'dual_use': 'Dual use',
              'military_tech': 'Military technology', 'luxury': 'Luxury goods'}
    order = ['industrial_cap', 'dual_use', 'military_tech', 'luxury']
    styles = {'OLS': ('#888888', 'o', -0.1), 'PPML': ('#2166ac', 's', 0.1)}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    for ax, cat in zip(axes.flat, order):
        for spec, (color, marker, off) in styles.items():
            d = df[(df['category'] == cat) & (df['spec'] == spec) &
                   (df['event_time'] >= -5)].sort_values('event_time')  # 표시 범위 -5..+2
            if d.empty:
                continue
            pct = (np.exp(d['beta']) - 1) * 100
            lo = (np.exp(d['ci_lower']) - 1) * 100
            hi = (np.exp(d['ci_upper']) - 1) * 100
            ax.errorbar(d['event_time'] + off, pct, yerr=[pct - lo, hi - pct],
                        fmt=marker + '-', color=color, label=spec, capsize=3,
                        markersize=5, linewidth=1.3, alpha=0.9)
        ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.6)
        ax.axvline(-0.5, color='red', ls=':', lw=1, alpha=0.6)
        ax.set_title(titles[cat], fontsize=12)
        ax.legend(frameon=False, fontsize=9)
        ax.set_xticks(range(-5, 3))
    for ax in axes[1]:
        ax.set_xlabel('Event time (years relative to 2022 onset)', fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel('Effect on trade value (%, 95% CI)', fontsize=11)
    plt.tight_layout()
    out = FIGURES_DIR / 'rev_category_dynamics.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    prt(f'Saved: {out}')
    plt.close()


if __name__ == '__main__':
    t_start = time.time()
    df = load_data()
    et_vars = ' + '.join(et_col(e) for e in EVENT_TIMES)

    subsamples = {}
    for cat in CATEGORIES:
        sub = df[(df['category'] == cat) | (df['category'].isin(CONTROLS))].copy()
        subsamples[cat] = sub
        prt(f'{cat}: {len(sub):,} rows')
    subsamples['ALL'] = df.copy()  # Table B1의 ALL 행 갱신용 (OLS만)
    prt(f'ALL: {len(df):,} rows')

    # ── Pass 1: OLS (b6 재현 검증 겸) ──
    prt('\n' + '=' * 60)
    prt('PASS 1: OLS category event studies (2022 cohort)')
    prt('=' * 60)
    ols_rows = []
    for cat in CATEGORIES + ['ALL']:
        t0 = time.time()
        try:
            m = pf.feols(f'ln_v ~ {et_vars} | {FE}', data=subsamples[cat], vcov=VCOV)
            rows = extract_es(m, 'OLS', cat)
            ols_rows += rows
            post = {r['event_time']: r['pct_effect'] for r in rows if r['event_time'] >= 0}
            prt(f'  {cat:16s}: t0={post.get(0, np.nan):+.1f}% t1={post.get(1, np.nan):+.1f}% '
                f't2={post.get(2, np.nan):+.1f}% ({time.time()-t0:.0f}s)')
        except Exception as e:
            prt(f'  {cat} OLS ERROR: {e}')
    ols_df = pd.DataFrame(ols_rows)
    ols_df.to_csv(RESULTS_DIR / 'rev_category_dynamics_ols.csv', index=False)
    prt('Saved: rev_category_dynamics_ols.csv')
    check_vs_b6(ols_df)
    make_figure()

    # ── Pass 2: PPML (완료 시마다 저장) ──
    prt('\n' + '=' * 60)
    prt('PASS 2: PPML category event studies (2022 cohort)')
    prt('=' * 60)
    ppml_rows = []
    for cat in CATEGORIES:
        t0 = time.time()
        prt(f'\n--- PPML {cat} ---')
        try:
            m = pf.fepois(f'v ~ {et_vars} | {FE}', data=subsamples[cat], vcov=VCOV)
            rows = extract_es(m, 'PPML', cat)
            ppml_rows += rows
            post = {r['event_time']: r['pct_effect'] for r in rows if r['event_time'] >= 0}
            displayed_pre = [r for r in rows if -5 <= r['event_time'] < -1]
            displayed_pre_sig = [r for r in displayed_pre if r['pval'] < 0.05]
            prt(f'  Converged {time.time()-t0:.0f}s: t0={post.get(0, np.nan):+.1f}% '
                 f't1={post.get(1, np.nan):+.1f}% t2={post.get(2, np.nan):+.1f}%, '
                f'displayed pre-trend sig: {len(displayed_pre_sig)}/{len(displayed_pre)}')
            pd.DataFrame(ppml_rows).to_csv(RESULTS_DIR / 'rev_category_dynamics_ppml.csv', index=False)
            prt('  Saved (누적): rev_category_dynamics_ppml.csv')
        except Exception as e:
            prt(f'  {cat} PPML ERROR ({time.time()-t0:.0f}s): {e}')

    make_figure()
    prt(f'\n=== Total: {(time.time()-t_start)/60:.1f} min ===')
