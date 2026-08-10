#!/usr/bin/env python3
"""Generate the manuscript figures from stored coefficient files.

Figures
  figure2_event_study.png            aggregate event study, PPML, full sample,
                                     percentage effects with 95% CIs
                                     (input: rev_ppml_event_study_full.csv)
  appendix_event_study_ols_ppml.png  OLS and PPML series side by side
                                     (Online Appendix K)
  figure3_category_dynamics.png      category dynamics, 2022 cohort, OLS vs
                                     PPML, four panels, with the excl. HS 30/33
                                     series on the military panel

Confidence intervals are formed on the coefficient scale as beta +/- 1.96 SE
and then transformed, (exp(beta +/- 1.96 SE) - 1) * 100, so they are asymmetric
in percentage terms and always contain the point estimate.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path('phase4_analysis/results')
WORK_FIG = Path('phase4_analysis/figures')
MS_FIG = Path('EJPE/figures')


def pct_frame(df, et_min=-5):
    d = df[df['event_time'] >= et_min].sort_values('event_time').copy()
    d['pct'] = (np.exp(d['beta']) - 1) * 100
    d['lo'] = (np.exp(d['beta'] - 1.96 * d['se']) - 1) * 100
    d['hi'] = (np.exp(d['beta'] + 1.96 * d['se']) - 1) * 100
    return d


def draw(ax, d, color, marker, off, label, ls='-'):
    ax.errorbar(d['event_time'] + off, d['pct'],
                yerr=[d['pct'] - d['lo'], d['hi'] - d['pct']],
                fmt=marker, ls=ls, color=color, label=label,
                capsize=3.5, markersize=5.5, linewidth=1.4, alpha=0.9)


def set_binned_event_ticks(ax):
    """Aggregate event-study endpoints are binned at <=-5 and >=+2."""
    ax.set_xticks(range(-5, 3))
    ax.set_xticklabels([r'$\leq -5$', '-4', '-3', '-2', '-1', '0', '+1', r'$\geq +2$'])


def figure2():
    """본문 Figure 2 — PPML 단독 (preferred estimator)"""
    ppml = pct_frame(pd.read_csv(RESULTS / 'rev_ppml_event_study_full.csv'))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    draw(ax, ppml, '#2166ac', 's', 0.0, 'PPML')
    ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.6)
    ax.axvline(-0.5, color='red', ls=':', lw=1, alpha=0.6)
    ax.set_xlabel('Event time (years relative to sanctions onset)', fontsize=11.5)
    ax.set_ylabel('Coalition–non-coalition differential (%, 95% CI)', fontsize=11.5)
    set_binned_event_ticks(ax)
    plt.tight_layout()
    out = MS_FIG / 'figure2_event_study.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def figure_appendix_k():
    """Appendix K — OLS(원 사양) vs PPML 비교"""
    ols = pct_frame(pd.read_csv(RESULTS / 'rev_ols_event_study_check.csv'))
    ppml = pct_frame(pd.read_csv(RESULTS / 'rev_ppml_event_study_full.csv'))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    draw(ax, ols, '#555555', 'o', -0.12, 'OLS, ln(v+1) (original specification)')
    draw(ax, ppml, '#2166ac', 's', 0.12, 'PPML (main text, Figure 2)')
    ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.6)
    ax.axvline(-0.5, color='red', ls=':', lw=1, alpha=0.6)
    ax.set_xlabel('Event time (years relative to sanctions onset)', fontsize=11.5)
    ax.set_ylabel('Effect on trade value (%, 95% CI)', fontsize=11.5)
    ax.legend(frameon=False, fontsize=10.5, loc='lower left')
    set_binned_event_ticks(ax)
    plt.tight_layout()
    out = MS_FIG / 'appendix_event_study_ols_ppml.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def figure2_worker():
    """작업용 3계열 비교 (Top-50 포함)"""
    ols = pct_frame(pd.read_csv(RESULTS / 'rev_ols_event_study_check.csv'))
    full = pct_frame(pd.read_csv(RESULTS / 'rev_ppml_event_study_full.csv'))
    top = pct_frame(pd.read_csv(RESULTS / 'rev_ppml_event_study_top50.csv'))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    draw(ax, ols, '#888888', 'o', -0.15, 'OLS ln(v+1), full sample')
    draw(ax, full, '#2166ac', 's', 0.0, 'PPML, full sample')
    draw(ax, top, '#b2182b', '^', 0.15, 'PPML, Top-50 partners')
    ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.6)
    ax.axvline(-0.5, color='red', ls=':', lw=1, alpha=0.6)
    ax.set_xlabel('Event time (years relative to sanction onset)', fontsize=11)
    ax.set_ylabel('Effect on trade value (%, 95% CI)', fontsize=11)
    ax.set_title('Event-Study Estimates: OLS vs. PPML (HDFE)', fontsize=12.5)
    ax.legend(frameon=False, fontsize=10)
    set_binned_event_ticks(ax)
    plt.tight_layout()
    out = WORK_FIG / 'rev_event_study_ppml_comparison.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def figure3():
    ols = pd.read_csv(RESULTS / 'rev_category_dynamics_ols.csv')
    ppml = pd.read_csv(RESULTS / 'rev_category_dynamics_ppml.csv')
    mil_ex = pd.read_csv(RESULTS / 'rev_military_expharma.csv')
    mil_ex = mil_ex[(mil_ex['design'] == 'dynamic') & (mil_ex['spec'] == 'PPML')].copy()
    mil_ex['ci_lower'] = mil_ex['beta'] - 1.96 * mil_ex['se']
    mil_ex['ci_upper'] = mil_ex['beta'] + 1.96 * mil_ex['se']

    titles = {'industrial_cap': 'Industrial capacity', 'dual_use': 'Dual use',
              'military_tech': 'Military technology', 'luxury': 'Luxury goods'}
    order = ['industrial_cap', 'dual_use', 'military_tech', 'luxury']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    for ax, cat in zip(axes.flat, order):
        draw(ax, pct_frame(ols[ols['category'] == cat]), '#888888', 'o', -0.13, 'OLS')
        draw(ax, pct_frame(ppml[ppml['category'] == cat]), '#2166ac', 's', 0.0, 'PPML')
        if cat == 'military_tech':
            draw(ax, pct_frame(mil_ex), '#b2182b', '^', 0.13, 'PPML, excl. HS 30/33', ls='--')
        ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.6)
        ax.axvline(-0.5, color='red', ls=':', lw=1, alpha=0.6)
        ax.set_title(titles[cat], fontsize=12)
        ax.legend(frameon=False, fontsize=9)
        ax.set_xticks(range(-5, 3))
    for ax in axes[1]:
        ax.set_xlabel('Event time (years relative to 2022 onset)', fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel('Coalition–non-coalition differential (%, 95% CI)', fontsize=11)
    plt.tight_layout()
    out = MS_FIG / 'figure3_category_dynamics.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    figure2()
    figure_appendix_k()
    figure2_worker()
    figure3()
