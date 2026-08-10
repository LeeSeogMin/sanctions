#!/usr/bin/env python3
"""Supplier replacement separated from the direct effect, with exclusions.

Confirms the stored baseline as a gate, then runs the same fixed effects and
treatment on

  all non-coalition exporters; excluding China; excluding India; China alone;
  India alone; excluding the top 1% of pre-sanctions pairs; excluding China and
  the top 1% together

The positive stage compares the two estimators on the same positive-flow
sample. The balanced stage expands observed partner x HS6 pairs across all
years, which reads unobserved cells as zeros, and therefore does not run
without an explicit confirmation flag.

The estimand is supplier replacement, not causal rerouting.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from revision_methodology_utils import (
    OutputStore,
    PANEL_PATH,
    PRE_YEARS,
    RESULTS_DIR,
    YEARS,
    assert_no_duplicate,
    check_pyfixest_version,
    ensure_inputs,
    iso3_to_code,
    load_country_codes,
    require_zero_confirmation,
    run_and_store,
    russia_code,
    sample_audit,
    save_audit,
    select_hs6_pilot,
    supported_fepois_kwargs,
)


B22_PATH = RESULTS_DIR / "b22_rerouting_ppml_staggered.csv"
CSV_PATH = RESULTS_DIR / "b35_supplier_replacement.csv"
LOG_PATH = RESULTS_DIR / "b35_supplier_replacement_log.txt"
AUDIT_PATH = RESULTS_DIR / "b35_supplier_replacement_sample_audit.csv"
CONTRIBUTION_PATH = RESULTS_DIR / "b35_supplier_replacement_contributions.csv"

FE = "partner_hs6 + partner_year + hs2_year"
VCOV = {"CRV1": "partner_hs6"}
EXPECTED_S4_N = 447_688
EXPECTED_S4_BETA = 0.204345
GATE_TOL = 0.005


def verify_b22_gate(store: OutputStore) -> None:
    ensure_inputs((B22_PATH,))
    results = pd.read_csv(B22_PATH)
    gate = results[
        results["spec"].str.startswith("S4:")
        & results["treatment"].eq("sanc_stag")
    ]
    if len(gate) != 1:
        raise ValueError("22번 결과에서 유일한 S4 행을 찾지 못했다")
    row = gate.iloc[0]
    n_ok = int(row["nobs"]) == EXPECTED_S4_N
    beta_ok = abs(float(row["beta"]) - EXPECTED_S4_BETA) <= GATE_TOL
    store.log(
        f"22번 S4 저장 게이트: N={int(row['nobs']):,}, beta={float(row['beta']):+.6f}; "
        f"{'PASSED' if n_ok and beta_ok else 'FAILED'}"
    )
    if not (n_ok and beta_ok):
        raise SystemExit("22번 S4 게이트가 어긋나 공급대체 분석을 중단한다")


def load_sample(store: OutputStore) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    columns = [
        "t", "v", "direction", "partner", "hs6", "sanctioned", "sanction_year",
        "category", "hs6_sanctioned_at_t", "coalition_tier1", "partner_hs6",
        "partner_year",
    ]
    panel = pd.read_parquet(PANEL_PATH, columns=columns)
    codes = load_country_codes()
    code_map = iso3_to_code(codes)
    china = int(code_map["CHN"])
    india = int(code_map["IND"])
    russia = russia_code(codes)

    sample = panel[
        panel["direction"].eq("to_russia")
        & panel["coalition_tier1"].eq(0)
        & panel["partner"].ne(russia)
    ].copy()
    sample["hs6"] = sample["hs6"].astype(str).str.zfill(6)
    years_per_pair = sample.groupby("partner_hs6")["t"].nunique()
    sample = sample[
        sample["partner_hs6"].isin(years_per_pair[years_per_pair.gt(1)].index)
    ].copy()
    sample["sanc_stag"] = sample["hs6_sanctioned_at_t"].astype("int8")
    sample["ln_v"] = np.log1p(sample["v"])
    sample["hs2_year"] = sample["hs6"].str[:2] + "_" + sample["t"].astype(str)

    pre_size = (
        sample[sample["t"].isin(PRE_YEARS)]
        .groupby("partner_hs6")["v"]
        .mean()
    )
    sample["pre_size"] = sample["partner_hs6"].map(pre_size)
    threshold = float(pre_size.quantile(0.99))
    sample["is_top1_pre_pair"] = sample["pre_size"].gt(threshold).fillna(False)
    store.log(
        f"비연합국→러시아 양(+) 표본: {len(sample):,}; "
        f"국가 {sample['partner'].nunique()}, HS6 {sample['hs6'].nunique():,}"
    )
    store.log(
        f"중국 코드={china}, 인도 코드={india}; 사전 쌍 상위 1% 임계={threshold:,.3f}"
    )
    return panel, sample, china, india


def make_specs(sample: pd.DataFrame, china: int, india: int) -> dict[str, pd.DataFrame]:
    not_top = ~sample["is_top1_pre_pair"]
    specs = {
        "all_noncoalition": sample,
        "exclude_china": sample[sample["partner"].ne(china)],
        "exclude_india": sample[sample["partner"].ne(india)],
        "china_only": sample[sample["partner"].eq(china)],
        "india_only": sample[sample["partner"].eq(india)],
        "exclude_top1": sample[not_top],
        "exclude_china_and_top1": sample[sample["partner"].ne(china) & not_top],
    }
    return {name: data.copy() for name, data in specs.items()}


def run_positive_grid(
    sample: pd.DataFrame,
    china: int,
    india: int,
    store: OutputStore,
    fepois_kwargs: dict,
    stage: str,
) -> list[dict]:
    work = select_hs6_pilot(sample) if stage == "pilot" else sample
    audits = []
    for name, subset in make_specs(work, china, india).items():
        audits.append(sample_audit(subset, f"positive_{name}"))
        if subset["sanc_stag"].nunique() < 2:
            store.add_error(
                f"P-{name}", "OLS/PPML", ValueError("처치·통제 변이가 모두 있지 않다"),
                {"stage": stage, "sample": name},
            )
            continue
        extra = {
            "stage": stage,
            "sample": name,
            "zero_share": float(subset["v"].eq(0).mean()),
        }
        run_and_store(
            store,
            subset,
            label=f"P-{name}: positive OLS",
            estimator="OLS",
            outcome="ln_v",
            terms=["sanc_stag"],
            fe=FE,
            vcov=VCOV,
            extra=extra,
            fail_hard=False,
        )
        run_and_store(
            store,
            subset,
            label=f"P-{name}: positive PPML",
            estimator="PPML",
            outcome="v",
            terms=["sanc_stag"],
            fe=FE,
            vcov=VCOV,
            fepois_kwargs=fepois_kwargs,
            extra=extra,
            fail_hard=False,
        )
    return audits


def product_year_treatment(panel: pd.DataFrame) -> pd.DataFrame:
    meta = panel[["hs6", "sanctioned", "sanction_year"]].copy()
    meta["hs6"] = meta["hs6"].astype(str).str.zfill(6)
    meta = meta.drop_duplicates()
    counts = meta.groupby("hs6").size()
    if counts.max() != 1:
        raise ValueError("HS6별 제재 메타데이터가 시간불변이 아니다")
    grid = meta.merge(pd.DataFrame({"t": YEARS}), how="cross")
    grid["sanc_stag"] = (
        grid["sanctioned"].fillna(0).eq(1)
        & grid["sanction_year"].notna()
        & grid["t"].ge(grid["sanction_year"])
    ).astype("int8")
    return grid[["hs6", "t", "sanc_stag"]]


def build_balanced(panel: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    pairs = sample[
        ["partner", "hs6", "partner_hs6", "pre_size", "is_top1_pre_pair"]
    ].drop_duplicates()
    flows = sample[["partner_hs6", "t", "v"]].copy()
    assert_no_duplicate(flows, ["partner_hs6", "t"], "공급대체 양(+) 흐름")
    balanced = pairs.merge(pd.DataFrame({"t": YEARS}), how="cross")
    balanced = balanced.merge(
        flows, on=["partner_hs6", "t"], how="left", validate="one_to_one"
    )
    balanced["v"] = balanced["v"].fillna(0.0)
    balanced = balanced.merge(
        product_year_treatment(panel), on=["hs6", "t"], how="left", validate="many_to_one"
    )
    if balanced["sanc_stag"].isna().any():
        raise ValueError("balanced 공급대체 패널에 처치가 없는 행이 있다")
    balanced["partner_year"] = (
        balanced["partner"].astype(str) + "_" + balanced["t"].astype(str)
    )
    balanced["hs2_year"] = balanced["hs6"].str[:2] + "_" + balanced["t"].astype(str)
    assert_no_duplicate(balanced, ["partner_hs6", "t"], "공급대체 balanced panel")
    return balanced


def run_balanced_grid(
    balanced: pd.DataFrame,
    china: int,
    india: int,
    store: OutputStore,
    fepois_kwargs: dict,
) -> list[dict]:
    audits = []
    for name, subset in make_specs(balanced, china, india).items():
        audits.append(sample_audit(subset, f"balanced_{name}"))
        run_and_store(
            store,
            subset,
            label=f"B-{name}: balanced PPML",
            estimator="PPML",
            outcome="v",
            terms=["sanc_stag"],
            fe=FE,
            vcov=VCOV,
            fepois_kwargs=fepois_kwargs,
            extra={
                "stage": "balanced",
                "sample": name,
                "zero_share": float(subset["v"].eq(0).mean()),
            },
            fail_hard=False,
        )
    return audits


def save_contributions(sample: pd.DataFrame, codes: pd.DataFrame) -> None:
    work = sample[sample["sanctioned"].eq(1)].copy()
    work["period"] = np.where(work["t"].lt(2022), "pre", "post")
    totals = (
        work.groupby(["partner", "category", "period"])["v"]
        .sum()
        .unstack("period")
        .reset_index()
    )
    means = totals
    for period in ("pre", "post"):
        if period not in means:
            means[period] = 0.0
    means["pre_mean_annual_v"] = means["pre"].fillna(0) / len(PRE_YEARS)
    means["post_mean_annual_v"] = means["post"].fillna(0) / (len(YEARS) - len(PRE_YEARS))
    means["change_mean_annual_v"] = (
        means["post_mean_annual_v"] - means["pre_mean_annual_v"]
    )
    means = means.merge(
        codes.rename(columns={"code": "partner"}), on="partner", how="left", validate="many_to_one"
    )
    means.sort_values("change_mean_annual_v", ascending=False).to_csv(
        CONTRIBUTION_PATH, index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("gate", "pilot", "positive", "balanced", "all"),
        default="gate",
    )
    parser.add_argument("--confirm-zeros-are-true-absences", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = OutputStore(CSV_PATH, LOG_PATH)
    started = time.time()
    try:
        version = check_pyfixest_version()
        store.log("=" * 76)
        store.log("35: third-country supplier replacement")
        store.log(f"stage={args.stage}; pyfixest={version}")
        store.log("=" * 76)
        verify_b22_gate(store)
        if args.stage == "gate":
            return

        panel, sample, china, india = load_sample(store)
        fepois_kwargs = supported_fepois_kwargs()
        audits: list[dict] = []
        if args.stage in {"pilot", "positive", "all"}:
            audits.extend(
                run_positive_grid(sample, china, india, store, fepois_kwargs, args.stage)
            )
        if args.stage in {"balanced", "all"}:
            require_zero_confirmation(args.confirm_zeros_are_true_absences)
            balanced = build_balanced(panel, sample)
            audits.extend(run_balanced_grid(balanced, china, india, store, fepois_kwargs))

        save_audit(audits, AUDIT_PATH)
        save_contributions(sample, load_country_codes())
        store.log(f"기여액 표 저장: {CONTRIBUTION_PATH}")
        store.log(f"Finished selected stage in {(time.time() - started) / 60:.1f} min")
    finally:
        store.checkpoint()


if __name__ == "__main__":
    main()
