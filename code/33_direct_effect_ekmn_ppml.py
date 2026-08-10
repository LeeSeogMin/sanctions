#!/usr/bin/env python3
"""Direct effect on Russia-facing exports using jurisdiction-specific treatment.

Treatment is each exporter's own jurisdiction's sanctions list and its own
implementation dates, rather than the EU list applied to every coalition
member.

Stages run separately so that a long estimation never starts by accident. The
default stage is the reproduction gate.

  --stage gate    reproduce the stored benchmark on the identical sample
  --stage pilot   reduced sample
  --stage full    full sample
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from revision_methodology_utils import (
    OutputStore,
    PANEL_PATH,
    RESULTS_DIR,
    YEARS,
    attach_ekmn_treatment,
    build_ekmn_lookup,
    check_pyfixest_version,
    ensure_inputs,
    percent_effect,
    require_zero_confirmation,
    run_and_store,
    sample_audit,
    save_audit,
    select_hs6_pilot,
    supported_fepois_kwargs,
    assert_no_duplicate,
)


CSV_PATH = RESULTS_DIR / "b33_direct_effect_ekmn.csv"
LOG_PATH = RESULTS_DIR / "b33_direct_effect_ekmn_log.txt"
AUDIT_PATH = RESULTS_DIR / "b33_direct_effect_sample_audit.csv"

FE = "partner_hs6 + hs6_year + partner_year"
VCOV_PAIR = {"CRV1": "partner_hs6"}
VCOV_HS6 = {"CRV1": "hs6"}

EXPECTED_GATE_N = 842_264
EXPECTED_GATE_BETA = -0.525173
GATE_BETA_TOL = 0.005


def load_positive_samples(store: OutputStore) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ensure_inputs((PANEL_PATH,))
    columns = [
        "t", "v", "direction", "partner", "hs6", "treated_t1",
        "partner_hs6", "hs6_year", "partner_year",
    ]
    panel = pd.read_parquet(PANEL_PATH, columns=columns)
    to_russia = panel[panel["direction"].eq("to_russia")].copy()
    del panel
    to_russia["hs6"] = to_russia["hs6"].astype(str).str.zfill(6)

    years_per_pair = to_russia.groupby("partner_hs6")["t"].nunique()
    to_russia = to_russia[
        to_russia["partner_hs6"].isin(years_per_pair[years_per_pair.gt(1)].index)
    ].copy()

    lookup = build_ekmn_lookup(to_russia["hs6"].unique())
    to_russia = attach_ekmn_treatment(to_russia, lookup=lookup)
    to_russia["ln_v"] = np.log1p(to_russia["v"])

    top50 = to_russia.groupby("partner")["v"].sum().nlargest(50).index
    gate = to_russia[to_russia["partner"].isin(top50)].copy()
    gate_years = gate.groupby("partner_hs6")["t"].nunique()
    gate = gate[gate["partner_hs6"].isin(gate_years[gate_years.gt(1)].index)].copy()

    direct = to_russia[to_russia["ekmn_juris"].notna()].copy()
    direct_years = direct.groupby("partner_hs6")["t"].nunique()
    direct = direct[
        direct["partner_hs6"].isin(direct_years[direct_years.gt(1)].index)
    ].copy()
    direct["ekmn_juris"] = direct["ekmn_juris"].astype(str)

    store.log(f"전체 대러 양(+) 표본: {len(to_russia):,}")
    store.log(f"재현 게이트 Top-50 표본: {len(gate):,}")
    store.log(
        f"EKMN 관할국 직접효과 표본: {len(direct):,}; "
        f"수출국 {direct['partner'].nunique()}, 관할권 {direct['ekmn_juris'].nunique()}"
    )
    return gate, direct, lookup


def run_replication_gate(
    gate: pd.DataFrame,
    store: OutputStore,
    fepois_kwargs: dict,
) -> None:
    _, rows = run_and_store(
        store,
        gate,
        label="GATE: existing EKMN Top-50 PPML",
        estimator="PPML",
        outcome="v",
        terms=["treated_ekmn"],
        fe=FE,
        vcov=VCOV_PAIR,
        fepois_kwargs=fepois_kwargs,
        extra={"stage": "gate", "sample": "top50_all_exporters"},
    )
    row = rows[0]
    n_ok = int(row["nobs"]) == EXPECTED_GATE_N
    beta_gap = abs(float(row["beta"]) - EXPECTED_GATE_BETA)
    beta_ok = beta_gap <= GATE_BETA_TOL
    store.log(
        f"Gate N: {row['nobs']:,} vs {EXPECTED_GATE_N:,} "
        f"[{'OK' if n_ok else 'FAIL'}]"
    )
    store.log(
        f"Gate beta: {row['beta']:+.6f} vs {EXPECTED_GATE_BETA:+.6f}; "
        f"gap={beta_gap:.6f} [{'OK' if beta_ok else 'FAIL'}]"
    )
    if not (n_ok and beta_ok):
        raise SystemExit("재현 게이트가 어긋나 직접효과 추정을 중단한다")
    store.log("Gate PASSED.")


def run_positive_models(
    direct: pd.DataFrame,
    store: OutputStore,
    fepois_kwargs: dict,
    stage: str,
) -> None:
    sample = select_hs6_pilot(direct) if stage == "pilot" else direct
    tag = "pilot" if stage == "pilot" else "full_positive"
    store.log(f"\n직접효과 {tag} 표본: {len(sample):,}")
    if sample["treated_ekmn"].nunique() < 2:
        raise ValueError(f"{tag} 표본에 처치·통제 변이가 모두 존재하지 않는다")

    run_and_store(
        store,
        sample,
        label=f"D1: EKMN sanctioners positive PPML ({tag})",
        estimator="PPML",
        outcome="v",
        terms=["treated_ekmn"],
        fe=FE,
        vcov=VCOV_HS6,
        fepois_kwargs=fepois_kwargs,
        extra={"stage": stage, "sample": "ekmn_positive"},
    )
    run_and_store(
        store,
        sample,
        label=f"D2: EKMN sanctioners positive OLS ({tag})",
        estimator="OLS",
        outcome="ln_v",
        terms=["treated_ekmn"],
        fe=FE,
        vcov=VCOV_HS6,
        extra={"stage": stage, "sample": "ekmn_positive"},
    )

    if stage == "pilot":
        return

    for jurisdiction in sorted(sample["ekmn_juris"].unique()):
        subset = sample[~sample["ekmn_juris"].eq(jurisdiction)].copy()
        run_and_store(
            store,
            subset,
            label=f"D3: leave out EKMN jurisdiction {jurisdiction}",
            estimator="PPML",
            outcome="v",
            terms=["treated_ekmn"],
            fe=FE,
            vcov=VCOV_HS6,
            fepois_kwargs=fepois_kwargs,
            extra={
                "stage": stage,
                "sample": "ekmn_positive_loo",
                "excluded_jurisdiction": jurisdiction,
            },
            fail_hard=False,
        )

    for label, subset in (
        ("EU_only", sample[sample["ekmn_juris"].eq("EU")].copy()),
        ("non_EU", sample[~sample["ekmn_juris"].eq("EU")].copy()),
        ("exclude_2024", sample[sample["t"].lt(2024)].copy()),
    ):
        run_and_store(
            store,
            subset,
            label=f"D4: {label}",
            estimator="PPML",
            outcome="v",
            terms=["treated_ekmn"],
            fe=FE,
            vcov=VCOV_HS6,
            fepois_kwargs=fepois_kwargs,
            extra={"stage": stage, "sample": label},
            fail_hard=False,
        )


def build_balanced_direct(
    direct: pd.DataFrame,
    lookup: dict,
) -> pd.DataFrame:
    pairs = direct[["partner", "ekmn_juris", "hs6", "partner_hs6"]].drop_duplicates()
    flows = direct[["partner_hs6", "t", "v"]].copy()
    assert_no_duplicate(flows, ["partner_hs6", "t"], "EKMN 직접효과 양(+) 흐름")

    balanced = pairs.merge(pd.DataFrame({"t": YEARS}), how="cross")
    balanced = balanced.merge(
        flows, on=["partner_hs6", "t"], how="left", validate="one_to_one"
    )
    balanced["v"] = balanced["v"].fillna(0.0)
    balanced = attach_ekmn_treatment(balanced, lookup=lookup)
    balanced["partner_year"] = (
        balanced["partner"].astype(str) + "_" + balanced["t"].astype(str)
    )
    balanced["hs6_year"] = balanced["hs6"] + "_" + balanced["t"].astype(str)
    assert_no_duplicate(balanced, ["partner_hs6", "t"], "EKMN balanced panel")
    return balanced


def run_balanced_model(
    direct: pd.DataFrame,
    lookup: dict,
    store: OutputStore,
    fepois_kwargs: dict,
) -> pd.DataFrame:
    balanced = build_balanced_direct(direct, lookup)
    store.log(
        f"\nBalanced EKMN panel: {len(balanced):,}; "
        f"zeros={balanced['v'].eq(0).mean() * 100:.1f}%"
    )
    run_and_store(
        store,
        balanced,
        label="D5: EKMN sanctioners balanced PPML",
        estimator="PPML",
        outcome="v",
        terms=["treated_ekmn"],
        fe=FE,
        vcov=VCOV_HS6,
        fepois_kwargs=fepois_kwargs,
        extra={"stage": "balanced", "sample": "ekmn_observed_pairs_x_years"},
    )
    return balanced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("gate", "pilot", "full", "balanced", "all"),
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
        store.log("33: EKMN sanctioner-only direct effect")
        store.log(f"stage={args.stage}; pyfixest={version}")
        store.log("=" * 76)

        gate, direct, lookup = load_positive_samples(store)
        audits = [sample_audit(gate, "gate_top50"), sample_audit(direct, "ekmn_positive")]
        save_audit(audits, AUDIT_PATH)
        fepois_kwargs = supported_fepois_kwargs()
        run_replication_gate(gate, store, fepois_kwargs)

        if args.stage in {"pilot", "full", "all"}:
            run_positive_models(direct, store, fepois_kwargs, args.stage)

        if args.stage in {"balanced", "all"}:
            require_zero_confirmation(args.confirm_zeros_are_true_absences)
            balanced = run_balanced_model(direct, lookup, store, fepois_kwargs)
            audits.append(sample_audit(balanced, "ekmn_balanced"))
            save_audit(audits, AUDIT_PATH)

        if store.rows:
            for row in store.rows:
                if row.get("term") == "treated_ekmn" and pd.notna(row.get("beta")):
                    row["pct_check"] = percent_effect(float(row["beta"]))
            store.checkpoint()
        store.log(f"\nFinished selected stage in {(time.time() - started) / 60:.1f} min")
    finally:
        store.checkpoint()


if __name__ == "__main__":
    main()
