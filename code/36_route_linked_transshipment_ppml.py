#!/usr/bin/env python3
"""Route-linked transshipment diagnostic across two trade legs.

  leg 1  sanctioning jurisdiction -> gateway, imports of a given HS6
  leg 2  the same gateway -> Russia, exports of the same HS6

The stored upstream gateway results are confirmed first. The model then takes
the gateway's exports to Russia as the outcome and interacts inflows from
sanctioning jurisdictions with restriction timing, contemporaneously and at one-
and two-year lags.

This is a mechanism diagnostic for whether the two legs move together. Inflows
are endogenous, so the coefficients are not read as a causal transshipment
effect.

Linking the legs requires filling gateway x HS6 x year cells in which one leg
is unobserved with zeros, so the build, pilot and full stages stop without
--confirm-zeros-are-true-absences. The default stage is the gate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from revision_methodology_utils import (
    BACI_DIR,
    OutputStore,
    PROCESSED_DIR,
    RESULTS_DIR,
    SANCTIONS_MASTER,
    YEARS,
    assert_no_duplicate,
    attach_ekmn_treatment,
    build_ekmn_lookup,
    check_pyfixest_version,
    ekmn_country_codes,
    ensure_inputs,
    gateway_country_codes,
    require_zero_confirmation,
    russia_code,
    run_and_store,
    sample_audit,
    save_audit,
    select_hs6_pilot,
    supported_fepois_kwargs,
)


B26_PATH = RESULTS_DIR / "b26_gateway_ppml.csv"
CSV_PATH = RESULTS_DIR / "b36_route_linked_transshipment.csv"
LOG_PATH = RESULTS_DIR / "b36_route_linked_transshipment_log.txt"
AUDIT_PATH = RESULTS_DIR / "b36_route_linked_sample_audit.csv"
CACHE_PATH = PROCESSED_DIR / "b36_route_linked_panel.parquet"
CACHE_META = PROCESSED_DIR / "b36_route_linked_panel.meta.json"
CACHE_VERSION = 1

FE = "gateway_hs6 + gateway_year + hs6_year"
VCOV = {"CRV1": "gateway_hs6"}
EXPECTED_GATE = {
    "G7:": ("OLS", 2.587255),
    "G9:": ("OLS", 2.598969),
    "G8:": ("PPML", 3.627224),
    "G10:": ("PPML", 3.848843),
}
EXPECTED_GATE_N = 8_377_907
GATE_TOL_PP = 0.2


def verify_b26_gate(store: OutputStore) -> None:
    ensure_inputs((B26_PATH,))
    results = pd.read_csv(B26_PATH)
    failures = []
    for prefix, (estimator, expected_pct) in EXPECTED_GATE.items():
        rows = results[
            results["spec"].str.startswith(prefix) & results["estimator"].eq(estimator)
        ]
        if len(rows) != 1:
            failures.append(f"{prefix} 행 수={len(rows)}")
            continue
        row = rows.iloc[0]
        n_ok = int(row["nobs"]) == EXPECTED_GATE_N
        pct_gap = abs(float(row["pct"]) - expected_pct)
        pct_ok = pct_gap <= GATE_TOL_PP
        store.log(
            f"{prefix} {estimator}: pct={float(row['pct']):+.3f}, N={int(row['nobs']):,}; "
            f"{'OK' if n_ok and pct_ok else 'FAIL'}"
        )
        if not (n_ok and pct_ok):
            failures.append(f"{prefix} N_ok={n_ok}, pct_gap={pct_gap:.3f}")
    if failures:
        raise SystemExit("26번 G7~G10 게이트 실패: " + "; ".join(failures))
    store.log("G7~G10 stored-result gate PASSED.")


def baci_path(year: int) -> Path:
    return BACI_DIR / f"BACI_HS92_Y{year}_V202601.csv"


def cache_signature() -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "years": list(YEARS),
        "baci_dir": str(BACI_DIR),
        "sanctions_master": str(SANCTIONS_MASTER),
        "design": "EKMN inflow to gateway linked to same gateway-HS6 outflow to Russia",
    }


def product_year_treatment() -> pd.DataFrame:
    sanctions = pd.read_csv(SANCTIONS_MASTER)
    sanctions["hs6"] = sanctions["hs6"].astype(str).str.zfill(6)
    meta = sanctions[["hs6", "sanctioned", "sanction_year"]].drop_duplicates()
    if meta.groupby("hs6").size().max() != 1:
        raise ValueError("제재 마스터가 HS6별로 유일하지 않다")
    grid = meta.merge(pd.DataFrame({"t": YEARS}), how="cross")
    grid["sanc_stag"] = (
        grid["sanctioned"].fillna(0).eq(1)
        & grid["sanction_year"].notna()
        & grid["t"].ge(grid["sanction_year"])
    ).astype("int8")
    return grid[["hs6", "t", "sanc_stag"]]


def build_route_panel(store: OutputStore) -> pd.DataFrame:
    ensure_inputs([baci_path(year) for year in YEARS] + [SANCTIONS_MASTER])
    gateways = gateway_country_codes()
    sanctioners = ekmn_country_codes()
    russia = russia_code()
    sanctions = pd.read_csv(SANCTIONS_MASTER, usecols=["hs6"])
    hs6_universe = sanctions["hs6"].astype(str).str.zfill(6).unique()
    lookup = build_ekmn_lookup(hs6_universe)

    frames = []
    for year in YEARS:
        started = time.time()
        raw = pd.read_csv(baci_path(year), usecols=["t", "i", "j", "k", "v"], dtype={"k": str})
        raw["hs6"] = raw["k"].astype(str).str.zfill(6)

        inbound = raw[raw["i"].isin(sanctioners) & raw["j"].isin(gateways)].copy()
        inbound = attach_ekmn_treatment(inbound, lookup=lookup, exporter_col="i")
        inbound["treated_v"] = inbound["v"] * inbound["treated_ekmn"]
        inbound = (
            inbound.groupby(["j", "hs6", "t"], as_index=False)
            .agg(inflow=("v", "sum"), treated_inflow=("treated_v", "sum"))
            .rename(columns={"j": "gateway"})
        )

        outflow = raw[raw["i"].isin(gateways) & raw["j"].eq(russia)].copy()
        outflow = (
            outflow.groupby(["i", "hs6", "t"], as_index=False)["v"]
            .sum()
            .rename(columns={"i": "gateway", "v": "outflow"})
        )
        linked = inbound.merge(outflow, on=["gateway", "hs6", "t"], how="outer")
        frames.append(linked)
        store.log(
            f"{year}: inbound cells={len(inbound):,}, outflow cells={len(outflow):,}, "
            f"linked={len(linked):,} [{time.time() - started:.0f}s]"
        )
        store.checkpoint()

    observed = pd.concat(frames, ignore_index=True)
    assert_no_duplicate(observed, ["gateway", "hs6", "t"], "route observed panel")
    pairs = observed[["gateway", "hs6"]].drop_duplicates()
    panel = pairs.merge(pd.DataFrame({"t": YEARS}), how="cross")
    panel = panel.merge(observed, on=["gateway", "hs6", "t"], how="left", validate="one_to_one")
    for column in ("inflow", "treated_inflow", "outflow"):
        panel[column] = panel[column].fillna(0.0)
    panel = panel.merge(
        product_year_treatment(), on=["hs6", "t"], how="left", validate="many_to_one"
    )
    # BACI에는 있지만 SANCTIONS_MASTER에 없는 HS6 (624개) → 비제재 → sanc_stag=0
    panel["sanc_stag"] = panel["sanc_stag"].fillna(0).astype("int8")

    panel["gateway_hs6"] = panel["gateway"].astype(str) + "_" + panel["hs6"]
    panel["gateway_year"] = panel["gateway"].astype(str) + "_" + panel["t"].astype(str)
    panel["hs6_year"] = panel["hs6"] + "_" + panel["t"].astype(str)
    panel["ln_outflow"] = np.log1p(panel["outflow"])
    panel = panel.sort_values(["gateway_hs6", "t"]).reset_index(drop=True)
    for lag in (0, 1, 2):
        source = panel["inflow"] if lag == 0 else panel.groupby("gateway_hs6")["inflow"].shift(lag)
        treated_source = (
            panel["treated_inflow"]
            if lag == 0
            else panel.groupby("gateway_hs6")["treated_inflow"].shift(lag)
        )
        panel[f"ln_inflow_l{lag}"] = np.log1p(source)
        panel[f"ln_treated_inflow_l{lag}"] = np.log1p(treated_source)
        panel[f"ln_inflow_l{lag}_x_sanc"] = panel[f"ln_inflow_l{lag}"] * panel["sanc_stag"]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(CACHE_PATH, index=False)
    CACHE_META.write_text(json.dumps(cache_signature(), indent=2), encoding="utf-8")
    store.log(f"route cache 저장: {CACHE_PATH}; rows={len(panel):,}")
    return panel


def load_route_panel(store: OutputStore, rebuild: bool) -> pd.DataFrame:
    if rebuild or not (CACHE_PATH.exists() and CACHE_META.exists()):
        return build_route_panel(store)
    try:
        metadata = json.loads(CACHE_META.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return build_route_panel(store)
    if metadata != cache_signature():
        return build_route_panel(store)
    panel = pd.read_parquet(CACHE_PATH)
    required = {
        "gateway", "hs6", "t", "inflow", "treated_inflow", "outflow",
        "sanc_stag", "gateway_hs6", "gateway_year", "hs6_year", "ln_outflow",
    }
    missing = required - set(panel.columns)
    if missing:
        store.log(f"route cache 필수 열 누락 {sorted(missing)}; 재생성한다")
        return build_route_panel(store)
    store.log(f"route cache 불러옴: {CACHE_PATH}; rows={len(panel):,}")
    return panel


def run_link_models(
    panel: pd.DataFrame,
    store: OutputStore,
    fepois_kwargs: dict,
    stage: str,
) -> None:
    work = select_hs6_pilot(panel) if stage == "pilot" else panel
    for lag in (0, 1, 2):
        main_term = f"ln_inflow_l{lag}"
        interaction = f"ln_inflow_l{lag}_x_sanc"
        treated_term = f"ln_treated_inflow_l{lag}"
        subset = work[work[[main_term, treated_term]].notna().all(axis=1)].copy()
        extra = {
            "stage": stage,
            "lag": lag,
            "causal_interpretation": False,
            "zero_share_outflow": float(subset["outflow"].eq(0).mean()),
        }
        run_and_store(
            store,
            subset,
            label=f"R{lag}: linked-route PPML lag {lag}",
            estimator="PPML",
            outcome="outflow",
            terms=[main_term, interaction],
            fe=FE,
            vcov=VCOV,
            fepois_kwargs=fepois_kwargs,
            extra=extra,
            fail_hard=False,
        )
        run_and_store(
            store,
            subset,
            label=f"R{lag}E: actual-EKMN-inflow PPML lag {lag}",
            estimator="PPML",
            outcome="outflow",
            terms=[main_term, treated_term],
            fe=FE,
            vcov=VCOV,
            fepois_kwargs=fepois_kwargs,
            extra={**extra, "treatment_definition": "actual_EKMN_inflow"},
            fail_hard=False,
        )
        run_and_store(
            store,
            subset,
            label=f"R{lag}E: actual-EKMN-inflow OLS lag {lag}",
            estimator="OLS",
            outcome="ln_outflow",
            terms=[main_term, treated_term],
            fe=FE,
            vcov=VCOV,
            extra={**extra, "treatment_definition": "actual_EKMN_inflow"},
            fail_hard=False,
        )
        run_and_store(
            store,
            subset,
            label=f"R{lag}: linked-route OLS lag {lag}",
            estimator="OLS",
            outcome="ln_outflow",
            terms=[main_term, interaction],
            fe=FE,
            vcov=VCOV,
            extra=extra,
            fail_hard=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("gate", "build", "pilot", "full"), default="gate"
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--confirm-zeros-are-true-absences", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = OutputStore(CSV_PATH, LOG_PATH)
    started = time.time()
    try:
        version = check_pyfixest_version()
        store.log("=" * 76)
        store.log("36: route-linked transshipment diagnostic")
        store.log(f"stage={args.stage}; pyfixest={version}")
        store.log("=" * 76)
        verify_b26_gate(store)
        if args.stage == "gate":
            return
        require_zero_confirmation(args.confirm_zeros_are_true_absences)
        panel = load_route_panel(store, args.rebuild)
        audit_panel = panel.assign(v=panel["outflow"])
        save_audit(
            [sample_audit(audit_panel, f"b36_{args.stage}", exporter_col="gateway")],
            AUDIT_PATH,
        )
        if args.stage in {"pilot", "full"}:
            run_link_models(panel, store, supported_fepois_kwargs(), args.stage)
        store.log(f"Finished selected stage in {(time.time() - started) / 60:.1f} min")
    finally:
        store.checkpoint()


if __name__ == "__main__":
    main()
