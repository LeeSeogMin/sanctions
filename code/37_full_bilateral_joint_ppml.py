#!/usr/bin/env python3
"""Three trade channels estimated in one bilateral PPML sample.

Supplies Online Appendix N.

  direct effect         sanctioning exporter -> Russia, jurisdiction-specific
                        treatment and timing
  gateway route         sanctioning exporter -> pre-specified gateway, same
                        treatment
  supplier replacement  non-sanctioning exporter -> Russia, EU-list treatment

Fixed effects: exporter x importer x HS6, HS6 x year, exporter x year,
importer x year. The three coefficients describe different channels and are not
summed into a leakage ratio.

Scope limits
  pilot uses the gateways, the ten largest non-gateway destinations of 2019, and
  Russia-facing flows
  full uses all destinations of the sanctioning exporters plus Russia-facing
  flows from every exporter
  both use positive flows observed in BACI. This is not a worldwide panel with
  unobserved cells filled as zeros, so it is not a total or extensive-margin
  PPML

The default stage is the gate, which stops unless the results of scripts 33 and
35 are present and reproduce.
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
    russia_code,
    run_and_store,
    sample_audit,
    save_audit,
    supported_fepois_kwargs,
)


B33_PATH = RESULTS_DIR / "b33_direct_effect_ekmn.csv"
B35_PATH = RESULTS_DIR / "b35_supplier_replacement.csv"
CSV_PATH = RESULTS_DIR / "b37_full_bilateral_joint.csv"
LOG_PATH = RESULTS_DIR / "b37_full_bilateral_joint_log.txt"
AUDIT_PATH = RESULTS_DIR / "b37_full_bilateral_joint_sample_audit.csv"

FE = "exporter_importer + hs6_year + exporter_year + importer_year"
VCOV = {"CRV1": "exporter_importer"}
TERMS = ["direct_effect", "gateway_route", "supplier_replacement"]
CACHE_VERSION = 2  # FE 단순화: exporter_importer_hs6 → exporter_importer (옵션 A)

EXPECTED_B33_N = 842_264
EXPECTED_B33_BETA = -0.525173
EXPECTED_B35_N = 447_688
EXPECTED_B35_BETA = 0.204345
GATE_TOL = 0.005


def verify_upstream_results(store: OutputStore) -> None:
    ensure_inputs((B33_PATH, B35_PATH))
    b33 = pd.read_csv(B33_PATH)
    gate33 = b33[
        b33["spec"].eq("GATE: existing EKMN Top-50 PPML")
        & b33["term"].eq("treated_ekmn")
    ]
    if len(gate33) != 1:
        raise ValueError("33번 재현 게이트 행이 없다")
    row33 = gate33.iloc[0]
    ok33 = (
        int(row33["nobs"]) == EXPECTED_B33_N
        and abs(float(row33["beta"]) - EXPECTED_B33_BETA) <= GATE_TOL
    )

    b35 = pd.read_csv(B35_PATH)
    gate35 = b35[
        b35["spec"].eq("P-all_noncoalition: positive PPML")
        & b35["term"].eq("sanc_stag")
    ]
    if len(gate35) != 1:
        raise ValueError(
            "35번 all_noncoalition positive PPML이 없다. 35번 --stage positive를 먼저 실행한다."
        )
    row35 = gate35.iloc[0]
    ok35 = (
        int(row35["nobs"]) == EXPECTED_B35_N
        and abs(float(row35["beta"]) - EXPECTED_B35_BETA) <= GATE_TOL
    )
    store.log(
        f"33 gate: N={int(row33['nobs']):,}, beta={float(row33['beta']):+.6f}; "
        f"{'OK' if ok33 else 'FAIL'}"
    )
    store.log(
        f"35 baseline: N={int(row35['nobs']):,}, beta={float(row35['beta']):+.6f}; "
        f"{'OK' if ok35 else 'FAIL'}"
    )
    if not (ok33 and ok35):
        raise SystemExit("상류 재현 게이트가 어긋나 37번을 중단한다")


def baci_path(year: int) -> Path:
    return BACI_DIR / f"BACI_HS92_Y{year}_V202601.csv"


def cache_paths(scope: str) -> tuple[Path, Path]:
    return (
        PROCESSED_DIR / f"b37_joint_panel_{scope}.parquet",
        PROCESSED_DIR / f"b37_joint_panel_{scope}.meta.json",
    )


def select_pilot_destinations(sanctioners: set[int], gateways: set[int], russia: int) -> set[int]:
    path = baci_path(2019)
    raw = pd.read_csv(path, usecols=["i", "j", "v"])
    exports = raw[raw["i"].isin(sanctioners) & raw["j"].ne(russia)]
    candidates = exports[~exports["j"].isin(sanctioners | gateways)]
    controls = set(candidates.groupby("j")["v"].sum().nlargest(10).index.astype(int))
    return gateways | controls


def cache_signature(scope: str, destinations: set[int] | None) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "scope": scope,
        "years": list(YEARS),
        "destinations": sorted(destinations) if destinations is not None else "all",
        "design": "EKMN all destinations plus world exports to Russia; observed positive flows",
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


def build_joint_panel(store: OutputStore, scope: str) -> pd.DataFrame:
    ensure_inputs([baci_path(year) for year in YEARS] + [SANCTIONS_MASTER])
    sanctioners = ekmn_country_codes()
    gateways = gateway_country_codes()
    russia = russia_code()
    destinations = (
        select_pilot_destinations(sanctioners, gateways, russia)
        if scope == "pilot"
        else None
    )
    sanctions = pd.read_csv(SANCTIONS_MASTER, usecols=["hs6"])
    hs6_universe = sanctions["hs6"].astype(str).str.zfill(6).unique()
    lookup = build_ekmn_lookup(hs6_universe)
    treatment = product_year_treatment()

    frames = []
    years = range(2019, 2025) if scope == "pilot" else YEARS
    pilot_hs6 = set(sorted(hs6_universe)[:500]) if scope == "pilot" else None
    for year in years:
        started = time.time()
        raw = pd.read_csv(
            baci_path(year), usecols=["t", "i", "j", "k", "v"], dtype={"k": str}
        )
        raw["hs6"] = raw["k"].astype(str).str.zfill(6)
        if pilot_hs6 is not None:
            raw = raw[raw["hs6"].isin(pilot_hs6)]

        to_russia = raw["j"].eq(russia) & raw["i"].ne(russia)
        from_sanctioners = raw["i"].isin(sanctioners) & raw["j"].ne(raw["i"])
        if destinations is not None:
            from_sanctioners &= raw["j"].isin(destinations | {russia})
        keep = to_russia | from_sanctioners
        selected = raw.loc[keep, ["t", "i", "j", "hs6", "v"]].copy()
        selected = (
            selected.groupby(["t", "i", "j", "hs6"], as_index=False)["v"].sum()
        )
        selected = attach_ekmn_treatment(selected, lookup=lookup, exporter_col="i")
        selected = selected.merge(
            treatment, on=["hs6", "t"], how="left", validate="many_to_one"
        )
        # BACI에는 있지만 SANCTIONS_MASTER에 없는 HS6 (624개) → 비제재 → sanc_stag=0
        selected["sanc_stag"] = selected["sanc_stag"].fillna(0).astype("int8")
        selected["is_ekmn_exporter"] = selected["i"].isin(sanctioners).astype("int8")
        selected["direct_effect"] = (
            selected["treated_ekmn"] * selected["j"].eq(russia)
        ).astype("int8")
        selected["gateway_route"] = (
            selected["treated_ekmn"] * selected["j"].isin(gateways)
        ).astype("int8")
        selected["supplier_replacement"] = (
            selected["sanc_stag"]
            * selected["j"].eq(russia)
            * selected["is_ekmn_exporter"].rsub(1)
        ).astype("int8")
        frames.append(selected)
        store.log(f"{year}: selected rows={len(selected):,} [{time.time() - started:.0f}s]")
        store.checkpoint()

    panel = pd.concat(frames, ignore_index=True)
    assert_no_duplicate(panel, ["t", "i", "j", "hs6"], "joint positive panel")
    panel.rename(columns={"i": "exporter", "j": "importer"}, inplace=True)
    panel["ln_v"] = np.log1p(panel["v"])
    panel["exporter_importer"] = (
        panel["exporter"].astype(str)
        + "_" + panel["importer"].astype(str)
    )
    panel["hs6_year"] = panel["hs6"] + "_" + panel["t"].astype(str)
    panel["exporter_year"] = panel["exporter"].astype(str) + "_" + panel["t"].astype(str)
    panel["importer_year"] = panel["importer"].astype(str) + "_" + panel["t"].astype(str)

    cache, meta = cache_paths(scope)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache, index=False)
    meta.write_text(
        json.dumps(cache_signature(scope, destinations), indent=2), encoding="utf-8"
    )
    store.log(f"joint cache 저장: {cache}; rows={len(panel):,}")
    return panel


def load_joint_panel(store: OutputStore, scope: str, rebuild: bool) -> pd.DataFrame:
    cache, meta = cache_paths(scope)
    sanctioners = ekmn_country_codes()
    gateways = gateway_country_codes()
    russia = russia_code()
    destinations = (
        select_pilot_destinations(sanctioners, gateways, russia)
        if scope == "pilot"
        else None
    )
    expected = cache_signature(scope, destinations)
    if rebuild or not (cache.exists() and meta.exists()):
        return build_joint_panel(store, scope)
    try:
        observed = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return build_joint_panel(store, scope)
    if observed != expected:
        return build_joint_panel(store, scope)
    panel = pd.read_parquet(cache)
    required = {"v", "ln_v", *TERMS, *[part.strip() for part in FE.split("+")]}
    missing = required - set(panel.columns)
    if missing:
        store.log(f"joint cache 열 누락 {sorted(missing)}; 재생성한다")
        return build_joint_panel(store, scope)
    store.log(f"joint cache 불러옴: {cache}; rows={len(panel):,}")
    return panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("gate", "pilot", "build", "full"), default="gate")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = OutputStore(CSV_PATH, LOG_PATH)
    started = time.time()
    try:
        version = check_pyfixest_version()
        store.log("=" * 76)
        store.log("37: joint bilateral PPML")
        store.log(f"stage={args.stage}; pyfixest={version}")
        store.log("=" * 76)
        verify_upstream_results(store)
        if args.stage == "gate":
            return

        scope = "pilot" if args.stage == "pilot" else "full"
        panel = load_joint_panel(store, scope, args.rebuild)
        save_audit([sample_audit(panel, f"b37_{scope}", exporter_col="exporter")], AUDIT_PATH)
        if args.stage == "build":
            return

        fepois_kwargs = supported_fepois_kwargs()
        extra = {
            "stage": args.stage,
            "sample": scope,
            "positive_only": True,
            "coefficients_must_not_be_summed": True,
        }
        run_and_store(
            store,
            panel,
            label=f"J1: joint bilateral PPML ({scope})",
            estimator="PPML",
            outcome="v",
            terms=TERMS,
            fe=FE,
            vcov=VCOV,
            fepois_kwargs=fepois_kwargs,
            extra=extra,
        )
        run_and_store(
            store,
            panel,
            label=f"J2: joint bilateral OLS ({scope})",
            estimator="OLS",
            outcome="ln_v",
            terms=TERMS,
            fe=FE,
            vcov=VCOV,
            extra=extra,
        )
        store.log("세 경로 계수는 합산하지 않는다.")
        store.log(f"Finished selected stage in {(time.time() - started) / 60:.1f} min")
    finally:
        store.checkpoint()


if __name__ == "__main__":
    main()
