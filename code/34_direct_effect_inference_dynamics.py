#!/usr/bin/env python3
"""Inference and dynamics for the jurisdiction-specific direct effect.

Requires the stored reproduction gate from script 33. It separates

  static PPML with HS6 clustering and with HS6-plus-exporter multiway clustering
  a PPML event study, reference e = -1, with e <= -4, -3, -2, 0, 1, e >= 2
  a static OLS robustness check on the same sample, fixed effects and clustering
  a jurisdiction-level wild cluster bootstrap on the OLS fit, where available

The PPML event study and the linear difference-in-differences ATT are not the
same estimand and are not compared as a single percentage effect. The default
stage is pilot.
"""

from __future__ import annotations

import argparse
import importlib.util
import time

import numpy as np
import pandas as pd

from revision_methodology_utils import (
    OutputStore,
    PANEL_PATH,
    RESULTS_DIR,
    attach_ekmn_treatment,
    build_ekmn_lookup,
    check_pyfixest_version,
    ensure_inputs,
    fit_model,
    run_and_store,
    sample_audit,
    save_audit,
    select_hs6_pilot,
    supported_fepois_kwargs,
)


B33_PATH = RESULTS_DIR / "b33_direct_effect_ekmn.csv"
CSV_PATH = RESULTS_DIR / "b34_direct_effect_inference_dynamics.csv"
LOG_PATH = RESULTS_DIR / "b34_direct_effect_inference_dynamics_log.txt"
AUDIT_PATH = RESULTS_DIR / "b34_direct_effect_sample_audit.csv"
WILD_PATH = RESULTS_DIR / "b34_direct_effect_wild_bootstrap.csv"

FE = "partner_hs6 + hs6_year + partner_year"
VCOV_HS6 = {"CRV1": "hs6"}
VCOV_TWOWAY = {"CRV1": "hs6 + partner"}
VCOV_POLICY_TWOWAY = {"CRV1": "hs6 + ekmn_juris"}

EXPECTED_GATE_N = 842_264
EXPECTED_GATE_BETA = -0.525173
GATE_TOL = 0.005


def verify_b33_gate(store: OutputStore) -> None:
    ensure_inputs((B33_PATH,))
    results = pd.read_csv(B33_PATH)
    gate = results[
        results["spec"].eq("GATE: existing EKMN Top-50 PPML")
        & results["term"].eq("treated_ekmn")
    ]
    if len(gate) != 1:
        raise ValueError("33번 결과에서 유일한 재현 게이트 행을 찾지 못했다")
    row = gate.iloc[0]
    n_ok = int(row["nobs"]) == EXPECTED_GATE_N
    beta_ok = abs(float(row["beta"]) - EXPECTED_GATE_BETA) <= GATE_TOL
    store.log(
        f"33번 저장 게이트: N={int(row['nobs']):,}, beta={float(row['beta']):+.6f}; "
        f"{'PASSED' if n_ok and beta_ok else 'FAILED'}"
    )
    if not (n_ok and beta_ok):
        raise SystemExit("33번 재현 게이트가 유효하지 않아 34번을 중단한다")


def load_direct_sample(store: OutputStore) -> pd.DataFrame:
    columns = [
        "t", "v", "direction", "partner", "hs6", "partner_hs6",
        "hs6_year", "partner_year",
    ]
    panel = pd.read_parquet(PANEL_PATH, columns=columns)
    direct = panel[panel["direction"].eq("to_russia")].copy()
    del panel
    direct["hs6"] = direct["hs6"].astype(str).str.zfill(6)
    lookup = build_ekmn_lookup(direct["hs6"].unique())
    direct = attach_ekmn_treatment(direct, lookup=lookup)
    direct = direct[direct["ekmn_juris"].notna()].copy()
    years_per_pair = direct.groupby("partner_hs6")["t"].nunique()
    direct = direct[
        direct["partner_hs6"].isin(years_per_pair[years_per_pair.gt(1)].index)
    ].copy()
    direct["ekmn_juris"] = direct["ekmn_juris"].astype(str)
    direct["ln_v"] = np.log1p(direct["v"])
    store.log(
        f"EKMN 직접효과 양(+) 표본: {len(direct):,}; "
        f"HS6 {direct['hs6'].nunique():,}; 수출국 {direct['partner'].nunique()}"
    )
    return direct


def add_event_indicators(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = data.copy()
    event_time = out["t"] - out["ekmn_first_year"]
    definitions = {
        "event_m4": event_time.le(-4),
        "event_m3": event_time.eq(-3),
        "event_m2": event_time.eq(-2),
        "event_0": event_time.eq(0),
        "event_p1": event_time.eq(1),
        "event_p2": event_time.ge(2),
    }
    for term, mask in definitions.items():
        out[term] = (out["ekmn_first_year"].notna() & mask).astype("int8")
    terms = list(definitions)
    empty = [term for term in terms if out[term].sum() == 0]
    if empty:
        raise ValueError(f"사건시간 셀에 처치 관측이 없다: {empty}")
    return out, terms


def run_wild_bootstrap(model, store: OutputStore, stage: str) -> None:
    if importlib.util.find_spec("wildboottest") is None:
        store.log(
            "wildboottest 패키지가 없어 관할권 수준 wild cluster bootstrap을 건너뛴다. "
            "의존성을 추가하기 전에는 이 결과를 완료로 표시하지 않는다."
        )
        pd.DataFrame(
            [{
                "stage": stage,
                "term": "treated_ekmn",
                "status": "not_run",
                "reason": "wildboottest package not installed",
            }]
        ).to_csv(WILD_PATH, index=False)
        return
    try:
        result = model.wildboottest(
            reps=9_999,
            cluster="ekmn_juris",
            param="treated_ekmn",
            seed=20260808,
            weights_type="webb",
        )
        result = result.copy()
        result.insert(0, "stage", stage)
        result.to_csv(WILD_PATH, index=False)
        store.log(f"wild cluster bootstrap 저장: {WILD_PATH}")
    except Exception as exc:
        pd.DataFrame(
            [{
                "stage": stage,
                "term": "treated_ekmn",
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }]
        ).to_csv(WILD_PATH, index=False)
        store.log(f"wild cluster bootstrap 실패: {type(exc).__name__}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "full"), default="pilot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = OutputStore(CSV_PATH, LOG_PATH)
    started = time.time()
    try:
        version = check_pyfixest_version()
        store.log("=" * 76)
        store.log("34: EKMN direct-effect inference and dynamics")
        store.log(f"stage={args.stage}; pyfixest={version}")
        store.log("=" * 76)
        verify_b33_gate(store)

        direct = load_direct_sample(store)
        if args.stage == "pilot":
            direct = select_hs6_pilot(direct)
        save_audit([sample_audit(direct, f"b34_{args.stage}")], AUDIT_PATH)
        fepois_kwargs = supported_fepois_kwargs()

        run_and_store(
            store,
            direct,
            label=f"I1: static PPML HS6 cluster ({args.stage})",
            estimator="PPML",
            outcome="v",
            terms=["treated_ekmn"],
            fe=FE,
            vcov=VCOV_HS6,
            fepois_kwargs=fepois_kwargs,
            extra={"stage": args.stage, "inference": "HS6_cluster"},
        )
        run_and_store(
            store,
            direct,
            label=f"I2: static PPML two-way cluster ({args.stage})",
            estimator="PPML",
            outcome="v",
            terms=["treated_ekmn"],
            fe=FE,
            vcov=VCOV_TWOWAY,
            fepois_kwargs=fepois_kwargs,
            extra={"stage": args.stage, "inference": "HS6_exporter_two_way"},
            fail_hard=False,
        )
        run_and_store(
            store,
            direct,
            label=f"I2b: static PPML policy-level two-way cluster ({args.stage})",
            estimator="PPML",
            outcome="v",
            terms=["treated_ekmn"],
            fe=FE,
            vcov=VCOV_POLICY_TWOWAY,
            fepois_kwargs=fepois_kwargs,
            extra={
                "stage": args.stage,
                "inference": "HS6_EKMN_jurisdiction_two_way",
                "few_policy_clusters": True,
            },
            fail_hard=False,
        )

        dynamic, event_terms = add_event_indicators(direct)
        run_and_store(
            store,
            dynamic,
            label=f"I3: PPML event study ({args.stage})",
            estimator="PPML",
            outcome="v",
            terms=event_terms,
            fe=FE,
            vcov=VCOV_HS6,
            fepois_kwargs=fepois_kwargs,
            extra={"stage": args.stage, "reference_event": -1},
        )

        run_and_store(
            store,
            direct,
            label=f"I4: static OLS two-way cluster ({args.stage})",
            estimator="OLS",
            outcome="ln_v",
            terms=["treated_ekmn"],
            fe=FE,
            vcov=VCOV_TWOWAY,
            extra={"stage": args.stage, "inference": "HS6_exporter_two_way"},
        )
        wild_model = fit_model(
            direct,
            estimator="OLS",
            outcome="ln_v",
            terms=["treated_ekmn"],
            fe=FE,
            vcov={"CRV1": "ekmn_juris"},
        )
        run_wild_bootstrap(wild_model, store, args.stage)

        store.log(
            "\nCS-DID와 IW-OLS는 기존 24·25·28번 결과를 별도 강건성으로 읽는다. "
            "이 스크립트의 PPML 계수와 같은 ATT로 합치지 않는다."
        )
        store.log(f"Finished selected stage in {(time.time() - started) / 60:.1f} min")
    finally:
        store.checkpoint()


if __name__ == "__main__":
    main()
