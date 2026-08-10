#!/usr/bin/env python3
"""Document why the saturated joint PPML does not complete.

Supplies the feasibility statement in Section 3.6 and Online Appendix N. The
most saturated joint specification, with exporter x importer x HS6 fixed
effects, did not return a completed estimate; only the less saturated version
with exporter x importer fixed effects finished, on 61,132,859 observations.
This script produces the logged basis for that statement.

What it does
  reads the cached full panel and counts the dimension of the saturated fixed
  effects and the number of singleton observations, without estimating,
  iterating the removal to convergence because dropping singletons creates new
  ones, and reporting the resulting estimation sample.

What it does not do
  attempt to make the saturated specification converge. Obtaining a coefficient
  is not the purpose. With --attempt it makes one wall-clock-bounded attempt and
  records how that attempt ended.

Counting singletons is a groupby size computation rather than an estimation, so
it finishes in minutes and is deterministic.

Outputs
  results/b39_saturated_feasibility.csv
  results/b39_saturated_feasibility_log.txt

Usage
  python 39_saturated_joint_feasibility.py
  python 39_saturated_joint_feasibility.py --attempt --max-minutes 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from revision_methodology_utils import (
    OutputStore,
    PROCESSED_DIR,
    RESULTS_DIR,
    check_pyfixest_version,
    exit_with_log,
)

CACHE = PROCESSED_DIR / "b37_joint_panel_full.parquet"
META = PROCESSED_DIR / "b37_joint_panel_full.meta.json"
CSV_PATH = RESULTS_DIR / "b39_saturated_feasibility.csv"
LOG_PATH = RESULTS_DIR / "b39_saturated_feasibility_log.txt"

TERMS = ["direct_effect", "gateway_route", "supplier_replacement"]

# --- 재현 게이트 (작업 원칙 3: 이미 아는 값을 다시 계산해 어긋나면 새 숫자를 쓰지 않는다)
EXPECTED_CACHE_ROWS = 61_133_364      # b37 로그: "joint cache 불러옴 ... rows=61,133,364"
EXPECTED_LESS_SAT_N = 61_132_859      # b37 로그: J1 세 계수의 N
ROW_TOL = 0                           # 캐시 행 수는 정확히 일치해야 한다

# 포화 사양 (37번 CACHE_VERSION 2에서 빠진 것 — 여기서 되살린다)
SATURATED_FE = ["exporter_importer_hs6", "hs6_year", "exporter_year", "importer_year"]


def gate(store: OutputStore, panel: pd.DataFrame) -> None:
    """캐시가 37번이 쓴 그 표본인지 확인한다. 어긋나면 아무것도 세지 않는다."""
    store.log("=" * 74)
    store.log("재현 게이트 — 이 표본이 37번 J1이 쓴 것과 같은가")
    store.log("=" * 74)
    ok = True

    rows = len(panel)
    hit = abs(rows - EXPECTED_CACHE_ROWS) <= ROW_TOL
    ok &= hit
    store.log(f"  [{'OK ' if hit else 'FAIL'}] 캐시 행 수: {rows:,} (기대 {EXPECTED_CACHE_ROWS:,})")

    needed = {"exporter", "importer", "hs6", "t", "v", *TERMS}
    missing = needed - set(panel.columns)
    hit = not missing
    ok &= hit
    store.log(f"  [{'OK ' if hit else 'FAIL'}] 필요한 열: {'전부 있음' if hit else f'누락 {sorted(missing)}'}")

    if META.exists():
        sig = json.loads(META.read_text(encoding="utf-8"))
        store.log(f"  캐시 서명: scope={sig.get('scope')}, cache_version={sig.get('cache_version')}")
    else:
        store.log("  [WARN] 캐시 메타데이터가 없다 — 서명을 확인하지 못했다")

    if not ok:
        exit_with_log(store, "게이트 실패: 캐시가 37번 표본이 아니다. 세지 않고 중단한다.")
    store.log("  게이트 통과.\n")


def build_codes(store: OutputStore, panel: pd.DataFrame) -> dict[str, np.ndarray]:
    """네 고정효과를 int32 코드로 만든다.

    문자열 키를 61M행에 들고 있으면 메모리가 터진다. 코드로 바꾸면 4열 × 4바이트 =
    약 1GB로 끝난다. 포화 차원(수출국×수입국×HS6)은 37번 CACHE_VERSION 2에서
    빠졌으므로 세 열에서 직접 만든다.
    """
    store.log("고정효과를 int32 코드로 변환한다 (문자열 키는 61M행에서 메모리로 터진다)")
    codes: dict[str, np.ndarray] = {}
    started = time.time()
    codes["exporter_importer_hs6"] = (
        panel.groupby(["exporter", "importer", "hs6"], sort=False).ngroup().to_numpy(dtype="int64")
    )
    store.log(f"  exporter_importer_hs6 코드 생성 [{time.time() - started:.0f}s]")
    for name in ("hs6_year", "exporter_year", "importer_year"):
        started = time.time()
        codes[name] = pd.factorize(panel[name], sort=False)[0].astype("int64")
        store.log(f"  {name} 코드 생성 [{time.time() - started:.0f}s]")
    store.log("")
    return codes


def count_fe_levels(store: OutputStore, codes: dict[str, np.ndarray], nobs: int) -> dict[str, int]:
    """포화 사양의 고정효과 차원. '왜 안 끝났는가'의 1차 답이다."""
    store.log("=" * 74)
    store.log("A. 포화 고정효과의 차원")
    store.log("=" * 74)
    levels: dict[str, int] = {}
    for name in SATURATED_FE:
        n = int(codes[name].max()) + 1
        levels[name] = n
        store.log(f"  {name:24s} {n:>12,} 수준")
    total = sum(levels.values())
    store.log(f"  {'합계':24s} {total:>12,} 모수")
    store.log(f"  관측 대비: {nobs / total:.2f} 관측/모수\n")
    return levels


def iterative_singletons(
    store: OutputStore, panel: pd.DataFrame, codes: dict[str, np.ndarray]
) -> dict[str, int]:
    """singleton 반복 제거. 한 번 빼면 다른 차원에 새 singleton이 생기므로 수렴시킨다.

    HDFE 추정기가 추정 전에 수행하는 절차와 같다. 여기서는 세기만 한다.
    np.bincount로 세므로 라운드마다 한 번씩 훑는 것으로 끝난다.
    """
    store.log("=" * 74)
    store.log("B. singleton 반복 제거 (추정 없이 세기만 한다)")
    store.log("=" * 74)
    nobs = len(panel)
    alive = np.ones(nobs, dtype=bool)
    rounds = 0

    for rnd in range(1, 101):
        started = time.time()
        drop = np.zeros(nobs, dtype=bool)
        for name in SATURATED_FE:
            col = codes[name]
            sizes = np.bincount(col[alive], minlength=int(col.max()) + 1)
            drop[alive] |= sizes[col[alive]] == 1
        n_drop = int(drop.sum())
        if n_drop == 0:
            store.log(f"  라운드 {rnd}: 새 singleton 0 — 수렴  [{time.time() - started:.0f}s]")
            break
        alive &= ~drop
        rounds = rnd
        store.log(
            f"  라운드 {rnd}: singleton {n_drop:,} 제거 → 잔여 {int(alive.sum()):,}"
            f"  [{time.time() - started:.0f}s]"
        )
    else:
        store.log("  [WARN] 100라운드에서 수렴하지 않았다 — 아래 수치는 하한이다")

    total_dropped = nobs - int(alive.sum())
    store.log(f"\n  총 제거: {total_dropped:,} ({total_dropped / nobs:.2%})")
    store.log(f"  포화 사양의 추정 표본: {int(alive.sum()):,}")
    store.log(f"  (참고) 덜 포화된 사양의 실제 N: {EXPECTED_LESS_SAT_N:,}\n")

    treated_lost: dict[str, int] = {}
    store.log("  제거된 행에 포함된 처치 관측:")
    for term in TERMS:
        values = panel[term].to_numpy()
        lost = int(values[~alive].sum())
        total = int(values.sum())
        treated_lost[term] = lost
        share = lost / total if total else 0.0
        store.log(f"    {term:24s} {lost:>10,} / {total:,} ({share:.1%})")
    store.log("")

    return {
        "singletons_removed": total_dropped,
        "estimation_rows": int(alive.sum()),
        "rounds": rounds,
        **{f"treated_lost_{k}": v for k, v in treated_lost.items()},
    }


def attempt_fit(store: OutputStore, panel: pd.DataFrame, max_minutes: float) -> dict:
    """Optional: make one attempt at the saturated specification and record how it
    ended. The coefficient is not reported."""
    store.log("=" * 74)
    store.log(f"C. 포화 사양 1회 시도 (벽시계 상한 {max_minutes:.0f}분)")
    store.log("=" * 74)
    store.log("  목적은 계수가 아니라 종료 사유의 기록이다.")
    import pyfixest as pf  # noqa: PLC0415

    # 포화 차원은 캐시에 없다. 시도할 때만 만든다 (문자열 61M행 — 메모리를 많이 쓴다).
    panel["exporter_importer_hs6"] = (
        panel["exporter"].astype(str) + "_" + panel["importer"].astype(str) + "_" + panel["hs6"]
    )
    formula = f"v ~ {' + '.join(TERMS)} | {' + '.join(SATURATED_FE)}"
    store.log(f"  PPML: {formula}")
    started = time.time()
    try:
        model = pf.fepois(formula, data=panel, vcov={"CRV1": "exporter_importer_hs6"})
        elapsed = (time.time() - started) / 60
        store.log(f"  completed [{elapsed:.1f} min] - the feasibility statement would need revision")
        return {"outcome": "converged", "minutes": round(elapsed, 1), "summary": str(model.coef())}
    except Exception as exc:  # noqa: BLE001 — 종료 사유 자체가 기록 대상이다
        elapsed = (time.time() - started) / 60
        store.log(f"  종료: {type(exc).__name__}: {exc}")
        store.log(f"  경과 {elapsed:.1f}분")
        return {"outcome": f"{type(exc).__name__}", "minutes": round(elapsed, 1), "summary": str(exc)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attempt", action="store_true",
        help="세기만 하지 않고 포화 사양을 한 번 시도한다 (오래 걸린다)",
    )
    parser.add_argument("--max-minutes", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = OutputStore(csv_path=CSV_PATH, log_path=LOG_PATH)
    try:
        store.log("=" * 74)
        store.log("39: 포화 공동 PPML의 미완료를 문서화한다 (R2 주요의견 4)")
        store.log("=" * 74)
        store.log(f"pyfixest {check_pyfixest_version()}")

        if not CACHE.exists():
            exit_with_log(
                store,
                f"전체 패널 캐시가 없다: {CACHE}\n"
                "먼저 `python phase4_analysis/code/37_full_bilateral_joint_ppml.py --stage build`를 돌린다.",
            )

        started = time.time()
        panel = pd.read_parquet(CACHE)
        store.log(f"캐시 로드: {len(panel):,}행 [{time.time() - started:.0f}s]\n")

        gate(store, panel)
        codes = build_codes(store, panel)
        levels = count_fe_levels(store, codes, len(panel))
        counts = iterative_singletons(store, panel, codes)

        row = {"spec": "saturated joint PPML feasibility", **levels, **counts}
        if args.attempt:
            row.update(attempt_fit(store, panel, args.max_minutes))
        else:
            row["outcome"] = "not attempted (counting only)"
        store.add_rows([row])

        store.log("=" * 74)
        store.log("Feasibility summary (this log is the basis)")
        store.log("=" * 74)
        store.log(
            f"  포화 사양은 고정효과 모수가 {sum(levels.values()):,}개이고, singleton 반복 제거로"
        )
        store.log(
            f"  {counts['singletons_removed']:,}행({counts['singletons_removed'] / len(panel):.1%})이 빠진다."
        )
        store.log("  ⚠️ 서술 규칙: 이 수치는 '왜 완료되지 않았는가'의 근거이지, 포화 사양의")
        store.log("     추정 결과가 아니다. 계수를 보고하지 않는다. 덜 포화된 사양(Appendix N)만")
        store.log("     보고하고, 두 사양은 추정대상이 달라 직접 비교하지 않는다.")
    finally:
        store.checkpoint()


if __name__ == "__main__":
    main()
