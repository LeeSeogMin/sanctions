#!/usr/bin/env python3
"""Collect the results of scripts 33-37 into one audit table.

Reads stored CSVs and estimates nothing. By default every input must be
present. --allow-partial exists for checking table layout during development
and is not used for reported tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from revision_methodology_utils import REPO_ROOT, RESULTS_DIR


SOURCES = {
    "direct_effect": RESULTS_DIR / "b33_direct_effect_ekmn.csv",
    "inference_dynamics": RESULTS_DIR / "b34_direct_effect_inference_dynamics.csv",
    "supplier_replacement": RESULTS_DIR / "b35_supplier_replacement.csv",
    "route_link": RESULTS_DIR / "b36_route_linked_transshipment.csv",
    "joint_model": RESULTS_DIR / "b37_full_bilateral_joint.csv",
}
OUT_CSV = RESULTS_DIR / "b38_methodology_revision_summary.csv"
OUT_MD = REPO_ROOT / "EJPE/second-revision/methodology_revision_results.md"

CORE_COLUMNS = [
    "analysis", "spec", "stage", "sample", "estimator", "outcome", "term",
    "beta", "se", "pval", "pct", "nobs", "fe", "vcov", "error",
]


def load_results(allow_partial: bool) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    missing = []
    for analysis, path in SOURCES.items():
        if not path.exists():
            missing.append(str(path.relative_to(REPO_ROOT)))
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "analysis", analysis)
        frame["source_file"] = str(path.relative_to(REPO_ROOT))
        frames.append(frame)
    if missing and not allow_partial:
        raise FileNotFoundError(
            "33~37번 결과가 모두 있어야 한다. 누락: " + ", ".join(missing)
        )
    if not frames:
        raise FileNotFoundError("읽을 수 있는 33~37번 결과가 하나도 없다")
    return pd.concat(frames, ignore_index=True, sort=False), missing


def add_decision_flags(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    out["significant_5pct"] = out.get("pval", pd.Series(index=out.index, dtype=float)).lt(0.05)
    out["successful_fit"] = out.get("nobs", pd.Series(index=out.index, dtype=float)).gt(0)
    out["decision_flag"] = "review"

    direct = out["analysis"].eq("direct_effect") & out["term"].eq("treated_ekmn")
    out.loc[direct & ~out["significant_5pct"], "decision_flag"] = "no_general_direct_effect"

    supplier = out["analysis"].eq("supplier_replacement")
    china_excluded = supplier & out.get("sample", "").astype(str).str.contains("exclude_china")
    out.loc[china_excluded & ~out["significant_5pct"], "decision_flag"] = "china_dependent"

    route = out["analysis"].eq("route_link") & out["term"].astype(str).str.contains("_x_sanc")
    out.loc[route & ~out["significant_5pct"], "decision_flag"] = "no_general_transshipment"

    joint = out["analysis"].eq("joint_model")
    out.loc[joint, "decision_flag"] = "do_not_sum_route_coefficients"
    out.loc[~out["successful_fit"], "decision_flag"] = "fit_failed"
    return out


def fmt_number(value, digits: int = 3) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def markdown_table(rows: pd.DataFrame) -> str:
    headers = ["분석", "사양", "추정량", "항", "β", "SE", "%", "p", "N", "판정"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in rows.iterrows():
        nobs = "—" if pd.isna(row.get("nobs")) else f"{int(row['nobs']):,}"
        values = [
            str(row.get("analysis", "—")),
            str(row.get("spec", "—")).replace("|", "/"),
            str(row.get("estimator", "—")),
            str(row.get("term", "—")),
            fmt_number(row.get("beta")),
            fmt_number(row.get("se")),
            fmt_number(row.get("pct"), 1),
            fmt_number(row.get("pval")),
            nobs,
            str(row.get("decision_flag", "review")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown(results: pd.DataFrame, missing: list[str], allow_partial: bool) -> None:
    fitted = results[results["successful_fit"]].copy()
    fitted = fitted.sort_values(["analysis", "spec", "term"], kind="mergesort")
    status = "부분 결과 — 제출 금지" if allow_partial and missing else "33~37번 결과 취합"
    missing_text = ", ".join(missing) if missing else "없음"
    text = f"""# 2차 수정 방법론 결과 감사표

- 상태: {status}
- 누락 파일: {missing_text}
- 생성 스크립트: `phase4_analysis/code/38_methodology_revision_tables.py`

## 해석 규칙

- 직접효과, 공급대체, 물리적 환적 계수는 서로 다른 추정대상이다.
- PPML과 로그 OLS의 크기 차이를 한 추정량의 실패로 부르지 않는다.
- 결합모형의 세 경로 계수를 더해 누수율을 만들지 않는다.
- `route_link`는 두 다리의 연결성 진단이며 인과적 환적효과가 아니다.
- `positive_only=True`인 PPML은 0을 포함한 총효과가 아니다.

## 전체 추정 결과

{markdown_table(fitted)}

## 자동 판정의 한계

`decision_flag`는 사전 기준에 걸린 행을 찾는 감사 보조열이다. 표본 구성, 제외진단,
Do not settle a specification from this column alone, without reading the
pre-trend and convergence logs.
"""
    OUT_MD.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, missing = load_results(args.allow_partial)
    results = add_decision_flags(results)
    for column in CORE_COLUMNS:
        if column not in results:
            results[column] = np.nan
    ordered = CORE_COLUMNS + [column for column in results.columns if column not in CORE_COLUMNS]
    results[ordered].to_csv(OUT_CSV, index=False)
    write_markdown(results, missing, args.allow_partial)
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_MD}")
    if missing:
        print("Missing (partial mode): " + ", ".join(missing))


if __name__ == "__main__":
    main()
