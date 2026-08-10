#!/usr/bin/env python3
"""Shared data and estimation helpers for scripts 33-38.

Not a standalone script. It builds country codes and jurisdiction-specific
treatment in one place and standardises the pyfixest version check, the
required columns, and the result-file format.
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyfixest as pf


REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = REPO_ROOT / "phase4_analysis/processed/baci_russia_panel.parquet"
SANCTIONS_MASTER = REPO_ROOT / "phase4_analysis/processed/sanctions_hs6_master.csv"
EKMN_PATH = REPO_ROOT / "phase3_data/raw/EKMN_230342/data_sa_20241230.dta"
BACI_DIR = REPO_ROOT / "phase3_data/BACI_HS92_V202601"
PROCESSED_DIR = REPO_ROOT / "phase4_analysis/processed"
RESULTS_DIR = REPO_ROOT / "phase4_analysis/results"

EXPECTED_PYFIXEST = "0.40.1"
RUSSIA_ISO3 = "RUS"
YEARS = tuple(range(2015, 2025))
PRE_YEARS = tuple(range(2015, 2022))

EU27_ISO3 = (
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN",
    "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX",
    "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE",
)
EKMN_JURIS_ISO3 = {
    "EU": EU27_ISO3,
    "GB": ("GBR",),
    "US": ("USA",),
    "CA": ("CAN",),
    "JP": ("JPN",),
    "KR": ("KOR",),
    "CH": ("CHE",),
    "AU": ("AUS",),
}
TAIWAN_BACI_CODE = 490

GATEWAY_ISO3 = (
    "ARE", "ARM", "KGZ", "TUR", "CHN", "UZB", "KAZ", "TJK", "GEO", "IND",
)

FEPOIS_KWARGS = {
    "iwls_tol": 1e-8,
    "iwls_maxiter": 100,
    "fixef_tol": 1e-8,
    "fixef_maxiter": 100_000,
    "separation_check": ["fe"],
}


def ensure_inputs(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("필수 입력이 없다: " + ", ".join(missing))


def country_code_path() -> Path:
    candidates = (
        BACI_DIR / "country_codes_V202601.csv",
        REPO_ROOT / "phase3_data/raw/country_codes.csv",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("BACI 국가코드 파일을 찾지 못했다")


def load_country_codes() -> pd.DataFrame:
    path = country_code_path()
    codes = pd.read_csv(path)
    rename = {}
    if "country_code" in codes:
        rename["country_code"] = "code"
    elif "comtrade_code" in codes:
        rename["comtrade_code"] = "code"
    if "country_iso3" in codes:
        rename["country_iso3"] = "iso3"
    elif "iso3" in codes:
        rename["iso3"] = "iso3"
    codes = codes.rename(columns=rename)
    required = {"code", "iso3", "country_name"}
    missing = required - set(codes.columns)
    if missing:
        raise KeyError(f"{path}에 국가코드 열이 없다: {sorted(missing)}")
    codes = codes[["code", "iso3", "country_name"]].copy()
    codes["code"] = pd.to_numeric(codes["code"], errors="raise").astype(int)
    codes["iso3"] = codes["iso3"].astype(str).str.upper()
    if codes["code"].duplicated().any():
        raise ValueError(f"{path}의 국가코드가 중복된다")
    return codes


def iso3_to_code(codes: pd.DataFrame | None = None) -> dict[str, int]:
    if codes is None:
        codes = load_country_codes()
    # BACI 표에는 Belgium-Luxembourg(...1998), Fed. Rep. Germany(...1990)처럼
    # 같은 ISO3를 쓰는 역사 코드가 함께 있다. 현재 국가명을 우선하고 역사 코드는
    # 동일 ISO3의 대체 후보로만 남긴다.
    work = codes.copy()
    work["historical"] = work["country_name"].str.contains(r"\(\.\.\.", regex=True)
    work = work.sort_values(["iso3", "historical", "code"], kind="mergesort")
    current = work.drop_duplicates("iso3", keep="first")
    return current.set_index("iso3")["code"].astype(int).to_dict()


def require_iso3(mapping: dict[str, int], names: Iterable[str]) -> set[int]:
    names = tuple(names)
    missing = sorted(set(names) - set(mapping))
    if missing:
        raise KeyError(f"국가코드 표에 ISO3가 없다: {missing}")
    return {int(mapping[name]) for name in names}


def ekmn_exporter_map(codes: pd.DataFrame | None = None) -> dict[int, str]:
    mapping = iso3_to_code(codes)
    out: dict[int, str] = {}
    for jurisdiction, iso3s in EKMN_JURIS_ISO3.items():
        for code in require_iso3(mapping, iso3s):
            if code in out:
                raise ValueError(f"BACI 코드 {code}가 EKMN 관할권에 중복 매핑된다")
            out[code] = jurisdiction
    if TAIWAN_BACI_CODE in out:
        raise ValueError("대만 BACI 코드가 다른 EKMN 관할권과 겹친다")
    out[TAIWAN_BACI_CODE] = "TW"
    return out


def ekmn_country_codes(codes: pd.DataFrame | None = None) -> set[int]:
    return set(ekmn_exporter_map(codes))


def gateway_country_codes(codes: pd.DataFrame | None = None) -> set[int]:
    return require_iso3(iso3_to_code(codes), GATEWAY_ISO3)


def russia_code(codes: pd.DataFrame | None = None) -> int:
    mapping = iso3_to_code(codes)
    return next(iter(require_iso3(mapping, (RUSSIA_ISO3,))))


def _normalise_hs_code(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def build_ekmn_lookup(panel_hs6: Sequence[str]) -> dict[str, dict[str, int]]:
    """관할권별 HS6 최초 제재연도를 만든다.

    EKMN의 2·4자리 코드는 BACI HS92 유니버스에서 prefix로 확장한다. 같은 HS6에
    여러 행이 걸리면 가장 이른 시행연도를 쓴다.
    """
    ensure_inputs((EKMN_PATH,))
    ekmn = pd.read_stata(EKMN_PATH)
    required = {"country2", "len", "code", "code_6", "date_s"}
    missing = required - set(ekmn.columns)
    if missing:
        raise KeyError(f"EKMN 파일에 열이 없다: {sorted(missing)}")
    ekmn = ekmn.dropna(subset=["country2", "len", "date_s"]).copy()
    ekmn["prefix"] = [
        _normalise_hs_code(code6 if length >= 6 else code)
        for length, code, code6 in zip(ekmn["len"], ekmn["code"], ekmn["code_6"])
    ]
    ekmn = ekmn[ekmn["prefix"].str.len().isin((2, 4, 6))].copy()
    ekmn["year"] = pd.to_datetime(ekmn["date_s"]).dt.year.astype(int)

    universe = sorted({str(code).zfill(6) for code in panel_hs6})
    lookup: dict[str, dict[str, int]] = {}
    for jurisdiction, group in ekmn.groupby("country2"):
        first = group.groupby("prefix")["year"].min().to_dict()
        by_length = {
            length: {prefix: year for prefix, year in first.items() if len(prefix) == length}
            for length in (2, 4, 6)
        }
        expanded: dict[str, int] = {}
        for hs6 in universe:
            years = [
                int(by_length[length][hs6[:length]])
                for length in (2, 4, 6)
                if hs6[:length] in by_length[length]
            ]
            if years:
                expanded[hs6] = min(years)
        lookup[str(jurisdiction)] = expanded
    expected = set(EKMN_JURIS_ISO3) | {"TW"}
    missing_jurisdictions = sorted(expected - set(lookup))
    if missing_jurisdictions:
        raise ValueError(f"EKMN 자료에 관할권이 없다: {missing_jurisdictions}")
    return lookup


def attach_ekmn_treatment(
    data: pd.DataFrame,
    lookup: dict[str, dict[str, int]] | None = None,
    exporter_col: str = "partner",
) -> pd.DataFrame:
    required = {exporter_col, "hs6", "t"}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"EKMN 처치를 만들 열이 없다: {sorted(missing)}")
    out = data.copy()
    out["hs6"] = out["hs6"].astype(str).str.zfill(6)
    exporter_map = ekmn_exporter_map()
    out["ekmn_juris"] = out[exporter_col].map(exporter_map)
    if lookup is None:
        lookup = build_ekmn_lookup(out["hs6"].unique())

    pairs = out[["ekmn_juris", "hs6"]].drop_duplicates().copy()
    pairs["ekmn_first_year"] = [
        np.nan if pd.isna(juris) else lookup.get(str(juris), {}).get(hs6, np.nan)
        for juris, hs6 in zip(pairs["ekmn_juris"], pairs["hs6"])
    ]
    out = out.merge(pairs, on=["ekmn_juris", "hs6"], how="left", validate="many_to_one")
    out["treated_ekmn"] = (
        out["ekmn_first_year"].notna() & out["t"].ge(out["ekmn_first_year"])
    ).astype("int8")
    return out


def check_pyfixest_version() -> str:
    version = str(getattr(pf, "__version__", "unknown"))
    if version != EXPECTED_PYFIXEST:
        raise RuntimeError(
            f"pyfixest {version} != {EXPECTED_PYFIXEST}. 재현 게이트 전에 버전을 고정해야 한다."
        )
    return version


def supported_fepois_kwargs() -> dict[str, Any]:
    try:
        params = inspect.signature(pf.fepois).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError("pyfixest.fepois 시그니처를 읽지 못했다") from exc
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return dict(FEPOIS_KWARGS)
    return {key: value for key, value in FEPOIS_KWARGS.items() if key in params}


def get_nobs(model: Any) -> int:
    for attribute in ("nobs", "_N"):
        try:
            return int(getattr(model, attribute))
        except (AttributeError, TypeError):
            continue
    try:
        return int(len(model._Y))
    except AttributeError:
        return -1


def percent_effect(beta: float) -> float:
    return float(np.expm1(beta) * 100)


def star(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def fixed_effect_columns(fe: str) -> list[str]:
    return [part.strip() for part in fe.split("+") if part.strip()]


def cluster_columns(vcov: str | dict[str, str]) -> list[str]:
    if isinstance(vcov, str):
        return []
    formula = next(iter(vcov.values()))
    return [part.strip() for part in formula.split("+") if part.strip()]


def fit_model(
    data: pd.DataFrame,
    estimator: str,
    outcome: str,
    terms: Sequence[str],
    fe: str,
    vcov: str | dict[str, str],
    fepois_kwargs: dict[str, Any] | None = None,
) -> Any:
    required = [outcome, *terms, *fixed_effect_columns(fe), *cluster_columns(vcov)]
    required = list(dict.fromkeys(required))
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise KeyError(f"모형에 필요한 열이 없다: {missing}")
    if data[required].isna().any(axis=None):
        counts = data[required].isna().sum()
        bad = counts[counts.gt(0)].to_dict()
        raise ValueError(f"모형 입력에 결측이 있다: {bad}")
    formula = f"{outcome} ~ {' + '.join(terms)} | {fe}"
    model_data = data[required]
    if estimator.upper() == "PPML":
        if (model_data[outcome] < 0).any():
            raise ValueError("PPML 종속변수에 음수가 있다")
        return pf.fepois(
            formula,
            data=model_data,
            vcov=vcov,
            **(fepois_kwargs or supported_fepois_kwargs()),
        )
    if estimator.upper() == "OLS":
        return pf.feols(formula, data=model_data, vcov=vcov)
    raise ValueError(f"지원하지 않는 추정량: {estimator}")


def model_result_rows(
    model: Any,
    label: str,
    estimator: str,
    outcome: str,
    terms: Sequence[str],
    fe: str,
    vcov: str | dict[str, str],
    seconds: float,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    nobs = get_nobs(model)
    for term in terms:
        if term not in model.coef().index:
            raise KeyError(f"{label}: 추정 결과에서 {term}이 빠졌다")
        beta = float(model.coef()[term])
        se = float(model.se()[term])
        p_value = float(model.pvalue()[term])
        row = {
            "spec": label,
            "estimator": estimator.upper(),
            "outcome": outcome,
            "term": term,
            "beta": beta,
            "se": se,
            "pval": p_value,
            "pct": percent_effect(beta),
            "nobs": nobs,
            "fe": fe,
            "vcov": json.dumps(vcov, ensure_ascii=False) if isinstance(vcov, dict) else vcov,
            "seconds": round(seconds, 1),
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


@dataclass
class OutputStore:
    csv_path: Path
    log_path: Path
    rows: list[dict[str, Any]] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: Any = "") -> None:
        text = str(message)
        print(text, flush=True)
        self.log_lines.append(text)

    def checkpoint(self) -> None:
        # gate-only 스크립트가 결과행을 만들지 않았을 때 기존의 비싼 추정 CSV를
        # 빈 파일로 덮어쓰지 않는다. 새 결과행이 있으면 해당 실행 결과로 교체한다.
        if self.rows:
            pd.DataFrame(self.rows).to_csv(self.csv_path, index=False)
        self.log_path.write_text("\n".join(self.log_lines) + "\n", encoding="utf-8")

    def add_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        self.rows.extend(rows)
        self.checkpoint()

    def add_error(
        self,
        label: str,
        estimator: str,
        error: BaseException,
        extra: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "spec": label,
            "estimator": estimator,
            "nobs": -1,
            "error": f"{type(error).__name__}: {error}",
        }
        if extra:
            row.update(extra)
        self.rows.append(row)
        self.log(f"{label}: ERROR {type(error).__name__}: {error}")
        self.checkpoint()


def run_and_store(
    store: OutputStore,
    data: pd.DataFrame,
    label: str,
    estimator: str,
    outcome: str,
    terms: Sequence[str],
    fe: str,
    vcov: str | dict[str, str],
    fepois_kwargs: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    fail_hard: bool = True,
) -> tuple[Any | None, list[dict[str, Any]]]:
    store.log(f"\n--- {label} ---")
    store.log(f"{estimator}: {outcome} ~ {' + '.join(terms)} | {fe}; vcov={vcov}")
    started = time.time()
    try:
        model = fit_model(data, estimator, outcome, terms, fe, vcov, fepois_kwargs)
        rows = model_result_rows(
            model, label, estimator, outcome, terms, fe, vcov, time.time() - started, extra
        )
        for row in rows:
            store.log(
                f"{row['term']}: beta={row['beta']:+.4f} ({row['se']:.4f})"
                f"{star(row['pval'])}; pct={row['pct']:+.1f}%; N={row['nobs']:,}"
            )
        store.add_rows(rows)
        return model, rows
    except Exception as exc:
        store.add_error(label, estimator, exc, extra)
        if fail_hard:
            raise
        return None, []


def sample_audit(
    data: pd.DataFrame,
    sample: str,
    exporter_col: str = "partner",
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "sample": sample,
        "rows": int(len(data)),
        "years": int(data["t"].nunique()) if "t" in data else np.nan,
        "year_min": int(data["t"].min()) if len(data) and "t" in data else np.nan,
        "year_max": int(data["t"].max()) if len(data) and "t" in data else np.nan,
        "hs6": int(data["hs6"].nunique()) if "hs6" in data else np.nan,
        "exporters": int(data[exporter_col].nunique()) if exporter_col in data else np.nan,
        "zero_rows": int(data["v"].eq(0).sum()) if "v" in data else np.nan,
        "zero_share": float(data["v"].eq(0).mean()) if len(data) and "v" in data else np.nan,
        "value_sum": float(data["v"].sum()) if "v" in data else np.nan,
    }
    for treatment in ("treated_ekmn", "sanc_stag", "sanc_post"):
        if treatment in data:
            audit[f"{treatment}_rows"] = int(data[treatment].sum())
    return audit


def save_audit(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def select_hs6_pilot(data: pd.DataFrame, count: int = 500) -> pd.DataFrame:
    """파일럿용 HS6 부분집합을 뽑는다.

    처치가 있는 HS6와 없는 HS6를 절반씩 계층 추출한다. 알파벳순 앞 N개는
    식품·농업 HS 챕터(01~20)가 대부분이어서 EKMN 제재 제품이 거의 없다.
    계층 추출을 쓰면 파일럿에서 처치 변이가 확보된다.
    """
    df = data.copy()
    df["hs6"] = df["hs6"].astype(str).str.zfill(6)
    treatment_col = "treated_ekmn" if "treated_ekmn" in df.columns else None
    if treatment_col is not None:
        treated_hs6 = set(df.loc[df[treatment_col].eq(1), "hs6"].unique())
    else:
        treated_hs6 = set()
    all_hs6 = sorted(df["hs6"].unique())
    treated_pool = sorted(h for h in all_hs6 if h in treated_hs6)
    control_pool = sorted(h for h in all_hs6 if h not in treated_hs6)
    n_treated = min(count // 2, len(treated_pool))
    n_control = min(count - n_treated, len(control_pool))
    keep = set(treated_pool[:n_treated]) | set(control_pool[:n_control])
    return df[df["hs6"].isin(keep) & df["t"].between(2019, 2024)].copy()


def require_zero_confirmation(confirmed: bool) -> None:
    if not confirmed:
        raise SystemExit(
            "0 포함 패널은 미관측을 실제 0으로 해석한다. "
            "자료 감사를 마쳤다면 --confirm-zeros-are-true-absences를 명시해야 한다."
        )


def exit_with_log(store: OutputStore, message: str, code: int = 1) -> None:
    store.log(message)
    store.checkpoint()
    raise SystemExit(code)


def assert_no_duplicate(data: pd.DataFrame, keys: Sequence[str], label: str) -> None:
    duplicated = data.duplicated(list(keys))
    if duplicated.any():
        examples = data.loc[duplicated, list(keys)].head().to_dict("records")
        raise ValueError(f"{label}에 중복키가 있다: {examples}")


def main() -> None:
    print("이 파일은 33~38번 스크립트가 import하는 모듈이다.")
    print("독립 추정은 실행하지 않는다.")


if __name__ == "__main__":
    sys.exit(main())
