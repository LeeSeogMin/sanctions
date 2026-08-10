"""Concordance between the two independently compiled EU sanctions lists.

Compares the treatment list used here (Chupilkin et al. 2025, OpenICPSR 229004)
against the list of Egorov, Korovkin, Makarin and Nigmatulina (OpenICPSR
230342). Descriptive only; nothing is re-estimated. Supplies the agreement
rates reported in Section 3.1.

Method
  The EKMN file mixes 2- to 10-digit codes. Entries of six digits or more are
  treated as HS6; shorter entries are treated as prefixes and expanded across
  the estimation universe. The comparison universe is the set of HS6 codes that
  actually appear in the panel. Implementation dates are compared at the year
  level, matching the annual panel.

Limitation, recorded in the output: the EKMN source uses EU Combined
Nomenclature codes of mixed vintage while BACI uses HS92, so digit-string
matching across revisions is an approximation.

Outputs
  results/b18_ekmn_concordance_summary.csv
  results/b18_ekmn_concordance_codes.csv
"""

import pandas as pd
import pyarrow.parquet as pq

# ---------------------------------------------------------------- 1. 데이터 로드
chu = pd.read_stata("phase3_data/raw/EU_sanctions_HS6.dta")
chu["Code"] = chu["Code"].astype(str).str.zfill(6)
chu_s = chu[chu["EU_sanction"] == 1].copy()
chu_date = chu_s.groupby("Code")["Date"].min()

CATS = ["dual_use", "industrial_cap", "luxury", "military_tech",
        "aviation", "firearms", "oil_exploration", "oil_refining"]

ekmn = pd.read_stata("phase3_data/raw/EKMN_230342/data_sa_20241230.dta")
ekmn_eu = ekmn[ekmn["country2"] == "EU"].copy()
# prefix: len>=6 → code_6 (HS6), len<6 → 원 코드 그대로 (챕터/헤딩 prefix)
ekmn_eu["prefix"] = ekmn_eu.apply(
    lambda r: str(r["code_6"]) if r["len"] >= 6 else str(r["code"]), axis=1
)
ekmn_pref = ekmn_eu.groupby("prefix")["date_s"].min()  # prefix별 최초 시행일

universe = sorted(
    pq.read_table("phase4_analysis/processed/baci_russia_panel.parquet",
                  columns=["hs6"])["hs6"].unique().to_pylist()
)
universe = [str(x).zfill(6) for x in universe]
print(f"BACI 유니버스: {len(universe):,} HS6")
print(f"Chupilkin 제재 코드: {chu_date.shape[0]:,} (유니버스 매칭 전)")
print(f"EKMN EU prefix 수: {len(ekmn_pref):,} "
      f"(6자리 {sum(len(p)==6 for p in ekmn_pref.index):,} / "
      f"4자리 {sum(len(p)==4 for p in ekmn_pref.index):,} / "
      f"2자리 {sum(len(p)==2 for p in ekmn_pref.index):,})")

# ------------------------------------------------- 2. 유니버스 내 prefix 확장 매칭
# 빠른 매칭: HS6의 2·4·6자리 절단을 prefix 사전과 대조
pref_by_len = {L: {p: d for p, d in ekmn_pref.items() if len(p) == L} for L in (2, 4, 6)}

rows = []
for h in universe:
    e_dates = [pref_by_len[L][h[:L]] for L in (2, 4, 6) if h[:L] in pref_by_len[L]]
    e_date = min(e_dates) if e_dates else pd.NaT
    c_date = chu_date.get(h, pd.NaT)
    rows.append({"hs6": h,
                 "chupilkin": int(pd.notna(c_date)),
                 "ekmn": int(pd.notna(e_date)),
                 "chupilkin_date": c_date, "ekmn_date": e_date})
df = pd.DataFrame(rows)
df["both"] = df["chupilkin"] & df["ekmn"]

n_c, n_e = df["chupilkin"].sum(), df["ekmn"].sum()
n_both = df["both"].sum()
n_union = ((df["chupilkin"] | df["ekmn"]) == 1).sum()
agree_all = (df["chupilkin"] == df["ekmn"]).mean()  # 비제재 합치 포함 전체 일치율

print(f"\n=== 유니버스 내 제재 지정 대조 ===")
print(f"Chupilkin 제재: {n_c:,} | EKMN EU 제재: {n_e:,} | 교집합: {n_both:,}")
print(f"Chupilkin 중 EKMN 합치: {n_both/n_c*100:.1f}% | EKMN 중 Chupilkin 합치: {n_both/n_e*100:.1f}%")
print(f"Jaccard: {n_both/n_union:.3f} | 전체 코드 지정 일치율(비제재 포함): {agree_all*100:.1f}%")

# ------------------------------------------------------------- 3. 시행일 일치
b = df[df["both"] == 1].copy()
b["same_year"] = b["chupilkin_date"].dt.year == b["ekmn_date"].dt.year
b["diff_days"] = (b["ekmn_date"] - b["chupilkin_date"]).dt.days
print(f"\n=== 공통 {len(b):,}개 코드 시행일 ===")
print(f"동일 일자: {(b['diff_days']==0).mean()*100:.1f}% | 동일 연도(연간 패널 기준): "
      f"{b['same_year'].mean()*100:.1f}% | |차이|<=90일: {(b['diff_days'].abs()<=90).mean()*100:.1f}%")
print(f"일자 차이 중앙값: {b['diff_days'].median():.0f}일")

# -------------------------------------------------- 4. 불일치 코드 프로파일
cat_map = chu_s.set_index("Code")[CATS]
only_c = df[(df["chupilkin"] == 1) & (df["ekmn"] == 0)]["hs6"]
only_e = df[(df["chupilkin"] == 0) & (df["ekmn"] == 1)]["hs6"]
print(f"\n=== 불일치 프로파일 ===")
print(f"Chupilkin-only: {len(only_c):,}")
oc = cat_map.reindex(only_c).sum()
print("  카테고리:", {k: int(v) for k, v in oc.items() if v > 0})
# Chupilkin-only 중 컷오프 외(EKMN이 늦게 수집했을 가능성) 점검용 — 시행연도 분포
oc_years = chu_date.reindex(only_c).dt.year.value_counts().sort_index()
print("  시행연도 분포:", oc_years.to_dict())
print(f"EKMN-only: {len(only_e):,}")
print("  HS2 상위:", pd.Series([x[:2] for x in only_e]).value_counts().head(6).to_dict())
oe_years = df.set_index("hs6").loc[only_e, "ekmn_date"].dt.year.value_counts().sort_index()
print("  EKMN 시행연도 분포:", oe_years.to_dict())

# 컷오프 정렬 비교 (Chupilkin 수집 종료 = 그 최후 시행일)
cutoff = chu_date.max()
df["ekmn_cut"] = ((df["ekmn"] == 1) & (df["ekmn_date"] <= cutoff)).astype(int)
n_e_cut = df["ekmn_cut"].sum()
n_both_cut = ((df["chupilkin"] == 1) & (df["ekmn_cut"] == 1)).sum()
print(f"\n=== 컷오프 정렬 (EKMN 최초 시행일 <= {cutoff.date()}) ===")
print(f"EKMN(cut): {n_e_cut:,} | 교집합 {n_both_cut:,} | "
      f"EKMN(cut) 중 Chupilkin 합치: {n_both_cut/n_e_cut*100:.1f}%")

# ------------------------------------------------------------------ 5. 저장
df.to_csv("phase4_analysis/results/b18_ekmn_concordance_codes.csv", index=False)
summary = pd.DataFrame([{
    "universe_hs6": len(universe), "chupilkin_sanctioned": n_c, "ekmn_eu_sanctioned": n_e,
    "intersection": n_both, "share_chupilkin_in_ekmn": n_both / n_c,
    "share_ekmn_in_chupilkin": n_both / n_e, "jaccard": n_both / n_union,
    "designation_agreement_all_codes": agree_all,
    "same_day_share": (b["diff_days"] == 0).mean(),
    "same_year_share": b["same_year"].mean(),
    "within_90d_share": (b["diff_days"].abs() <= 90).mean(),
    "ekmn_cutoff_aligned": n_e_cut, "intersection_cutoff": n_both_cut,
    "chupilkin_cutoff_date": cutoff.date(),
    "note": ("EKMN national CN codes (mixed HS2017/2022 vintages) prefix-matched to "
             "BACI HS92 universe; cross-revision HS6 string matching is approximate."),
}])
summary.to_csv("phase4_analysis/results/b18_ekmn_concordance_summary.csv", index=False)
print("\n저장: results/b18_ekmn_concordance_{summary,codes}.csv")
