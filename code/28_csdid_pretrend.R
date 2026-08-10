#!/usr/bin/env Rscript
# 28_csdid_pretrend.R
# Diagnose the isolated long-lead coefficient in the CS-DID event study.
#
# Supplies the timing diagnostic reported in Online Appendix M. In the default
# event-time aggregation one pre-treatment cell, event time -8, carries a large
# positive ATT whose uniform confidence band excludes zero, larger in absolute
# value than the onset effect.
#
# That cell comes from a single group-time combination: the 2023 cohort observed
# in the first sample year. For the 2022 cohort event time -8 would be 2014 and
# for the 2016 cohort 2008, both outside the 2015-2024 window. The 2023 cohort's
# default reference year is 2022, by which point its trade had already adjusted.
#
# The script re-runs the estimator treating the year before adoption as an
# anticipation period and reports how that cell and the aggregate ATT move, and
# which cohorts remain in each run. Under that setting the software drops the
# 2016 cohort, because it is already treated or anticipating treatment in the
# first sample year, so the two runs are not a like-for-like comparison and both
# are reported.

suppressPackageStartupMessages({
  library(did)
  library(data.table)
})

main <- function() {

PROC_DIR    <- "phase4_analysis/processed"
RESULTS_DIR <- "phase4_analysis/results"
PANEL       <- file.path(PROC_DIR, "csdid_panel_coalition.csv")
ATTGT_REF   <- file.path(RESULTS_DIR, "b25_csdid_attgt.csv")

OUTS <- list(decomp = file.path(RESULTS_DIR, "b28_decomp.csv"),
             hs6    = file.path(RESULTS_DIR, "b28_hs6.csv"),
             clust  = file.path(RESULTS_DIR, "b28_cluster.csv"),
             sens   = file.path(RESULTS_DIR, "b28_sensitivity.csv"))
LOG_PATH <- file.path(RESULTS_DIR, "b28_pretrend_log.txt")
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

# 재현 게이트 기대값 (25_csdid.R --full --spec=coalition)
G_CELL    <- 2023L
T_CELL    <- 2015L
BASE_YEAR <- G_CELL - 1L        # universal 기준연도 = 2022
EXP_CELL  <- 0.5403
TOL_CELL  <- 5e-4               # 점추정은 bstrap과 무관하므로 좁게 잡는다
TOL_VEC   <- 1e-6               # b25 CSV와의 (g,t,ATT) 벡터 대조
BOOT_R    <- 1000L
SEED      <- 42L

.log <- character(0)
prt <- function(...) {
  msg <- paste0(...); cat(msg, "\n", sep = ""); flush.console()
  .log <<- c(.log, msg)
}
# 로그는 실패해도 남긴다. 결과 CSV는 전부 성공했을 때만 확정한다.
on.exit(writeLines(.log, LOG_PATH), add = TRUE)

failures <- character(0)
fail <- function(what) {
  prt(sprintf("  [FAIL] %s", what))
  failures <<- c(failures, what)
}

# 원자적 쓰기 — .partial에 쓰고 마지막에 이름을 바꾼다.
# A failed run must not leave the previous output in place to be read as new.
partial_keys <- character(0)
write_partial <- function(dt, key) {
  fwrite(dt, paste0(OUTS[[key]], ".partial"))
  partial_keys <<- c(partial_keys, key)
  invisible(NULL)
}
finalize <- function() {
  for (k in partial_keys) {
    p <- paste0(OUTS[[k]], ".partial")
    if (file.exists(p)) file.rename(p, OUTS[[k]])
  }
}
discard_partials <- function() {
  for (k in partial_keys) {
    p <- paste0(OUTS[[k]], ".partial")
    if (file.exists(p)) file.remove(p)
  }
}

prt(strrep("=", 74))
prt("A-6: CS-DID 사전추세 t-8 원인 규명")
prt(strrep("=", 74))
prt(sprintf("R %s.%s / did %s / seed %d / bootstrap %d회",
            R.version$major, R.version$minor,
            as.character(packageVersion("did")), SEED, BOOT_R))
prt("결과변수는 ln(v+1)이다. 아래 값은 전부 로그포인트다 — 퍼센트로 읽지 말 것.")

if (!file.exists(PANEL)) {
  prt(sprintf("입력 없음: %s — 24_export_csdid_panel.py를 먼저 돌릴 것", PANEL))
  return(1L)
}

d <- fread(PANEL)
need <- c("id", "t", "y", "g", "coalition_tier1", "hs6")
miss <- setdiff(need, names(d))
if (length(miss)) {
  prt(sprintf("입력에 열이 없다: %s", paste(miss, collapse = ", ")))
  prt("24_export_csdid_panel.py를 hs6·partner 보존 버전으로 다시 돌릴 것.")
  return(1L)
}
d <- d[coalition_tier1 == 1]
setorder(d, id, t)
prt(sprintf("표본 %s행, 단위 %s개, HS6 %s개, 연도 %d-%d",
            format(nrow(d), big.mark = ","), format(uniqueN(d$id), big.mark = ","),
            format(uniqueN(d$hs6), big.mark = ","), min(d$t), max(d$t)))
gt <- d[, .(units = uniqueN(id), n_hs6 = uniqueN(hs6)), by = g][order(g)]
for (i in seq_len(nrow(gt)))
  prt(sprintf("    %-16s 단위 %7s개, HS6 %5s개",
              if (gt$g[i] == 0) "never-treated" else paste0(gt$g[i], " 코호트"),
              format(gt$units[i], big.mark = ","),
              format(gt$n_hs6[i], big.mark = ",")))

# ---------------------------------------------------------------- D0 게이트
prt(""); prt(strrep("=", 74)); prt("D0 — 재현 게이트"); prt(strrep("=", 74))

t0 <- Sys.time()
m0 <- tryCatch(
  att_gt(yname = "y", tname = "t", idname = "id", gname = "g", xformla = ~1,
         data = d, panel = TRUE, allow_unbalanced_panel = FALSE,
         control_group = "nevertreated", est_method = "reg",
         base_period = "universal", bstrap = FALSE, print_details = FALSE),
  error = function(e) { fail(sprintf("att_gt: %s", conditionMessage(e))); NULL })
if (is.null(m0)) { discard_partials(); return(1L) }
prt(sprintf("  att_gt 완료 [%.1f분]",
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))

cells <- data.table(group = m0$group, time = m0$t, att = m0$att, se = m0$se)

# (1) 셀 유일성 — e=-8이 정말 (2023,2015) 하나인가
c8 <- cells[time - group == -8]
se_unit <- NA_real_
if (nrow(c8) != 1L) {
  fail(sprintf("e=-8 셀이 %d개다 (1개여야 한다)", nrow(c8)))
} else if (c8$group != G_CELL || c8$time != T_CELL) {
  fail(sprintf("e=-8 셀이 (g=%d,t=%d)다 (기대 (%d,%d))",
               c8$group, c8$time, G_CELL, T_CELL))
} else {
  se_unit <- c8$se
  gap <- abs(c8$att - EXP_CELL)
  prt(sprintf("  e=-8 셀 (g=%d, t=%d) 유일 [OK]", c8$group, c8$time))
  prt(sprintf("  ATT %+.4f (기대 %+.4f, 차 %.5f) %s",
              c8$att, EXP_CELL, gap, if (gap <= TOL_CELL) "[OK]" else "[FAIL]"))
  if (gap > TOL_CELL) fail("e=-8 셀 점추정이 기대값과 다르다")
}

# (2) Compare the full attgt vector against the stored b25 output
if (!file.exists(ATTGT_REF)) {
  prt(sprintf("  참조 없음: %s — 벡터 대조 생략", ATTGT_REF))
} else {
  ref <- fread(ATTGT_REF)
  if (!all(c("group", "time", "att") %in% names(ref))) {
    fail("b25 attgt CSV에 group/time/att 열이 없다")
  } else {
    cmp <- merge(cells[, .(group, time, att_now = att)],
                 ref[, .(group, time, att_ref = att)],
                 by = c("group", "time"), all = TRUE)
    n_miss <- cmp[is.na(att_now) | is.na(att_ref), .N]
    dmax <- suppressWarnings(max(abs(cmp$att_now - cmp$att_ref), na.rm = TRUE))
    ok <- (n_miss == 0L) && is.finite(dmax) && (dmax <= TOL_VEC)
    prt(sprintf("  (g,t) 셀 %d개 대조: 미매칭 %d개, 최대 차 %.2e %s",
                nrow(cmp), n_miss, dmax, if (ok) "[OK]" else "[FAIL]"))
    if (!ok) fail("b25 attgt 벡터와 일치하지 않는다")
  }
}

if (length(failures)) {
  prt(""); prt("게이트 실패 — 표본이나 설정이 25번과 다르다. 진단을 중단한다.")
  discard_partials(); return(1L)
}
prt("  Gate PASSED.")

# ---------------------------------------------------------------- A 분해 항등식
prt(""); prt(strrep("=", 74))
prt("A — 분해 항등식 (H2: 2022 기준시점 오염인가)")
prt(strrep("=", 74))
prt("  D_t = 2023 코호트 평균 y - never-treated 평균 y  (같은 균형 단위 표본)")
prt("  U_t = D_t - D_2022   (universal, 25번이 쓰는 기준)")
prt("  V_t = D_t - D_{t-1}  (varying, 직전 기간 대비)")
prt("  항등식: U_2015 = -(V_2016 + ... + V_2022)")

sub <- d[g %in% c(0L, G_CELL)]
Dt <- sub[, .(mean_y = mean(y)), by = .(grp = ifelse(g == 0L, "ctrl", "trt"), t)]
Dw <- dcast(Dt, t ~ grp, value.var = "mean_y")
setorder(Dw, t)
Dw[, D := trt - ctrl]
base_D <- Dw[t == BASE_YEAR, D]

if (!length(base_D)) {
  fail(sprintf("기준연도 %d가 표본에 없다", BASE_YEAR))
} else {
  Dw[, U := D - base_D]
  Dw[, V := D - shift(D)]
  for (i in seq_len(nrow(Dw)))
    prt(sprintf("    %d  D %+.4f   U %+.4f   V %s", Dw$t[i], Dw$D[i], Dw$U[i],
                if (is.na(Dw$V[i])) "     -" else sprintf("%+.4f", Dw$V[i])))

  u2015 <- Dw[t == T_CELL, U]
  vsum  <- Dw[t > T_CELL & t <= BASE_YEAR, sum(V)]
  prt("")
  prt(sprintf("  검산: U_%d = %+.4f, -(V_%d..V_%d 합) = %+.4f, 차 %.2e %s",
              T_CELL, u2015, T_CELL + 1L, BASE_YEAR, -vsum, abs(u2015 + vsum),
              if (abs(u2015 + vsum) < 1e-8) "[OK]" else "[FAIL]"))
  if (abs(u2015 + vsum) >= 1e-8) fail("분해 항등식이 성립하지 않는다")

  gapcell <- abs(u2015 - EXP_CELL)
  prt(sprintf("  U_%d와 att_gt 셀 대조: %+.4f vs %+.4f, 차 %.5f %s",
              T_CELL, u2015, EXP_CELL, gapcell,
              if (gapcell <= TOL_CELL) "[OK]" else "[FAIL]"))
  if (gapcell > TOL_CELL)
    fail("단순 평균 분해가 att_gt 셀과 다르다 (est_method·가중 확인 필요)")

  vv <- Dw[t > T_CELL & t <= BASE_YEAR, .(t, V, share = 100 * V / vsum)]
  prt("")
  prt("  각 연도 변화가 전체 격차에서 차지하는 비중:")
  for (i in seq_len(nrow(vv)))
    prt(sprintf("    V_%d %+.4f  (%+.1f%%)%s", vv$t[i], vv$V[i], vv$share[i],
                if (vv$t[i] == BASE_YEAR) "   <- 2022 충격" else ""))
  s22 <- vv[t == BASE_YEAR, share]
  prt("")
  prt(sprintf("  판정: V_2022 기여 %.1f%%", s22))
  prt(if (s22 >= 50)
        "    -> 격차의 절반 이상이 2022 한 해에서 나온다. H2(기준시점 오염)를 지지한다."
      else if (s22 >= 25)
        "    -> 2022가 크지만 단독은 아니다. 부분적 H2 + 일반 사전추세."
      else "    -> 여러 해에 걸쳐 벌어졌다. 일반적인 평행추세 위반에 가깝다.")
  write_partial(Dw, "decomp")
}

# ---------------------------------------------------------------- B HS6 집중도
prt(""); prt(strrep("=", 74))
prt("B — HS6 집중도 (H1: 소수 제품이 만드는가)")
prt(strrep("=", 74))

ctrl_chg <- sub[g == 0L & t %in% c(T_CELL, BASE_YEAR), .(m = mean(y)), by = t]
ctrl_delta <- ctrl_chg[t == T_CELL, m] - ctrl_chg[t == BASE_YEAR, m]

unit_delta <- dcast(sub[g == G_CELL & t %in% c(T_CELL, BASE_YEAR)],
                    id + hs6 ~ t, value.var = "y")
setnames(unit_delta, c(as.character(T_CELL), as.character(BASE_YEAR)),
         c("y_t", "y_base"))
unit_delta[, delta := y_t - y_base]
n_unit <- nrow(unit_delta)

hs6_tab <- unit_delta[, .(units = .N, mean_delta = mean(delta),
                          contrib = sum(delta) / n_unit), by = hs6]
setorder(hs6_tab, -contrib)
hs6_tab[, share := 100 * contrib / sum(contrib)]
prt(sprintf("  2023 코호트: 단위 %s개, HS6 %d개. 통제군 변화 %+.4f",
            format(n_unit, big.mark = ","), nrow(hs6_tab), ctrl_delta))
prt(sprintf("  처치군 변화 합 %+.4f  -> U_%d %+.4f",
            sum(hs6_tab$contrib), T_CELL, sum(hs6_tab$contrib) - ctrl_delta))
prt("")
prt("  기여 상위 10개 HS6:")
for (i in seq_len(min(10L, nrow(hs6_tab))))
  prt(sprintf("    %2d. HS %-8s 단위 %4d  평균차 %+.3f  기여 %+.4f (%.1f%%)",
              i, as.character(hs6_tab$hs6[i]), hs6_tab$units[i],
              hs6_tab$mean_delta[i], hs6_tab$contrib[i], hs6_tab$share[i]))
for (k in c(1L, 5L, 10L))
  if (nrow(hs6_tab) >= k)
    prt(sprintf("  상위 %2d개 누적 기여 비중: %.1f%%", k,
                sum(hs6_tab$share[seq_len(k)])))

n_loo <- min(20L, nrow(hs6_tab))
loo <- rbindlist(lapply(seq_len(n_loo), function(i) {
  keep <- unit_delta[hs6 != hs6_tab$hs6[i]]
  data.table(hs6 = hs6_tab$hs6[i], units_dropped = hs6_tab$units[i],
             u2015_loo = mean(keep$delta) - ctrl_delta)
}))
loo[, change := u2015_loo - EXP_CELL]
setorder(loo, u2015_loo)
prt("")
prt(sprintf("  leave-one-HS6-out (기여 상위 %d개 중 U_2015를 가장 낮추는 5개):", n_loo))
for (i in seq_len(min(5L, nrow(loo))))
  prt(sprintf("    HS %-8s 제외 -> U_2015 %+.4f (원 %+.4f, 변화 %+.4f)",
              as.character(loo$hs6[i]), loo$u2015_loo[i], EXP_CELL, loo$change[i]))
prt("")
top1 <- hs6_tab$share[1]
top5 <- sum(hs6_tab$share[seq_len(min(5L, nrow(hs6_tab)))])
prt(sprintf("  판정: 상위 1개 HS6 기여 %.1f%%, 상위 5개 %.1f%%", top1, top5))
prt(if (top1 >= 40) "    -> 한 제품이 지배한다. H1(소수 제품 집중)을 지지한다."
    else if (top5 >= 60) "    -> 소수 제품군에 몰려 있다. H1을 부분적으로 지지한다."
    else "    -> 제품 전반에 퍼져 있다. H1을 지지하지 않는다.")
write_partial(hs6_tab, "hs6")

# ---------------------------------------------------------------- C 클러스터 추론
prt(""); prt(strrep("=", 74))
prt("C — HS6 클러스터 부트스트랩 (H4: 표준오차가 좁은가)")
prt(strrep("=", 74))
prt("  처치는 HS6 수준인데 단위는 partner x HS6다. 같은 HS6의 여러 partner가")
prt("  공통 제품 충격을 공유하면 단위를 독립으로 세는 SE는 과소평가된다.")

ctrl_unit <- dcast(sub[g == 0L & t %in% c(T_CELL, BASE_YEAR)],
                   id + hs6 ~ t, value.var = "y")
setnames(ctrl_unit, c(as.character(T_CELL), as.character(BASE_YEAR)),
         c("y_t", "y_base"))
ctrl_unit[, delta := y_t - y_base]

trt_h <- split(unit_delta$delta, unit_delta$hs6)
ctl_h <- split(ctrl_unit$delta,  ctrl_unit$hs6)
prt(sprintf("  처치 HS6 클러스터 %d개, 통제 HS6 클러스터 %d개",
            length(trt_h), length(ctl_h)))

set.seed(SEED)
boot <- vapply(seq_len(BOOT_R), function(b) {
  ti <- sample.int(length(trt_h), length(trt_h), replace = TRUE)
  ci <- sample.int(length(ctl_h), length(ctl_h), replace = TRUE)
  mean(unlist(trt_h[ti], use.names = FALSE)) -
    mean(unlist(ctl_h[ci], use.names = FALSE))
}, numeric(1))

se_clu <- sd(boot)
ci_clu <- unname(quantile(boot, c(0.025, 0.975)))
prt("")
prt(sprintf("  단위 수준 SE (25번 보고값)   %.4f", se_unit))
prt(sprintf("  HS6 클러스터 부트스트랩 SE   %.4f  (%.1f배)",
            se_clu, se_clu / se_unit))
prt(sprintf("  점추정 %+.4f, 클러스터 95%% 구간 [%+.4f, %+.4f]",
            EXP_CELL, ci_clu[1], ci_clu[2]))
sig_clu <- (ci_clu[1] * ci_clu[2]) > 0
prt(sprintf("  0 배제 여부: %s",
            if (sig_clu) "배제 — 여전히 유의" else "포함 — 유의성 사라짐"))
prt("")
prt(if (!sig_clu)
      "  판정: H4를 지지한다. HS6로 묶으면 t-8 사전추세는 유의하지 않다."
    else if (se_clu / se_unit >= 2)
      "  판정: SE가 크게 넓어지지만 부호는 남는다. 균일 신뢰대를 다시 계산해야 한다."
    else "  판정: 클러스터를 바꿔도 추론이 크게 달라지지 않는다. H4를 지지하지 않는다.")
write_partial(data.table(
  stat  = c("se_unit", "se_hs6_cluster", "ratio", "att", "ci_lo", "ci_hi",
            "boot_R", "n_trt_clusters", "n_ctl_clusters"),
  value = c(se_unit, se_clu, se_clu / se_unit, EXP_CELL, ci_clu[1], ci_clu[2],
            BOOT_R, length(trt_h), length(ctl_h))), "clust")

# ---------------------------------------------------------------- D 민감도
prt(""); prt(strrep("=", 74))
prt("D - Sensitivity: a separate estimand, not a test of H2")
prt(strrep("=", 74))

sens <- list()
run_sens <- function(label, ...) {
  prt(""); prt(sprintf("--- %s ---", label))
  m <- tryCatch(att_gt(yname = "y", tname = "t", idname = "id", gname = "g",
                       xformla = ~1, data = d, panel = TRUE,
                       allow_unbalanced_panel = FALSE,
                       control_group = "nevertreated", est_method = "reg",
                       base_period = "universal", bstrap = FALSE,
                       print_details = FALSE, ...),
                error = function(e) {
                  fail(sprintf("%s: %s", label, conditionMessage(e))); NULL })
  if (is.null(m)) return(invisible(NULL))
  cc <- data.table(group = m$group, time = m$t, att = m$att, se = m$se)
  included_groups <- sort(unique(cc$group))
  prt(sprintf("    포함 코호트: %s", paste(included_groups, collapse = ", ")))
  cell <- cc[group == G_CELL & time == T_CELL]
  a <- tryCatch(aggte(m, type = "simple", na.rm = TRUE, bstrap = FALSE),
                error = function(e) {
                  fail(sprintf("%s simple: %s", label, conditionMessage(e))); NULL })
  if (nrow(cell))
    prt(sprintf("    ATT(g=%d,t=%d) %+.4f (%.4f)", G_CELL, T_CELL,
                cell$att, cell$se))
  else
    prt(sprintf("    (g=%d,t=%d) 셀이 없다 — estimand가 다르다", G_CELL, T_CELL))
  if (!is.null(a))
    prt(sprintf("    전체 ATT %+.4f (%.4f)", a$overall.att, a$overall.se))
  sens[[length(sens) + 1L]] <<- data.table(
    label       = label,
    cell_att    = if (nrow(cell)) cell$att else NA_real_,
    cell_se     = if (nrow(cell)) cell$se  else NA_real_,
    overall_att = if (!is.null(a)) a$overall.att else NA_real_,
    overall_se  = if (!is.null(a)) a$overall.se  else NA_real_,
    groups      = paste(included_groups, collapse = ","))
  invisible(NULL)
}

run_sens("D1: clustervars=hs6 (단위+제품 이중 클러스터)", clustervars = "hs6")
run_sens("D2: anticipation=1 (기준연도 2021 — 2022를 처치영향 기간으로 재분류)",
         anticipation = 1)

if (length(sens)) {
  prt("")
  prt("  주의: D1은 추론만 바꾼다(점추정 불변). D2는 estimand 자체가 다르다.")
  write_partial(rbindlist(sens, fill = TRUE), "sens")
}

# ---------------------------------------------------------------- 요약
prt(""); prt(strrep("=", 74)); prt("SUMMARY"); prt(strrep("=", 74))
if (length(failures)) {
  prt(sprintf("  실패 %d건:", length(failures)))
  for (f in failures) prt(sprintf("    - %s", f))
  prt("  결과 파일을 확정하지 않는다 (.partial 삭제).")
  discard_partials()
  return(1L)
}
prt("  모든 검사 성공. 결과 파일을 확정한다.")
for (k in partial_keys) prt(sprintf("    %s", OUTS[[k]]))
finalize()
prt("")
prt("  서술 규칙: 로그포인트로만 쓴다. 이 2x2 결과를 본문 삼중차분의 확인으로")
prt("  부르지 않는다. 단일 코호트 e=-8을 onset·overall ATT와 크기 비교하지 않는다.")
return(0L)
}

status <- main()
quit(status = if (is.null(status)) 1L else as.integer(status))
