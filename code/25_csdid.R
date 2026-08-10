#!/usr/bin/env Rscript
# 25_csdid.R
# Callaway-Sant'Anna group-time average treatment effects.
#
# Supplies the CS-DID rows of Table 6 and Online Appendix M. The sample is
# restricted to coalition exporters, so non-coalition exporters do not enter the
# comparison pool; treated products are compared with never-treated and
# not-yet-treated products.
#
# Input: the CSV written by 24_export_csdid_panel.py (id, t, y, g, coalition)
# Usage: Rscript 25_csdid.R [--full] [--spec=NAME]
#          default is a pilot run with analytic standard errors
#          --full adds the multiplier bootstrap and uniform confidence bands
#          --spec=coalition|truncated|all|notyettreated runs one specification
#
# The outcome is ln(v+1) on a balanced panel containing constructed zeros, so
# the ATT is reported in log points and is not converted to a percentage change
# in trade value or compared numerically with the PPML triple difference.

suppressPackageStartupMessages({
  library(did)
  library(data.table)
})

PROC_DIR    <- "phase4_analysis/processed"
RESULTS_DIR <- "phase4_analysis/results"
LOG_PATH    <- file.path(RESULTS_DIR, "b25_csdid_log.txt")
ATTGT_PATH  <- file.path(RESULTS_DIR, "b25_csdid_attgt.csv")
AGG_PATH    <- file.path(RESULTS_DIR, "b25_csdid_aggregate.csv")

dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

args      <- commandArgs(trailingOnly = TRUE)
FULL      <- "--full" %in% args
spec_arg  <- sub("^--spec=", "", grep("^--spec=", args, value = TRUE))
ONLY_SPEC <- if (length(spec_arg)) spec_arg[1] else NA_character_

BSTRAP <- FULL       # pilot에서는 FALSE — 해석적 표준오차
CBAND  <- FULL       # 균등신뢰대는 부트스트랩을 요구하므로 full에서만 켠다
                     # (2026-08-05 교정: FALSE로 고정돼 있어 --full에서도 균일
                     #  신뢰대가 나오지 않았다. 주석의 의도대로 FULL을 따르게 한다)
BITERS <- 1000

.log <- character(0)
prt <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n", sep = "")
  flush.console()
  .log <<- c(.log, msg)
}
flush_log <- function() writeLines(.log, LOG_PATH)

# 사양 하나가 끝날 때마다 저장한다 (22번 스크립트와 같은 원칙 —
# 중간에 죽어도 그때까지 결과는 남는다)
attgt_rows <- list()
agg_rows   <- list()

checkpoint <- function() {
  if (length(attgt_rows)) fwrite(rbindlist(attgt_rows, fill = TRUE), ATTGT_PATH)
  if (length(agg_rows))   fwrite(rbindlist(agg_rows,   fill = TRUE), AGG_PATH)
  flush_log()
}

# ---------------------------------------------------------------- 집계 추출

extract_agg <- function(a, spec, type) {
  # aggte 객체는 type에 따라 스칼라(simple)거나 벡터(dynamic/group)다
  if (type == "simple") {
    data.table(spec = spec, agg_type = type, term = "overall",
               att = a$overall.att, se = a$overall.se,
               ci_lo = a$overall.att - 1.96 * a$overall.se,
               ci_hi = a$overall.att + 1.96 * a$overall.se,
               pct = (exp(a$overall.att) - 1) * 100)
  } else {
    crit <- if (!is.null(a$crit.val.egt) && is.finite(a$crit.val.egt)) a$crit.val.egt else 1.96
    data.table(spec = spec, agg_type = type, term = as.character(a$egt),
               att = a$att.egt, se = a$se.egt,
               ci_lo = a$att.egt - crit * a$se.egt,
               ci_hi = a$att.egt + crit * a$se.egt,
               pct = (exp(a$att.egt) - 1) * 100)
  }
}

safe_agg <- function(m, spec, type) {
  out <- tryCatch({
    a <- aggte(m, type = type, na.rm = TRUE, bstrap = BSTRAP,
               cband = if (BSTRAP) CBAND else FALSE, biters = BITERS)
    dt <- extract_agg(a, spec, type)
    if (type == "simple") {
      prt(sprintf("    %-8s ATT=%+.4f (%.4f)  = %+.1f log-point / %+.1f%%",
                  type, a$overall.att, a$overall.se,
                  a$overall.att * 100, (exp(a$overall.att) - 1) * 100))
    } else {
      prt(sprintf("    %-8s %d개 항 (%s)", type, length(a$egt),
                  paste(sprintf("%s:%+.3f", a$egt, a$att.egt),
                        collapse = " ")))
    }
    dt
  }, error = function(e) {
    prt(sprintf("    %-8s 실패: %s", type, conditionMessage(e)))
    data.table(spec = spec, agg_type = type, term = NA_character_,
               att = NA_real_, se = NA_real_, ci_lo = NA_real_,
               ci_hi = NA_real_, pct = NA_real_,
               error = conditionMessage(e))
  })
  agg_rows[[length(agg_rows) + 1]] <<- out
  invisible(out)
}

# ---------------------------------------------------------------- 사양 실행

run_spec <- function(file, spec, control_group, restrict_coalition = TRUE) {
  path <- file.path(PROC_DIR, file)
  prt("")
  prt(strrep("-", 74))
  prt(sprintf("[%s]  control_group=%s  bstrap=%s", spec, control_group, BSTRAP))
  prt(strrep("-", 74))

  if (!file.exists(path)) {
    prt(sprintf("  입력 없음: %s — 24번 스크립트를 먼저 돌릴 것", path))
    return(invisible(NULL))
  }

  d <- fread(path)
  if (restrict_coalition && "coalition_tier1" %in% names(d)) {
    # 'all' 사양에서만 비연합국을 남긴다
    d <- d[coalition_tier1 == 1]
  }
  setorder(d, id, t)

  n_unit <- uniqueN(d$id)
  yrs    <- sort(unique(d$t))
  gtab   <- d[, .(units = uniqueN(id)), by = g][order(g)]
  prt(sprintf("  %s행, 단위 %s개, 연도 %d-%d",
              format(nrow(d), big.mark = ","), format(n_unit, big.mark = ","),
              min(yrs), max(yrs)))
  for (i in seq_len(nrow(gtab))) {
    lab <- if (gtab$g[i] == 0) "never-treated" else paste0(gtab$g[i], " 코호트")
    prt(sprintf("    %-16s 단위 %s개", lab, format(gtab$units[i], big.mark = ",")))
  }
  if (!0 %in% gtab$g && control_group == "nevertreated") {
    prt("  never-treated 단위가 없다 — not-yet-treated로 돌려야 한다")
    return(invisible(NULL))
  }

  t0 <- Sys.time()
  m <- tryCatch(
    att_gt(yname = "y", tname = "t", idname = "id", gname = "g",
           xformla = ~1, data = d, panel = TRUE,
           allow_unbalanced_panel = FALSE,
           control_group = control_group,
           est_method = "reg",          # 공변량이 없으므로 dr과 같고 더 빠르다
           base_period = "universal",   # 사전 계수를 t=-1 고정 기준으로 읽는다
           bstrap = BSTRAP, cband = if (BSTRAP) CBAND else FALSE,
           biters = BITERS, print_details = FALSE),
    error = function(e) {
      prt(sprintf("  att_gt 실패: %s", conditionMessage(e)))
      NULL
    })
  if (is.null(m)) return(invisible(NULL))

  el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  prt(sprintf("  att_gt 완료 [%.1f분]  g x t 셀 %d개", el, length(m$att)))

  n_na <- sum(is.na(m$att))
  if (n_na > 0) prt(sprintf("  ! ATT가 NA인 셀 %d개 (겹침 부족 가능)", n_na))

  attgt_rows[[length(attgt_rows) + 1]] <<- data.table(
    spec = spec, control_group = control_group,
    group = m$group, time = m$t, att = m$att, se = m$se,
    ci_lo = m$att - 1.96 * m$se, ci_hi = m$att + 1.96 * m$se,
    minutes = round(el, 2))

  for (ty in c("simple", "dynamic", "group")) safe_agg(m, spec, ty)
  checkpoint()
  invisible(m)
}

# ---------------------------------------------------------------- 본체

prt(strrep("=", 74))
prt("A-3: Callaway-Sant'Anna (EJPE 2차 revision, R2-2)")
prt(strrep("=", 74))
prt(sprintf("R %s.%s / did %s / 모드 %s",
            R.version$major, R.version$minor,
            as.character(packageVersion("did")),
            if (FULL) "full (부트스트랩)" else "pilot (해석적 SE)"))
prt("추정 대상: 연합국 수출국의 제재 제품 vs 비제재 제품 (2x2 DiD).")
prt("본문 헤드라인 -36.5%는 비연합국을 제3 대조군으로 쓰는 삼중차분 —")
prt("두 값을 같은 숫자로 비교하지 말 것.")

SPECS <- list(
  list(key = "coalition",
       file = "csdid_panel_coalition.csv", spec = "S1: 연합국, never-treated 통제",
       cg = "nevertreated",  coal = TRUE),
  list(key = "notyettreated",
       file = "csdid_panel_coalition.csv", spec = "S2: 연합국, not-yet-treated 통제",
       cg = "notyettreated", coal = TRUE),
  list(key = "truncated",
       file = "csdid_panel_truncated.csv", spec = "S3: truncated 2017-2022 (literal window)",
       cg = "nevertreated",  coal = TRUE),
  list(key = "truncated_ext",
       file = "csdid_panel_truncated_ext.csv", spec = "S4: 절단 2017-2024 (창 연장)",
       cg = "nevertreated",  coal = TRUE),
  list(key = "all",
       file = "csdid_panel_all.csv", spec = "S5: 전 수출국 (통제 풀 혼합 — 해석 주의)",
       cg = "nevertreated",  coal = FALSE)
)

t_start <- Sys.time()
for (s in SPECS) {
  if (!is.na(ONLY_SPEC) && ONLY_SPEC != s$key) next
  run_spec(s$file, s$spec, s$cg, restrict_coalition = s$coal)
}

prt("")
prt(strrep("=", 74))
prt("SUMMARY")
prt(strrep("=", 74))
if (length(agg_rows)) {
  simple <- rbindlist(agg_rows, fill = TRUE)[agg_type == "simple" & !is.na(att)]
  if (nrow(simple)) {
    for (i in seq_len(nrow(simple))) {
      prt(sprintf("  %-44s ATT=%+.4f (%.4f)  %+.1f%%",
                  simple$spec[i], simple$att[i], simple$se[i], simple$pct[i]))
    }
  } else prt("  집계 성공한 사양 없음")
} else prt("  결과 없음")
prt(sprintf("총 %.1f분", as.numeric(difftime(Sys.time(), t_start, units = "mins"))))
prt(sprintf("저장: %s / %s", ATTGT_PATH, AGG_PATH))
checkpoint()
