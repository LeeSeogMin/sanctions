# The Trade Effects of Product-Specific Sanctions: Evidence from Russia

Replication code for the paper. The analysis estimates the trade effects of the
staggered EU restrictions on Russia at the six-digit Harmonized System (HS6)
product level, using a PPML triple-difference design on CEPII BACI bilateral
trade data for 2015–2024.

## What this repository contains

Code only. The trade and sanctions data are redistributed by their original
providers under their own licences and are not mirrored here; the table below
gives the access route for each. Every script reads from and writes to paths
relative to the repository root, so run them from the root.

## Software

- Python 3.13
- `pyfixest` **0.40.1** — pinned in `requirements.txt`. The default for singleton
  removal changed in this release, and the scripts check the installed version
  and stop if it differs, because the reproduction gates described below are
  meaningless under a different default.
- R 4.6.0 with `did` 2.5.1 and `data.table`, for the Callaway–Sant'Anna estimates

```bash
pip install -r requirements.txt
```

```r
install.packages(c("did", "data.table"))
```

## Data

| Source | Access | Use |
|:--|:--|:--|
| CEPII BACI HS92 v202601 | Free registration at [cepii.fr](https://www.cepii.fr/CEPII/en/bdd_modele/presentation.asp?id=37) | Bilateral HS6 trade values and quantities, 2015–2024 |
| Chupilkin, Javorcik, Peeva and Plekhanov (2025), [OpenICPSR 229004](https://doi.org/10.3886/E229004V1) | Free download | EU sanctions HS6 coverage and package implementation dates (treatment) |
| Egorov, Korovkin, Makarin and Nigmatulina (2025), OpenICPSR 230342 | CC BY-NC 4.0 | Jurisdiction-specific sanctions lists, used as an external check on treatment measurement |

Place the downloads as follows:

```
phase3_data/BACI_HS92_V202601/BACI_HS92_Y{2015..2024}_V202601.csv
phase3_data/BACI_HS92_V202601/country_codes_V202601.csv
phase3_data/raw/EU_sanctions_HS6.dta
phase3_data/raw/EKMN_230342/data_sa_20241230.dta
```

## Running the analysis

Run `code/05_baci_sanctions_panel.py` first; it writes the panel that every
other script reads. The remaining scripts are independent of one another with
one exception: `37_full_bilateral_joint_ppml.py` checks the saved output of
`33_direct_effect_ekmn_ppml.py` and `35_supplier_replacement.py` before it
estimates anything, so run those two before it.

```bash
python code/05_baci_sanctions_panel.py
```

| Script | Produces |
|:--|:--|
| `05_baci_sanctions_panel.py` | The estimation panel (run first) |
| `12_ppml_event_study.py` | Aggregate event study, PPML — Figure 2, Appendix K |
| `13_category_dynamics.py` | Category event studies, 2022 cohort — Figure 3 |
| `14_price_effects.py` | Value / quantity / unit-value decomposition — Appendix J |
| `15_category_static_ppml.py` | Category static effects, PPML — Table 4 |
| `16_military_expharma.py` | Military technology excluding HS 30/33 — Table 4, Figure 3 |
| `17_make_revision_figures.py` | Figures 2, 3 and the Appendix K figure |
| `18_ekmn_concordance.py` | Agreement between the two sanctions lists — Section 3.1 |
| `20_agri_embargo_robustness.py` | Excluding HS 01–24 — Table 6 |
| `21_save_reported_aggregates.py` | Descriptive aggregates reported in the text |
| `22_rerouting_ppml_staggered.py` | Supplier replacement, PPML — Table 5 Panel A |
| `23_bridge_ppml_joint.py` | Coalition and non-coalition sides jointly — Section 4.1 |
| `24_export_csdid_panel.py` | Panel export for the CS-DID estimator |
| `25_csdid.R` | Callaway–Sant'Anna estimates — Table 6, Appendix L |
| `26_gateway_ppml.py` | Upstream gateway test, Equation (5) — Table 6, Appendix I |
| `27_rerouting_diagnostics.py` | Influence diagnostics for supplier replacement — Table 5 |
| `28_csdid_pretrend.R` | CS-DID timing diagnostic — Appendix L |
| `29_headline_diagnostics.py` | Leave-one-country-out for the benchmark — Section 4.1 |
| `30_baci_coverage_check.py` | Post-2022 BACI coverage of Russia flows — Section 6.4 |
| `33_direct_effect_ekmn_ppml.py` | Direct effect under the jurisdiction-specific treatment definition — required input to the gate in `37` |
| `35_supplier_replacement.py` | Supplier replacement with country and size exclusions — Section 4.4, Table 5 |
| `37_full_bilateral_joint_ppml.py` | Three channels in one bilateral PPML — Appendix M, Table M1 |
| `39_saturated_joint_feasibility.py` | Fixed-effect dimensions of the saturated joint PPML — Appendix M.1 |
| `revision_methodology_utils.py` | Shared helpers for `33`, `35`, `37` and `39` (not run directly) |

The longer estimations expose `--stage` so that a gate or a reduced-sample pilot
can be run before the full fit. Several take one to three hours on the full
panel; each writes its results as it goes, so an interrupted run keeps what it
had finished.

### Reproduction gates

Several scripts recompute a value that is already known before estimating
anything new, and stop with a non-zero exit code if it does not reproduce
within tolerance. This is deliberate: it prevents a plausible-looking
coefficient from a subtly different sample from being taken as a result. If a
gate fails, the sample or the package version differs from the one used here,
and the diagnostics downstream of it are not meaningful.

## Reading the estimates

The benchmark coefficient is a **coalition versus non-coalition differential**:
the sanctioned-versus-non-sanctioned change is more negative among coalition
exporters than among non-coalition exporters. It is not the standalone change
in coalition exports.

The direct, supplier-replacement and gateway specifications condition on
different trade links — coalition exports to Russia, non-coalition exports to
Russia, and coalition exports to potential gateways. Their coefficients are not
additive and should not be read as coming from one structural equation.

Some comments in the code are in Korean. The module headers, the script map
above, and the printed output labels are in English.

## Citation

Please cite the paper. Data sources should be cited separately, following the
terms set by each provider.
