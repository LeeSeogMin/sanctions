# Replication Package

## Sanctions Leakage Dynamics: How Third-Country Rerouting Undermines Trade Restrictions on Russia

This repository contains all code, data, and pre-computed results needed to reproduce every table and figure in the paper. The analysis-ready panel dataset (`baci_russia_panel.parquet`) is included, so all scripts can be run immediately without downloading external data.

## 1. Setup

### System Requirements

- Python 3.13+
- ~500 MB disk space (repository)
- ~4 GB RAM recommended

### Installation

```bash
git clone https://github.com/LeeSeogMin/sanctions.git
cd sanctions
pip install -r requirements.txt
```

Key packages: `pyfixest` (PPML/HDFE), `pandas`, `numpy`, `pyarrow`, `matplotlib`.

## 2. Repository Structure

```
sanctions/
├── README.md
├── requirements.txt
├── code/                          # Replication scripts (run in order)
│   ├── 01_baci_panel_construction.py
│   ├── 02_ppml_triple_did.py
│   ├── 03_rerouting_analysis.py
│   ├── 04_gateway_test.py
│   ├── 05_robustness_checks.py
│   └── 06_figures_tables.py
├── data/
│   ├── raw/                       # Source data files
│   └── processed/                 # Analysis-ready panel (included)
└── output/
    ├── results/                   # CSV estimation outputs (included)
    ├── figures/                   # PNG figures (included)
    └── tables/                    # CSV formatted tables (included)
```

## 3. Data

### Analysis Panel (`data/processed/`)

| File | Rows | Description |
|------|------|-------------|
| `baci_russia_panel.parquet` | 2,064,487 | Exporter × HS6 × year panel for Russia bilateral trade (2015-2024), merged with EU sanctions indicators |
| `sanctions_hs6_master.csv` | 2,304 | HS6-level sanctions classification with enactment dates and categories |

The panel is pre-built from BACI and Chupilkin et al. (2023) data. Scripts 02-06 use this panel directly.

### Raw Source Data (`data/raw/`)

| File | Source | Notes |
|------|--------|-------|
| `EU_sanctions_HS6.dta` | Chupilkin et al. (2023) | 2,304 sanctioned HS6 codes with enactment dates. [OpenICPSR 229004](https://www.openicpsr.org/openicpsr/project/229004) |
| `COMTRADE reporters.dta` | UN Comtrade | Reporter country code mapping |
| `COMTRADE partners.dta` | UN Comtrade | Partner country code mapping |
| `country_codes.csv` | UN Comtrade | ISO country codes |
| `country_codes_V202601.csv` | CEPII BACI | BACI country code mapping |

### External Data (not included due to size)

| Source | Size | Access | Required By |
|--------|------|--------|-------------|
| CEPII BACI HS92 v202601 | ~8.2 GB | [CEPII](http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37) (free registration) | Script 01 only |

**Note**: Script 01 (`01_baci_panel_construction.py`) requires BACI raw data to rebuild the panel from scratch. If you only want to replicate the analysis, you can skip script 01 and start from script 02 using the included `baci_russia_panel.parquet`.

To use script 01, download BACI HS92 v202601 and place it in `data/baci/`.

## 4. Reproducing the Paper

### Main Text

| Paper Output | Script |
|-------------|--------|
| **Table 2**: Summary statistics | `06_figures_tables.py` |
| **Table 3**: PPML triple-DID main results (value, quantity, unit value; T1/T2/EU coalitions) | `02_ppml_triple_did.py` |
| **Table 4**: Category heterogeneity (industrial capacity, dual-use, military, luxury) | `02_ppml_triple_did.py` |
| **Table 5**: Third-country rerouting (OLS with exposure index) | `03_rerouting_analysis.py` |
| **Table 6**: Robustness — balanced-panel PPML, Sun-Abraham IW, gateway test, rerouting identification | `04_gateway_test.py`, `05_robustness_checks.py` |
| **Figure 1**: Staggered event study | `02_ppml_triple_did.py` |
| **Figure 2**: Category heterogeneity coefficients | `06_figures_tables.py` |
| **Figure 3**: Top rerouting countries (exposure index) | `06_figures_tables.py` |

### Running All Analyses

```bash
# Start from script 02 if using the included panel (recommended)
python code/02_ppml_triple_did.py        # Tables 3-4, Figure 1
python code/03_rerouting_analysis.py     # Table 5, Figure 3
python code/04_gateway_test.py           # Table 6 (gateway columns)
python code/05_robustness_checks.py      # Table 6 (robustness)
python code/06_figures_tables.py         # Figures 1-3, Tables 2-5

# Only if rebuilding the panel from BACI raw data:
# python code/01_baci_panel_construction.py
```

All output is written to `output/results/` (CSV), `output/figures/` (PNG), and `output/tables/` (CSV).

## 5. Expected Results

| Specification | Method | Coefficient | Trade Effect |
|--------------|--------|-------------|--------------|
| Sanctioned HS6 × Coalition (preferred) | PPML | -0.473*** | -37.7% |
| Sanctioned HS6 × Coalition (T2) | PPML | -0.523*** | -40.7% |
| Sanctioned HS6 × Coalition (EU) | PPML | -0.487*** | -38.6% |
| Quantity | PPML | -0.501*** | -39.4% |
| Full sample | OLS | -0.570*** | -43.5% |
| Balanced panel | PPML | -0.680*** | -49.4% |
| Sun-Abraham IW | OLS | -0.686*** | -49.7% |

### Category Heterogeneity

| Category | Coefficient | Trade Effect |
|----------|-------------|--------------|
| Industrial capacity | -1.320*** | -73.3% |
| Dual-use | -1.107*** | -66.9% |
| Military technology | -0.417*** | -34.1% |
| Luxury goods | +0.195** | +21.5% |

### Third-Country Rerouting

| Specification | Coefficient |
|--------------|-------------|
| All third countries | +13.8%*** |
| High-exposure countries | +48.3%*** |
| Gateway differential | +20.9%*** |
| Net leakage ratio | 45.4% |

All random seeds are fixed. Point estimates should match across platforms.

**Expected runtime**: approximately 30 minutes on a modern laptop (scripts 02-06).

## 6. Citation

```bibtex
@article{lee2026sanctions,
  title={Sanctions Leakage Dynamics: How Third-Country Rerouting
         Undermines Trade Restrictions on Russia},
  author={Lee, Seog-Min},
  year={2026},
  journal={Working Paper}
}
```

## License

This replication package is provided for academic use. The underlying data sources retain their original licenses and terms of use.

- BACI: CEPII, free for academic use
- Chupilkin et al. sanctions data: [OpenICPSR 229004](https://www.openicpsr.org/openicpsr/project/229004), CC-BY 4.0
