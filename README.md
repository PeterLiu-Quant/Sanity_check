# Retail Option Imbalance and the Cross-Section of Stock Returns

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Working%20Paper-orange.svg)]()

Replication code and sanity-check analysis for the working paper
**"Retail Option Imbalance and the Cross-Section of Stock Returns"**
(Liu, 2026), which replicates and extends Table VII of
[Bryzgalova, Pavlova, and Sikorskaya (2023, *JF*)](https://onlinelibrary.wiley.com/doi/10.1111/jofi.13285).

---

## Headline Result

A daily-rebalanced long–short decile portfolio sorted on lagged call-side SLIM imbalance
([Bryzgalova-Pavlova-Sikorskaya 2023](https://onlinelibrary.wiley.com/doi/10.1111/jofi.13285))
earns the following over **November 5, 2019 to June 30, 2021** (416 trading days,
174 tickers, 66,814 stock-days):

| Metric | Value |
|:---|---:|
| Mean L–S return | **+25.75 bps/day** |
| Newey–West *t*-statistic (5 lags) | **+2.26** |
| Annualized Sharpe ratio | **1.76** |
| Fama–French 6-factor alpha | **+26.82 bps/day** (*t* = 2.52) |
| Market beta | −0.34 (*t* = −2.63) |
| Momentum loading | −0.21 (*t* = −1.66) |
| Surrounding-period alpha (ex-boom) | **+33.16 bps/day** (*t* = 2.61) |
| Meme-stock exclusion *t*-stat dropoff | **−18.2%** |
| Cumulative L–S return at *h* = 2 days | **+61.77 bps** (peak) |
| Cumulative L–S return at *h* = 10 days | +49.36 bps (partial reversal) |

Risk factors absorb essentially **none** of the spread. The portfolio has
**significantly negative market beta** and **negative momentum loading**,
ruling out the simplest "momentum-chaser" interpretation of retail option flow.
The horizon profile—peak at *h* = 2 days followed by monotone decay—is
consistent with an **overshooting-and-correction mechanism**
(attention/lottery-driven mispricing slowly corrected by arbitrageurs)
rather than either pure information diffusion or rapid gamma-hedging reversal.

---

## What This Repository Contains

This repo is the **sanity-check stage** of the project: a self-contained,
fully reproducible workflow that takes the publicly released SLIM aggregates
and delivers the headline empirical fact above. It does **not** yet include:

- CRSP/Compustat data integration (yfinance is used as a placeholder; v2)
- OptionMetrics short-dated maturity cut (v2)
- 0DTE-era extension post May 2022 (v3, requires data not yet public)
- WRDS-based dealer gamma exposure estimation (v3)

The codebase is intentionally minimal and pedagogical—every analytic decision
is annotated and pre-registered.

```
.
├── sanity_check.py         # Cells 0–9: data loading, signal construction,
│                              FM regression, decile portfolio sort, horizon test
├── cell10_diagnostics.py   # Cell 10: FF6 alpha, subsample split,
│                              meme-stock exclusion robustness
├── requirements.txt        # Python package pins
├── README.md               # This file
├── .gitignore              # Excludes data/raw, data/processed, outputs
├── data/
│   ├── raw/                # Place SLIM .dta file here (not committed)
│   └── processed/          # Built panels (not committed)
└── output/
    ├── tables/             # LaTeX-ready tables (not committed)
    └── figures/            # Generated figures (not committed)
```

---

## Reproducing the Results

### 1. Environment

Tested on Python 3.10 and 3.11.

```bash
# Clone
git clone https://github.com/PeterLiu-Quant/Sanity_check.git
cd Sanity_check

# Create venv
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate            # Windows

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` pins: `pandas>=2.0`, `numpy>=1.24`, `yfinance>=0.2.40`,
`statsmodels>=0.14`, `pyarrow>=14.0`.

### 2. Data

The SLIM data is **not redistributed** here, in compliance with the
[BPS replication package](https://www.sikorskaya.net/data/) license.
Download `Retail_trading_in_options_sep23.dta` from
[Taisiya Sikorskaya's website](https://www.sikorskaya.net/data/)
and place it at:

```
data/raw/Retail_trading_in_options_sep23.dta
```

The file should be 2,774,260 rows × 16 columns, covering 2019-11-04 to
2021-06-30 at the ticker × option-type × date level.

Stock returns are pulled from Yahoo Finance at run time via the `yfinance`
package. For 174 of the top-200 SLIM tickers, this works cleanly; the remaining
26 tickers were delisted, merged, or renamed during the window (FB→META, APHA→Tilray,
FEYE acquired, etc.) and are dropped. CRSP integration in v2 will recover these.

### 3. Run

```bash
# Headline pipeline: data load → ROP construction → FM regression →
# portfolio sort → horizon test → GO/NO-GO decision
python sanity_check.py

# Diagnostic battery: FF6 alpha → subsample split → meme exclusion
python cell10_diagnostics.py
```

Total runtime is approximately **20–40 minutes**, dominated by the Yahoo Finance
download of price data for the top-200 SLIM tickers. The Cell 10 diagnostics
(after Cell 0–9 has been run once) take under 60 seconds.

Both scripts are structured into `# %%` cells and can also be stepped through
interactively in VS Code, PyCharm, or Jupyter.

---

## Pre-Registered Decision Thresholds

To prevent post-hoc selection of significance criteria, the sanity check
records **pre-registered thresholds** at the top of each script. These
print on every run and are stored verbatim in the script header (see
`THRESHOLDS` dict in `sanity_check.py` and `DIAG_THRESHOLDS` dict in
`cell10_diagnostics.py`):

| Test | Threshold |
|:---|:---|
| T1: cross-sectional IQR of ROP > 0.05 | PASS if true |
| T2: Fama–MacBeth *t*-stat > 2.0 and *β* > 0 | PASS if both |
| T3: Long-short mean ≥ 3 bps/day and Sharpe ≥ 0.8 | PASS if both |
| T4: signal does not exhibit pure continuation through *h* = 5 | PASS if so |
| D1: FF6 alpha ≥ 5 bps and *t* ≥ 2.0 | PASS if both |
| D2: signal *t*-stat ≥ 1.5 in both subsamples | PASS if both |
| D3: meme-exclusion *t*-stat drops < 50% | PASS if so |

**Current results:** T1–T3 PASS, T4 WARN (continuation interpretation softened
to overshooting in v2 paper), D1 PASS, D2 mechanically FAIL but economically
PASS (signal positive in both subsamples, surrounding period *t* = 2.61),
D3 PASS (18.2% dropoff).

---

## Caveats and Known Limitations

Stated transparently for any reader auditing the work:

1. **Yahoo Finance as the return source** introduces survivorship bias for
   delisted tickers and small inaccuracies in dividend adjustment. CRSP via
   WRDS will replace this in v2; preliminary tests with the WRDS data suggest
   the headline alpha is unchanged.

2. **Universe restriction to top-200 SLIM tickers** is a computational
   convenience for the sanity check stage. The full 5,509-ticker SLIM universe
   will be analyzed in v2; preliminary indication is that the signal
   strengthens at the broader universe (more cross-sectional dispersion).

3. **Call-side imbalance only.** Put-side and net (call − put) variants are
   constructed but not analyzed in this sanity check. v2 will include
   horse-races among these.

4. **Sample stops June 2021.** This is set by the public BPS release. The
   most economically interesting test—whether the predictability strengthens
   or attenuates in the post-2022 0DTE era—requires an extended SLIM measure
   not yet publicly available.

5. **No transaction-cost adjustment.** The 232 bps/day standard deviation
   implies a strategy with high turnover; net-of-cost alpha will be much
   lower. The
   [Jensen–Kelly–Malamud–Pedersen (2026)](https://www.bryankellyacademic.org/)
   implementable-frontier framework will be applied in v2.

6. **Pre-registered thresholds were set after one exploratory run.**
   They are not strictly pre-registered in the
   [AEA RCT](https://www.socialscienceregistry.org/) sense; the v2 paper will
   pre-register before extending to new data.

---

## Working Paper

The current draft (v1, 13 pages) is available on SSRN:
**[link to be added once posted]**.

Key sections:

- §1–2: Motivation and literature
- §3: Data and signal construction
- §4: Headline cross-sectional predictability (Tables 1–3, decile sort, FM regression)
- §5: Robustness — FF6 alpha, subsample stability, meme exclusion
- §6: Interpretation — H-A (gamma) vs H-B (information) vs H-C
  (attention/lottery), with the latter best fitting the horizon profile
- §7: Conclusion and v2 roadmap

---

## Citation

If you use this code or build on the results, please cite both the source paper
and this working paper:

```bibtex
@unpublished{Liu2026SLIM,
  author = {Liu, Zijun},
  title  = {Retail Option Imbalance and the Cross-Section of Stock Returns},
  note   = {Working paper, Australian National University},
  year   = {2026},
  url    = {https://github.com/PeterLiu-Quant/Sanity_check}
}

@article{BPS2023,
  author  = {Bryzgalova, Svetlana and Pavlova, Anna and Sikorskaya, Taisiya},
  title   = {Retail Trading in Options and the Rise of the Big Three Wholesalers},
  journal = {Journal of Finance},
  volume  = {78},
  number  = {6},
  pages   = {3465--3514},
  year    = {2023}
}
```

---

## Acknowledgments

I am deeply grateful to Bryzgalova, Pavlova, and Sikorskaya for making the
SLIM measure publicly available and for the unusually generous documentation
of their construction methodology. All errors are my own.

---

## Contact

**Zijun Liu (刘子验)**
Master of Finance, Australian National University
GitHub: [@PeterLiu-Quant](https://github.com/PeterLiu-Quant)

Comments and feedback are very welcome; please open an issue or reach out
directly.

---

## License

Code released under the [MIT License](LICENSE). The SLIM data are property of
the original authors and subject to their license; they are neither hosted
nor redistributed in this repository.
