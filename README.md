# Retail Option Imbalance and the Cross-Section of Stock Returns

Replication and extension of Bryzgalova, Pavlova, and Sikorskaya (2023, JF), 
"Retail Trading in Options and the Rise of the Big Three Wholesalers."

## Overview

This repository contains the code for the sanity check supporting 
"Retail Option Imbalance and the Cross-Section of Stock Returns" 
(working paper, May 2026). Headline finding: a daily-rebalanced long-short 
decile portfolio sorted on call-side SLIM imbalance earns a Fama-French 
six-factor alpha of +26.82 bps/day (t = 2.52) over Nov 2019 - Jun 2021.

## Files

- `sanity_check.py` — Cells 0-9: data loading, ROP construction, FM regression, 
  portfolio sort, reversal horizon test, FF6 alpha, subsample stability, 
  meme-stock exclusion robustness tests.
- `README.md` — this file.

## Data

The SLIM data come from the September 2023 public release by 
[Sikorskaya](https://www.sikorskaya.net/data/). They are not redistributed here 
under the BPS license; download `Retail_trading_in_options_sep23.dta` 
and place under `data/raw/` before running.

Stock prices are pulled from Yahoo Finance via `yfinance`. CRSP integration is 
planned in v2 of this project.

## Reproducing the Main Results

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python sanity_check.py        # outputs T1-T4 + GO/NO-GO decision
```

## Citation

If you use this code or derived results, please cite both the BPS source paper 
and this working paper:

> Liu, Zijun (2026). "Retail Option Imbalance and the Cross-Section of Stock 
> Returns." Working paper, Australian National University. SSRN abstract: [TBA].
> 
> Bryzgalova, Pavlova, and Sikorskaya (2023). "Retail Trading in Options and 
> the Rise of the Big Three Wholesalers." Journal of Finance 78(6): 3465-3514.
