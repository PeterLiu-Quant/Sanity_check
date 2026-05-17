"""
Retail Option Pressure (ROP) — Sanity Check
============================================
Minimum viable empirical test of the hypothesis:
    Retail short-dated option call-vs-put pressure (ROP) on a stock
    predicts that stock's next-day cross-sectional return.

What this script does
---------------------
1. Loads Bryzgalova-Pavlova-Sikorskaya (2023) public SLIM data
   (downloaded from https://sabryzgalova.com/data-and-code).
2. Constructs a stock-day ROP signal from SLIM aggregates.
3. Pulls daily prices via Yahoo Finance for the SLIM tickers.
4. Runs four sanity tests:
   T1. Cross-sectional dispersion of ROP (descriptive).
   T2. Fama-MacBeth predictive regression: ROP_{i,t-1} -> R_{i,t}.
   T3. Decile long-short portfolio sort: ROP-decile-10 minus decile-1.
   T4. Reversal test: cumulative L-S return at h = 1, 3, 5, 10 days.
5. Prints a clear GO / NO-GO recommendation based on pre-registered thresholds.

This is a sanity check, NOT the publishable paper.
Cells are marked with # %% so the file runs cell-by-cell in
VS Code, PyCharm, or Jupyter, and also runs end-to-end as `python sanity_check.py`.

Author: Zijun (Peter) Liu — research prototype
"""

# %% Cell 0 — Imports and setup ------------------------------------------------
from __future__ import annotations
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS  # noqa: F401  (optional later)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

# Project paths --------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_RAW = PROJECT_DIR / "data" / "raw"
DATA_PROC = PROJECT_DIR / "data" / "processed"
OUTPUT = PROJECT_DIR / "output"
for p in (DATA_RAW, DATA_PROC, OUTPUT):
    p.mkdir(parents=True, exist_ok=True)

# Pre-registered sanity-check thresholds (decide BEFORE seeing results) ------
THRESHOLDS = {
    "t_stat_FM_min": 2.0,        # FM regression t-stat on ROP must exceed this
    "LS_alpha_bps_min": 3.0,     # daily long-short raw return ≥ 3 bps
    "LS_sharpe_min": 0.8,        # annualized Sharpe of L-S portfolio ≥ 0.8
    "reversal_required": True,   # H-A/H-C predict positive contemp + reversal by day 5
}

print("=" * 78)
print(" ROP Sanity Check  —  pre-registered decision thresholds")
print("=" * 78)
for k, v in THRESHOLDS.items():
    print(f"  {k:<28} = {v}")
print()

# To check Data-Raw in the right place
print(PROJECT_DIR)
print(DATA_RAW)
print(DATA_RAW.exists())

# %% Cell 1 — Load SLIM data (BPS Sep-2023 release) ----------------------------
# File: data/raw/Retail_trading_in_options_sep23.dta
# Structure: 2,774,260 rows at (ticker, option_type, date) granularity
# Variables we use:
#   ticker       : str
#   option_type  : 'C' = call, 'P' = put
#   date         : datetime
#   slan_imb     : SLIM (retail) buy-minus-sell imbalance — BPS's headline signal
#   slan_vimb    : volume-weighted imbalance
#   slan_dvimb   : dollar-volume-weighted imbalance
#   slan_vol     : SLIM contract volume (used for liquidity filter)
#   PERMNO       : CRSP permanent identifier (useful for later merging)

slim_path = DATA_RAW / "Retail_trading_in_options_sep23.dta"
if not slim_path.exists():
    print(f"!! File not found: {slim_path}")
    sys.exit(1)

print(f"[1] Loading SLIM data from: {slim_path.name}")
slim_raw = pd.read_stata(slim_path)
print(f"    rows: {len(slim_raw):,}  cols: {slim_raw.shape[1]}")
print(f"    columns: {list(slim_raw.columns)}")
print(slim_raw.head(3))

# Standardize types --------------------------------------------------------
slim_raw["date"] = pd.to_datetime(slim_raw["date"])
slim_raw["ticker"] = slim_raw["ticker"].astype(str).str.strip().str.upper()
slim_raw["option_type"] = slim_raw["option_type"].astype(str).str.strip().str.upper()

print(f"\n[1] Standardized.")
print(f"    Unique tickers: {slim_raw['ticker'].nunique():,}")
print(f"    Date range:     {slim_raw['date'].min().date()} → {slim_raw['date'].max().date()}")
print(f"    Option types:   {sorted(slim_raw['option_type'].unique())}")


# %% Cell 2 — Construct ROP signal --------------------------------------------
# BPS data already provides slan_imb at (ticker, option_type, date) level.
# We build the ROP signal in TWO versions:
#
#   ROP-call  : just the call-side SLIM imbalance.
#               Positive = retail net-buying calls = bullish pressure.
#   ROP-net   : call_imb minus put_imb (volume-weighted).
#               Positive = retail bullish on calls AND/OR bearish on puts.
#
# Start with ROP-call (simpler). If signal is significant, extend to ROP-net.

# --- Liquidity filter: drop ticker-days with very thin SLIM volume -----------
# BPS use roughly $1M cum volume; we'll be looser since we work at call/put
# level. Require SLIM contract volume >= 50 to avoid pure-noise observations.
slim_filtered = slim_raw[slim_raw["slan_vol"] >= 50].copy()
print(f"[2] After liquidity filter (slan_vol >= 50): {len(slim_filtered):,} rows")

# --- ROP-call: keep only call rows, use slan_imb directly --------------------
slim_call = slim_filtered[slim_filtered["option_type"] == "C"].copy()
slim_call = slim_call[["ticker", "date", "slan_imb", "slan_vimb", "slan_dvimb", "slan_vol"]]
slim_call = slim_call.rename(columns={
    "slan_imb":   "rop_call",       # contract-count imbalance
    "slan_vimb":  "rop_call_vimb",  # volume-weighted
    "slan_dvimb": "rop_call_dvimb", # dollar-volume-weighted
})
print(f"    Call-only rows kept: {len(slim_call):,}")
print(f"    Unique stock-days:   {len(slim_call.drop_duplicates(['ticker', 'date'])):,}")

# Use 'rop_call' (count-based imbalance) as the headline signal -----
# Range should be roughly [-1, +1] for an imbalance measure.
signal = "rop_call"
print(f"\n[2] Headline signal: '{signal}'")
print(slim_call[[signal, "rop_call_vimb", "rop_call_dvimb"]].describe().T)

# --- Cross-sectional standardization (daily) ---------------------------------
def _winsorize(s, lo=0.01, hi=0.99):
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)

slim_call["rop_w"] = slim_call.groupby("date")[signal].transform(_winsorize)
slim_call["rop_z"] = slim_call.groupby("date")["rop_w"].transform(
    lambda s: (s - s.mean()) / s.std() if s.std() > 0 else s * 0
)

# This is the dataframe Cell 4 will use --------------------------------------
slim = slim_call.copy()
print(f"\n[2] Final signal frame ready: {len(slim):,} stock-days.")

##
dup = slim.groupby(["ticker", "date"]).size()
print(f"\n[2-check] Duplicates: max {dup.max()} per (ticker, date)")
if dup.max() > 1:
    print(f"           Will keep only the row with largest slan_vol per (ticker, date).")
    slim = (slim.sort_values("slan_vol", ascending=False)
                .drop_duplicates(["ticker", "date"], keep="first")
                .reset_index(drop=True))
    print(f"           After dedup: {len(slim):,} stock-days")


# %% Cell 3 — Pull daily prices from Yahoo Finance -----------------------------
# We need next-day forward returns at the ticker level.
# Strategy: download only the universe present in SLIM, restricted to the
# overlapping date range, with a 30-day buffer at the end for forward returns.


# Temporary — first pass, top-200 most-active tickers by SLIM volume
top_universe = (slim.groupby("ticker")["slan_vol"]
                    .sum().nlargest(200).index.tolist())
slim = slim[slim["ticker"].isin(top_universe)].copy()
print(f"[3-pre] Restricted to top-200 by SLIM volume: {len(slim):,} stock-days")
##
universe = sorted(slim["ticker"].unique())

date_min = slim["date"].min()
date_max = slim["date"].max() + pd.Timedelta(days=30)
print(f"[3] Downloading prices for {len(universe)} tickers via yfinance.")
print(f"    range: {date_min.date()} → {date_max.date()}")

# yfinance is rate-limited; download in batches of ~50
BATCH = 50
price_frames = []
for i in range(0, len(universe), BATCH):
    chunk = universe[i:i + BATCH]
    try:
        px = yf.download(
            chunk, start=date_min, end=date_max,
            auto_adjust=True, progress=False, threads=True, group_by="ticker"
        )
    except Exception as e:
        print(f"    batch {i}: error {e}")
        continue
    # Reshape (long) -- yfinance returns MultiIndex columns when >1 ticker
    if isinstance(px.columns, pd.MultiIndex):
        for t in chunk:
            if (t, "Close") in px.columns:
                sub = pd.DataFrame({
                    "ticker": t,
                    "date":   px.index,
                    "close":  px[(t, "Close")].values,
                })
                price_frames.append(sub)
    else:
        sub = pd.DataFrame({
            "ticker": chunk[0],
            "date":   px.index,
            "close":  px["Close"].values,
        })
        price_frames.append(sub)
    print(f"    batch {i // BATCH + 1}/{(len(universe) + BATCH - 1) // BATCH} done")

prices = pd.concat(price_frames, ignore_index=True).dropna(subset=["close"])
prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

# Daily simple returns, plus h-day forward returns up to 10 trading days ------
prices["ret"] = prices.groupby("ticker")["close"].pct_change()
for h in (1, 2, 3, 5, 10):
    prices[f"fwd_{h}d"] = prices.groupby("ticker")["close"].pct_change(h).shift(-h)

print(f"[3] Prices fetched: {len(prices):,} rows, "
      f"{prices['ticker'].nunique()} tickers retained.")
print()


# %% Cell 4 — Merge ROP signal with forward returns ---------------------------
# Critical timing: ROP measured at end of day t-1 predicts return on day t.
# We align by shifting ROP forward one trading day per ticker.

slim_lag = slim[["ticker", "date", "rop_w", "rop_z"]].copy()
slim_lag = slim_lag.sort_values(["ticker", "date"])
slim_lag["rop_w_lag1"] = slim_lag.groupby("ticker")["rop_w"].shift(1)
slim_lag["rop_z_lag1"] = slim_lag.groupby("ticker")["rop_z"].shift(1)

panel = prices.merge(slim_lag, on=["ticker", "date"], how="inner")
panel = panel.dropna(subset=["ret", "rop_z_lag1"]).reset_index(drop=True)

print(f"[4] Merged panel: {len(panel):,} stock-days, "
      f"{panel['ticker'].nunique()} tickers.")
print(f"    date range: {panel['date'].min().date()} → {panel['date'].max().date()}")
print()


# %% Cell 5 — Test T1: Cross-sectional dispersion of ROP -----------------------
print("=" * 78)
print(" T1 — Cross-sectional dispersion of ROP")
print("=" * 78)
xs_std = panel.groupby("date")["rop_w"].std()
xs_iqr = panel.groupby("date")["rop_w"].quantile(0.75) - panel.groupby("date")["rop_w"].quantile(0.25)
print(f"  Mean cross-sectional std of raw ROP: {xs_std.mean():.4f}")
print(f"  Mean cross-sectional IQR of raw ROP: {xs_iqr.mean():.4f}")
print(f"  -> If IQR > 0.05, there is enough variation to identify a cross-sectional effect.")
t1_pass = xs_iqr.mean() > 0.05
print(f"  T1 result: {'PASS' if t1_pass else 'WARN — low variation, may need different sample'}")
print()


# %% Cell 6 — Test T2: Fama-MacBeth predictive regression ----------------------
print("=" * 78)
print(" T2 — Fama-MacBeth regression: R_{i,t} on ROP_{i,t-1}")
print("=" * 78)

def fama_macbeth(df: pd.DataFrame, y_col: str, x_col: str) -> tuple[float, float, float, int]:
    """Return (mean beta, NW t-stat, R^2 mean, num cross-sections)."""
    betas, rsqs = [], []
    for d, g in df.groupby("date"):
        if len(g) < 20 or g[x_col].std() == 0:
            continue
        X = sm.add_constant(g[x_col].values)
        try:
            res = sm.OLS(g[y_col].values, X).fit()
            betas.append(res.params[1])
            rsqs.append(res.rsquared)
        except Exception:
            continue
    if len(betas) < 30:
        return (np.nan, np.nan, np.nan, len(betas))
    b = pd.Series(betas)
    mean_b = b.mean()
    # Newey-West with lag 5
    nw = sm.OLS(b.values, np.ones(len(b))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5}
    )
    return (mean_b, float(nw.tvalues[0]), float(np.mean(rsqs)), len(betas))

beta, tstat, r2, n_days = fama_macbeth(panel, "ret", "rop_z_lag1")
print(f"  Mean coefficient on ROP_z (lagged 1 day): {beta:+.6f}")
print(f"  Newey-West t-statistic (5 lags):          {tstat:+.3f}")
print(f"  Mean cross-sectional R^2:                 {r2:.4f}")
print(f"  Days with valid cross-section:            {n_days:,}")
print(f"  Implied daily basis points (per 1-σ ROP): {beta * 1e4:+.2f} bps")
t2_pass = (not np.isnan(tstat)) and (tstat > THRESHOLDS["t_stat_FM_min"]) and (beta > 0)
print(f"  T2 result: {'PASS' if t2_pass else 'FAIL'}")
print()


# %% Cell 7 — Test T3: Decile portfolio sort and long-short ---------------------
print("=" * 78)
print(" T3 — Decile portfolio sort on lagged ROP")
print("=" * 78)

def assign_decile(s: pd.Series) -> pd.Series:
    """Daily cross-sectional decile (1 = lowest ROP, 10 = highest)."""
    try:
        return pd.qcut(s, 10, labels=False, duplicates="drop") + 1
    except ValueError:
        return pd.Series(np.nan, index=s.index)

panel["decile"] = panel.groupby("date")["rop_w_lag1"].transform(assign_decile)
panel_sort = panel.dropna(subset=["decile"]).copy()
panel_sort["decile"] = panel_sort["decile"].astype(int)

# Equal-weighted portfolio returns by decile, by day
port = panel_sort.groupby(["date", "decile"])["ret"].mean().unstack("decile")
port = port.dropna(how="any")
ls = port[port.columns.max()] - port[port.columns.min()]

print(f"  Decile means (daily, bps):")
for d in sorted(port.columns):
    print(f"    D{d}: {port[d].mean() * 1e4:+.2f} bps")
ls_mean_bps = ls.mean() * 1e4
ls_sharpe = ls.mean() / ls.std() * np.sqrt(252) if ls.std() > 0 else np.nan
ls_tstat = ls.mean() / (ls.std() / np.sqrt(len(ls))) if ls.std() > 0 else np.nan
print(f"  Long-short (D10 - D1):")
print(f"    Mean daily return:           {ls_mean_bps:+.2f} bps")
print(f"    Std daily return:            {ls.std() * 1e4:.2f} bps")
print(f"    t-statistic:                 {ls_tstat:+.3f}")
print(f"    Annualized Sharpe ratio:     {ls_sharpe:.3f}")
t3_pass = (ls_mean_bps > THRESHOLDS["LS_alpha_bps_min"]
           and ls_sharpe > THRESHOLDS["LS_sharpe_min"])
print(f"  T3 result: {'PASS' if t3_pass else 'FAIL'}")
print()


# %% Cell 8 — Test T4: Reversal horizon ---------------------------------------
# H-A (gamma hedging): same-day pop + reversal by day 5
# H-B (information):  continuation through day 10
# H-C (lottery):      same-day pop + slow reversal over days 3-10
print("=" * 78)
print(" T4 — Reversal horizon test")
print("=" * 78)

# Build cumulative return of L-S portfolio over h days using forward returns
# Use D10-D1 portfolios re-built on fwd_h
horizon_results = {}
for h in (1, 2, 3, 5, 10):
    col = f"fwd_{h}d"
    if col not in panel_sort.columns:
        continue
    p = panel_sort.dropna(subset=[col, "decile"])
    if len(p) < 1000:
        continue
    port_h = p.groupby(["date", "decile"])[col].mean().unstack("decile")
    port_h = port_h.dropna(how="any")
    ls_h = port_h[port_h.columns.max()] - port_h[port_h.columns.min()]
    horizon_results[h] = {
        "mean_bps": ls_h.mean() * 1e4,
        "t_stat":   ls_h.mean() / (ls_h.std() / np.sqrt(len(ls_h))),
    }

print(f"  Cumulative L-S forward return (D10 - D1):")
print(f"    Horizon       Mean (bps)   t-stat")
for h, r in horizon_results.items():
    print(f"    h = {h:2d} days  {r['mean_bps']:+8.2f}    {r['t_stat']:+6.2f}")

# Reversal: cumulative return at h=5 should be less than at h=1
t4_pass = False
if 1 in horizon_results and 5 in horizon_results:
    cum_1 = horizon_results[1]["mean_bps"]
    cum_5 = horizon_results[5]["mean_bps"]
    if cum_1 > 0 and cum_5 < cum_1 * 1.5:  # partial reversal/no strong continuation
        t4_pass = True
    print(f"\n  Reversal signature: 1-day = {cum_1:+.1f} bps, 5-day = {cum_5:+.1f} bps")
    if cum_5 > cum_1 * 1.5:
        print(f"  -> Looks like continuation (H-B / information story).")
    elif cum_5 < cum_1 * 0.5:
        print(f"  -> Strong reversal (consistent with H-A gamma hedging).")
    else:
        print(f"  -> Partial reversal (consistent with H-C lottery/attention).")
print(f"  T4 result: {'PASS' if t4_pass else 'WARN'}")
print()


# %% Cell 9 — Decision: GO / NO-GO ---------------------------------------------
print("=" * 78)
print(" SANITY CHECK SUMMARY")
print("=" * 78)
print(f"  T1 (variation):       {'PASS' if t1_pass else 'WARN'}")
print(f"  T2 (FM regression):   {'PASS' if t2_pass else 'FAIL'}")
print(f"  T3 (portfolio sort):  {'PASS' if t3_pass else 'FAIL'}")
print(f"  T4 (reversal shape):  {'PASS' if t4_pass else 'WARN'}")
print()

n_pass = sum([t1_pass, t2_pass, t3_pass, t4_pass])
if t2_pass and t3_pass:
    decision = "GO"
    msg = ("Both headline tests pass.  Proceed with the full 12-week plan: "
           "request WRDS/OptionMetrics access, build the proper ROP with "
           "short-dated cut, and write the paper.")
elif t2_pass or t3_pass:
    decision = "CONDITIONAL GO"
    msg = ("One headline test passes.  The signal exists but is weaker than "
           "ideal.  Likely solutions: (a) restrict to the post-2020 retail "
           "boom subsample; (b) restrict to retail-popular tickers (top 200 "
           "WSB-mentioned); (c) wait for short-dated cut from OPRA.")
else:
    decision = "NO-GO"
    msg = ("Neither headline test passes.  Before abandoning the project: "
           "(a) verify SLIM data is correctly merged; (b) check the sample "
           "covers 2020-2021 (the retail boom); (c) consider that the public "
           "SLIM aggregation across ALL maturities dilutes the signal — the "
           "short-dated cut may still work.")
print(f"  >>> DECISION: {decision}")
print(f"  {msg}")
print()
print(f"  Run timestamp: {datetime.now().isoformat(timespec='seconds')}")
print("=" * 78)

# Save the merged panel and a small summary to disk for later inspection ------
panel.to_parquet(DATA_PROC / "panel_mvp.parquet", index=False)
summary = pd.DataFrame({
    "test":   ["T1", "T2", "T3", "T4"],
    "result": [t1_pass, t2_pass, t3_pass, t4_pass],
    "note":   [
        f"xs IQR mean = {xs_iqr.mean():.4f}",
        f"FM beta = {beta:+.6f}, t = {tstat:+.3f}",
        f"LS mean = {ls_mean_bps:+.2f} bps, Sharpe = {ls_sharpe:.3f}",
        f"1d -> 5d cum: {horizon_results.get(1, {}).get('mean_bps', np.nan):+.1f} "
        f"-> {horizon_results.get(5, {}).get('mean_bps', np.nan):+.1f}",
    ],
})
summary.to_csv(OUTPUT / "sanity_summary.csv", index=False)
print(f"  Outputs saved: {DATA_PROC / 'panel_mvp.parquet'}, "
      f"{OUTPUT / 'sanity_summary.csv'}")


# %% Cell 10 — Diagnostic battery: is the signal real or just retail-boom momentum?
# Three checks before committing to the full paper:
#   D1. FF6 alpha — how much of the +20 bps L-S return survives risk adjustment?
#   D2. Sample-period split — does the signal exist outside the retail boom?
#   D3. Meme-stock exclusion — does it survive without GME/AMC/etc?
#
# Pre-registered decision rules (set before seeing results):
#   PASS criteria:
#     D1: FF6 alpha t-stat > 2.0 AND alpha >= 5 bps/day
#     D2: signal positive and significant (t > 1.5) in BOTH halves
#     D3: t-stat drops by less than 50% after meme exclusion
#
# If all three pass → signal is robust, commit to full paper.
# If D1 fails → need to control for momentum; signal may be momentum in disguise.
# If D2 fails → narrative becomes "transient retail-boom phenomenon," not anomaly.
# If D3 fails → effect is meme-driven; need to reframe scope.

import urllib.request
import zipfile
import io

print("=" * 78)
print(" CELL 10 — DIAGNOSTIC BATTERY")
print("=" * 78)

# Sanity-check the panel is in memory from earlier cells
assert "panel_sort" in dir(), "Run Cells 0–9 first; panel_sort must exist."
assert "ls" in dir(), "Run Cell 7 first; ls (long-short series) must exist."

# Pre-registered diagnostic thresholds ----------------------------------------
DIAG_THRESHOLDS = {
    "D1_FF6_alpha_bps_min": 5.0,
    "D1_FF6_tstat_min":     2.0,
    "D2_subsample_tstat_min": 1.5,
    "D3_tstat_dropoff_max_pct": 50.0,  # max acceptable drop in t-stat
}
print("\n  Pre-registered diagnostic thresholds:")
for k, v in DIAG_THRESHOLDS.items():
    print(f"    {k:<32} = {v}")
print()


# =============================================================================
# D1 — FF6 alpha test
# =============================================================================
print("=" * 78)
print(" D1 — Fama-French 6-factor alpha of the long-short portfolio")
print("=" * 78)

# Download Ken French daily FF5 + Momentum factors -----------------------------
FF_URL_5F  = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
FF_URL_MOM = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

def _fetch_french_zip(url: str) -> str:
    """Download a Ken French zip and return the CSV text."""
    with urllib.request.urlopen(url, timeout=60) as resp:
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        return z.read(csv_name).decode("latin-1")

def _parse_french_csv(text: str) -> pd.DataFrame:
    """Parse Ken French daily CSV — skips header, stops at the trailing copyright block."""
    lines = text.splitlines()
    # Find the row where the first column is an 8-digit date
    start = next(i for i, ln in enumerate(lines)
                 if ln.strip()[:8].isdigit() and len(ln.strip()[:8]) == 8)
    # Find the row where the date pattern breaks (annual block or copyright)
    end = start
    for i in range(start, len(lines)):
        head = lines[i].strip()[:8]
        if not (head.isdigit() and len(head) == 8):
            break
        end = i + 1
    df = pd.read_csv(io.StringIO("\n".join(lines[start - 1 : end])),
                     skipinitialspace=True)
    df.columns = ["date"] + [c.strip() for c in df.columns[1:]]
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    for c in df.columns[1:]:
        df[c] = df[c].astype(float) / 100.0  # French reports in percent
    return df

try:
    print("  Downloading FF5 factors from Ken French data library...")
    ff5_text = _fetch_french_zip(FF_URL_5F)
    ff5 = _parse_french_csv(ff5_text)
    print(f"    FF5: {len(ff5):,} rows, columns = {list(ff5.columns)}")

    print("  Downloading momentum factor...")
    mom_text = _fetch_french_zip(FF_URL_MOM)
    mom = _parse_french_csv(mom_text)
    # The momentum file's column name varies (Mom, MOM, Mom )
    mom_col = [c for c in mom.columns if c != "date"][0]
    mom = mom.rename(columns={mom_col: "MOM"})
    print(f"    Momentum: {len(mom):,} rows, columns = {list(mom.columns)}")

    factors = ff5.merge(mom, on="date", how="inner")
    factors = factors.rename(columns={"Mkt-RF": "MktRF"})
    print(f"  Merged factor panel: {len(factors):,} rows, "
          f"{factors['date'].min().date()} → {factors['date'].max().date()}")
except Exception as e:
    print(f"  !! Failed to fetch French factors: {e}")
    print("  Skipping D1.  You can rerun later with manual factor download.")
    factors = None

# Regress long-short returns on FF6 factors -----------------------------------
d1_pass = False
if factors is not None:
    ls_df = ls.rename("ls").reset_index()
    ls_df["date"] = pd.to_datetime(ls_df["date"])
    reg_df = ls_df.merge(factors, on="date", how="inner").dropna()
    print(f"  L-S aligned with factors: {len(reg_df):,} daily obs.")

    if len(reg_df) > 50:
        X = sm.add_constant(reg_df[["MktRF", "SMB", "HML", "RMW", "CMA", "MOM"]])
        y = reg_df["ls"]
        ff6 = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

        alpha_bps = ff6.params["const"] * 1e4
        alpha_t   = ff6.tvalues["const"]
        print(f"\n  FF6 regression results:")
        print(f"    alpha (daily):       {alpha_bps:+.2f} bps")
        print(f"    alpha t-stat (HAC):  {alpha_t:+.3f}")
        print(f"    R-squared:           {ff6.rsquared:.4f}")
        print(f"  Factor loadings:")
        for f in ["MktRF", "SMB", "HML", "RMW", "CMA", "MOM"]:
            b, tv = ff6.params[f], ff6.tvalues[f]
            print(f"    {f:<6} = {b:+.3f}  (t = {tv:+.2f})")

        d1_pass = (alpha_bps >= DIAG_THRESHOLDS["D1_FF6_alpha_bps_min"]
                   and alpha_t >= DIAG_THRESHOLDS["D1_FF6_tstat_min"])

        # Diagnose what factor is absorbing the raw spread ----------------------
        raw_bps = ls.mean() * 1e4
        absorbed_pct = (raw_bps - alpha_bps) / raw_bps * 100 if raw_bps != 0 else np.nan
        print(f"\n  Risk absorption: raw {raw_bps:+.2f} → alpha {alpha_bps:+.2f} "
              f"({absorbed_pct:.1f}% absorbed by factors)")
        if abs(ff6.tvalues.get("MOM", 0)) > 2:
            print(f"  -> Momentum loading is significant — signal partially explained by MOM.")
print(f"\n  D1 result: {'PASS' if d1_pass else 'FAIL'}")
print()


# =============================================================================
# D2 — Sample-period split
# =============================================================================
print("=" * 78)
print(" D2 — Subsample split: retail boom vs surrounding period")
print("=" * 78)

# Boundaries chosen to isolate the retail-frenzy window -----------------------
BOOM_START = pd.Timestamp("2020-03-01")
BOOM_END   = pd.Timestamp("2021-03-31")

def _ls_stats(series: pd.Series, label: str) -> dict:
    n = len(series)
    if n < 30 or series.std() == 0:
        return {"label": label, "n": n, "mean_bps": np.nan,
                "tstat": np.nan, "sharpe": np.nan}
    mean = series.mean()
    se   = series.std() / np.sqrt(n)
    return {
        "label": label, "n": n,
        "mean_bps": mean * 1e4,
        "tstat":    mean / se,
        "sharpe":   mean / series.std() * np.sqrt(252),
    }

ls_df = ls.rename("ls").reset_index()
ls_df["date"] = pd.to_datetime(ls_df["date"])
boom_mask = (ls_df["date"] >= BOOM_START) & (ls_df["date"] <= BOOM_END)
ls_boom = ls_df.loc[boom_mask, "ls"]
ls_quiet = ls_df.loc[~boom_mask, "ls"]

stats_boom  = _ls_stats(ls_boom,  f"Retail boom ({BOOM_START.date()} – {BOOM_END.date()})")
stats_quiet = _ls_stats(ls_quiet, "Surrounding period")
stats_full  = _ls_stats(ls_df["ls"], "Full sample")

print(f"  {'Subsample':<55}{'N':>6}{'Mean (bps)':>14}{'t-stat':>10}{'Sharpe':>10}")
print(f"  {'-' * 95}")
for s in (stats_full, stats_boom, stats_quiet):
    print(f"  {s['label']:<55}{s['n']:>6}{s['mean_bps']:>+14.2f}"
          f"{s['tstat']:>+10.2f}{s['sharpe']:>+10.2f}")

# Decision logic --------------------------------------------------------------
t_boom  = stats_boom["tstat"]
t_quiet = stats_quiet["tstat"]
d2_pass = (not np.isnan(t_boom)) and (not np.isnan(t_quiet)) and \
          (t_boom > DIAG_THRESHOLDS["D2_subsample_tstat_min"]) and \
          (t_quiet > DIAG_THRESHOLDS["D2_subsample_tstat_min"])

# Sharper diagnostic: ratio of t-stats
if not np.isnan(t_boom) and not np.isnan(t_quiet) and t_quiet != 0:
    ratio = t_boom / max(abs(t_quiet), 1e-6)
    print(f"\n  Boom t-stat / Quiet t-stat = {ratio:+.2f}")
    if abs(ratio) > 3:
        print(f"  -> Signal heavily concentrated in retail boom — caution warranted.")
    elif t_quiet < 0:
        print(f"  -> Signal REVERSES outside the boom — likely momentum/sentiment, not anomaly.")
    else:
        print(f"  -> Signal present in both periods — supports a structural mechanism.")
print(f"\n  D2 result: {'PASS' if d2_pass else 'FAIL'}")
print()


# =============================================================================
# D3 — Meme-stock exclusion
# =============================================================================
print("=" * 78)
print(" D3 — Meme-stock exclusion")
print("=" * 78)

# Hand-curated meme list — the most discussed WSB tickers Jan 2021 onward ----
MEME_TICKERS = {
    "GME", "AMC", "BB", "BBBY", "KOSS", "NOK", "EXPR", "NAKD",
    "PLTR", "WISH", "CLOV", "SNDL", "TLRY", "WKHS", "MVIS",
    "RKT", "RIDE", "SPCE", "OCGN", "GEVO",
}

memes_in_universe = sorted([t for t in MEME_TICKERS if t in panel_sort["ticker"].unique()])
print(f"  Meme tickers present in sample ({len(memes_in_universe)}): "
      f"{memes_in_universe}")

panel_nomeme = panel_sort[~panel_sort["ticker"].isin(MEME_TICKERS)].copy()
print(f"  Sample size: full = {len(panel_sort):,}, no-meme = {len(panel_nomeme):,}")
print(f"  Tickers dropped: {panel_sort['ticker'].nunique() - panel_nomeme['ticker'].nunique()}")

# Rebuild deciles WITHOUT meme stocks ----------------------------------------
def assign_decile(s: pd.Series) -> pd.Series:
    try:
        return pd.qcut(s, 10, labels=False, duplicates="drop") + 1
    except ValueError:
        return pd.Series(np.nan, index=s.index)

panel_nomeme["decile_nm"] = (panel_nomeme.groupby("date")["rop_w_lag1"]
                                         .transform(assign_decile))
panel_nomeme = panel_nomeme.dropna(subset=["decile_nm"])
panel_nomeme["decile_nm"] = panel_nomeme["decile_nm"].astype(int)

port_nm = (panel_nomeme.groupby(["date", "decile_nm"])["ret"]
                       .mean().unstack("decile_nm").dropna(how="any"))
ls_nm = port_nm[port_nm.columns.max()] - port_nm[port_nm.columns.min()]

stats_full_panel = _ls_stats(ls,    "Full sample (with memes)")
stats_nomeme     = _ls_stats(ls_nm, "Meme stocks excluded")

print(f"\n  {'Sample':<35}{'N':>6}{'Mean (bps)':>14}{'t-stat':>10}{'Sharpe':>10}")
print(f"  {'-' * 75}")
for s in (stats_full_panel, stats_nomeme):
    print(f"  {s['label']:<35}{s['n']:>6}{s['mean_bps']:>+14.2f}"
          f"{s['tstat']:>+10.2f}{s['sharpe']:>+10.2f}")

t_full   = stats_full_panel["tstat"]
t_nomeme = stats_nomeme["tstat"]
if not np.isnan(t_full) and not np.isnan(t_nomeme) and t_full > 0:
    dropoff_pct = (t_full - t_nomeme) / t_full * 100
    print(f"\n  t-stat dropoff after meme exclusion: {dropoff_pct:+.1f}%")
    d3_pass = dropoff_pct < DIAG_THRESHOLDS["D3_tstat_dropoff_max_pct"]
    if dropoff_pct > 70:
        print(f"  -> Signal is meme-driven — narrative must be reframed.")
    elif dropoff_pct > 40:
        print(f"  -> Meme stocks contribute meaningfully but effect survives.")
    else:
        print(f"  -> Signal is broad-based, not driven by a few meme tickers.")
else:
    d3_pass = False
print(f"\n  D3 result: {'PASS' if d3_pass else 'FAIL'}")
print()


# =============================================================================
# Verdict
# =============================================================================
print("=" * 78)
print(" DIAGNOSTIC VERDICT")
print("=" * 78)
print(f"  D1 (FF6 alpha):           {'PASS' if d1_pass else 'FAIL'}")
print(f"  D2 (subsample stability): {'PASS' if d2_pass else 'FAIL'}")
print(f"  D3 (meme robustness):     {'PASS' if d3_pass else 'FAIL'}")

n_diag_pass = sum([d1_pass, d2_pass, d3_pass])
print()
if n_diag_pass == 3:
    print("  >>> 3/3 PASS — signal is robust.  Commit to the full paper.")
    print("      Next step: pull OptionMetrics + CRSP from WRDS, build short-dated ROP.")
elif n_diag_pass == 2:
    print("  >>> 2/3 PASS — signal is broadly real but has one caveat to address.")
    print("      Identify which test failed and design a refinement before scaling up.")
elif n_diag_pass == 1:
    print("  >>> 1/3 PASS — signal is fragile.  Reframe scope before investing more time.")
    print("      The paper may need to be about WHEN the signal works, not whether.")
else:
    print("  >>> 0/3 PASS — raw spread is mostly known factors / meme / boom artifact.")
    print("      Reconsider whether to proceed without short-dated and 0DTE-era data.")
print()

# Save the diagnostic results so they're recoverable later --------------------
diag_summary = pd.DataFrame({
    "test":   ["D1", "D2", "D3"],
    "result": [d1_pass, d2_pass, d3_pass],
    "detail": [
        f"FF6 alpha = {alpha_bps:+.2f} bps, t = {alpha_t:+.3f}"
            if factors is not None else "Factors not available",
        f"Boom t = {t_boom:+.2f}, Quiet t = {t_quiet:+.2f}",
        f"Full t = {t_full:+.2f}, No-meme t = {t_nomeme:+.2f}",
    ],
})
diag_summary.to_csv(OUTPUT / "diagnostic_summary.csv", index=False)
print(f"  Saved: {OUTPUT / 'diagnostic_summary.csv'}")
print("=" * 78)

