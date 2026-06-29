# %% [markdown]
# # Project 1 - Temperature Anomaly Detection
#
# **Goal:** detect unusual temperature behaviour in a stream of weather
# measurements collected from the shared REST API.
#
# This file:
# 1. loads the raw data (Bronze layer) produced by `collect_weather.py`,
# 2. cleans and validates it,
# 3. applies five complementary anomaly-detection methods,
# 4. produces charts and a list of detected anomalies,
# 5. quantifies sensitivity / false positives via a controlled injection test.
#
# Run as a script (`python anomaly_detection.py`) to regenerate every chart in
# `outputs/`, or open it as a notebook (see README) for an inline report.

# %%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

import config

config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Detection parameters (tune these to change sensitivity) ----------------
ROLL_WINDOW = 18          # rolling window (18 x 10 min = 3 h of context)
ROLL_K = 3.0              # rolling z-score threshold
GLOBAL_K = 3.0            # global z-score threshold
JUMP_QUANTILE = 0.99      # 'sudden jump' = top 1% of step-to-step changes.
                          # Data-driven on purpose: the raw 10-min stream is
                          # very noisy, so a fixed deg-C threshold is meaningless.
ISO_CONTAMINATION = 0.02  # expected anomaly fraction for the ML model

# Read the data from "local" (CSV on disk) or "s3" (the Bronze copy in your bucket).
# Override via env: set WEATHER_DATA_SOURCE=s3 and WEATHER_S3_BUCKET=<bucket>.
DATA_SOURCE = os.getenv("WEATHER_DATA_SOURCE", "local")

# %% [markdown]
# ## 1. Load & clean (Bronze -> Silver)
#
# The API data is already complete, so cleaning focuses on parsing timestamps,
# removing duplicates, sorting chronologically, and range-validating each
# physical quantity.

# %%
def load_data() -> pd.DataFrame:
    """Load the data from the local CSV, or straight from S3 (DATA_SOURCE='s3').

    Falls back to a live API pull if no local file exists, so the script always runs.
    """
    if DATA_SOURCE == "s3":
        import io, boto3
        if not config.S3_BUCKET:
            raise SystemExit("Set WEATHER_S3_BUCKET to read from S3 (or edit config.S3_BUCKET).")
        obj = boto3.client("s3").get_object(Bucket=config.S3_BUCKET, Key=config.S3_KEY)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        print(f"Loaded {len(df)} rows from s3://{config.S3_BUCKET}/{config.S3_KEY}")
    elif config.RAW_CSV.exists():
        df = pd.read_csv(config.RAW_CSV)
    else:
        import requests
        url = f"{config.BASE_URL}/weather/batch"
        params = {"station_id": config.STATION, "limit": config.MAX_BATCH}
        r = requests.get(url, params=params, headers=config.auth_header(), timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json()["records"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = (df.dropna(subset=["timestamp", "temperature"])
            .drop_duplicates(subset="timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True))
    return df


df = load_data()

# Range validation (physically plausible bounds) -----------------------------
valid = (
    df["temperature"].between(-50, 60)
    & df["humidity"].between(0, 100)
    & df["pressure"].between(870, 1085)
    & (df["wind_speed"] >= 0)
)
print(f"Loaded {len(df)} rows for station {config.STATION}.")
print(f"Out-of-range rows: {int((~valid).sum())}")
print(f"Period: {df['timestamp'].min()} -> {df['timestamp'].max()}")

df = df.set_index("timestamp")

# %% [markdown]
# ## 2. Detection methods
#
# | Method | Catches | Idea |
# |---|---|---|
# | Global Z-score | global outliers | value far from the overall mean |
# | Rolling Z-score | local deviations | value far from the **recent** mean (best for streams) |
# | IQR | distribution outliers | value outside `Q1 - 1.5*IQR` .. `Q3 + 1.5*IQR` |
# | Rate-of-change | sudden jumps | large change between consecutive readings |
# | Isolation Forest | multivariate | simple ML model on (temperature, rate) |

# %%
t = df["temperature"]

# Global z-score
df["global_z"] = (t - t.mean()) / t.std()
df["globalz_anom"] = df["global_z"].abs() > GLOBAL_K

# Rolling z-score (local context)
min_p = max(3, ROLL_WINDOW // 2)
df["roll_mean"] = t.rolling(ROLL_WINDOW, min_periods=min_p).mean()
df["roll_std"] = t.rolling(ROLL_WINDOW, min_periods=min_p).std()
df["roll_z"] = (t - df["roll_mean"]) / df["roll_std"]
df["rollz_anom"] = df["roll_z"].abs() > ROLL_K

# IQR
q1, q3 = t.quantile(0.25), t.quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df["iqr_anom"] = (t < lo) | (t > hi)

# Rate of change (sudden jump). Threshold is data-driven because the raw
# 10-minute stream is noise-dominated (see the raw-vs-hourly section below).
df["rate"] = t.diff()
JUMP_C = round(df["rate"].abs().quantile(JUMP_QUANTILE), 2)
df["jump_anom"] = df["rate"].abs() > JUMP_C
print(f"Data-driven jump threshold (p{int(JUMP_QUANTILE * 100)} of |dT|): {JUMP_C} C")

# Isolation Forest (simple ML model)
features = df[["temperature", "rate"]].fillna(0.0)
iso = IsolationForest(contamination=ISO_CONTAMINATION, random_state=42)
df["iso_anom"] = iso.fit_predict(features) == -1

METHODS = ["globalz_anom", "rollz_anom", "iqr_anom", "jump_anom", "iso_anom"]
df["votes"] = df[METHODS].sum(axis=1)
df["anomaly_any"] = df["votes"] > 0

summary = {m: int(df[m].sum()) for m in METHODS}
print("Anomalies flagged per method:", summary)
print(f"Flagged by at least one method: {int(df['anomaly_any'].sum())}")
print(f"Flagged by >=2 methods (high confidence): {int((df['votes'] >= 2).sum())}")

# %% [markdown]
# ## 3. Charts

# %%
def save(fig, name):
    path = config.OUTPUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print("saved", path)


# 3a. Temperature with rolling-z anomalies + confidence band
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(df.index, df["temperature"], color="#33bb66", lw=1, label="temperature")
ax.plot(df.index, df["roll_mean"], color="#0088cc", lw=1, ls="--",
        label=f"rolling mean ({ROLL_WINDOW})")
ax.fill_between(df.index,
                df["roll_mean"] - ROLL_K * df["roll_std"],
                df["roll_mean"] + ROLL_K * df["roll_std"],
                color="#0088cc", alpha=0.12, label=f"+/-{ROLL_K} sigma band")
# Mark the CONSENSUS anomalies (not just rolling-z), so the figure shows what
# was actually flagged: candidates (1 method) and high-confidence (>=2 methods).
cand = df[df["votes"] == 1]
conf = df[df["votes"] >= 2]
ax.scatter(cand.index, cand["temperature"], facecolors="none", edgecolors="darkorange",
           s=60, lw=1.6, zorder=5, label="candidate (1 method)")
ax.scatter(conf.index, conf["temperature"], color="crimson", s=80, zorder=6,
           label="anomaly (>=2 methods)")
ax.set_title(f"Temperature anomalies - station {config.STATION}")
ax.set_xlabel("time"); ax.set_ylabel("temperature (C)"); ax.legend(loc="best")
save(fig, "01_temperature_anomalies.png")

# 3b. Consensus view: colour each point by how many methods agree
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(df.index, df["temperature"], color="0.75", lw=1, zorder=1)
sc = ax.scatter(df.index, df["temperature"], c=df["votes"], cmap="YlOrRd",
                s=18, zorder=2)
fig.colorbar(sc, ax=ax, label="methods in agreement")
ax.set_title("Consensus anomaly score (votes across 5 methods)")
ax.set_xlabel("time"); ax.set_ylabel("temperature (C)")
save(fig, "02_consensus_votes.png")

# 3c. Distribution with IQR fences
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(t, bins=30, color="#66aabb", alpha=0.85)
ax.axvline(lo, color="crimson", ls="--", label="IQR fence")
ax.axvline(hi, color="crimson", ls="--")
ax.set_title("Temperature distribution & IQR fences")
ax.set_xlabel("temperature (C)"); ax.set_ylabel("count"); ax.legend()
save(fig, "03_distribution_iqr.png")

# 3d. Rate of change with jump anomalies
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(df.index, df["rate"], color="#999999", lw=1)
ax.axhline(JUMP_C, color="crimson", ls="--", label=f"+/-{JUMP_C} C jump")
ax.axhline(-JUMP_C, color="crimson", ls="--")
jp = df[df["jump_anom"]]
ax.scatter(jp.index, jp["rate"], color="crimson", s=30, zorder=5)
ax.set_title("Temperature rate of change between readings")
ax.set_xlabel("time"); ax.set_ylabel("delta temperature (C)"); ax.legend()
save(fig, "04_rate_of_change.png")

# 3e. Method comparison
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(list(summary.keys()), list(summary.values()), color="#cc5555")
ax.set_title("Anomalies detected per method")
ax.set_ylabel("count"); plt.xticks(rotation=20)
save(fig, "05_method_comparison.png")

plt.show()

# %% [markdown]
# ## 3b. Raw vs hourly-aggregated data
#
# The raw 10-minute stream is noise-dominated: the typical change between two
# consecutive readings is about as large as the standard deviation of the whole
# series, which is why the z-score / IQR methods find almost no organic outliers.
# Aggregating to **hourly means** averages out that measurement noise and exposes
# the underlying temperature signal, where anomalies become detectable.
# This answers the brief's question *"raw or aggregated?"* -> aggregated.

# %%
hourly = t.resample("1h").mean().dropna()
h_roll = hourly.rolling(6, min_periods=3)
h_z = (hourly - h_roll.mean()) / h_roll.std()
print(f"Raw:    std={t.std():.2f} C, typical step |dT|={t.diff().abs().median():.2f} C")
print(f"Hourly: std={hourly.std():.2f} C, typical step |dT|={hourly.diff().abs().median():.2f} C")
print(f"Hourly rolling-z anomalies (|z|>{ROLL_K}): {int((h_z.abs() > ROLL_K).sum())}")

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
axes[0].plot(df.index, df["temperature"], color="#bbbbbb", lw=0.8)
axes[0].set_title("Raw 10-minute temperature (noise-dominated)")
axes[0].set_ylabel("C")
axes[1].plot(hourly.index, hourly.values, color="#0088cc", lw=1.5)
h_an = hourly[h_z.abs() > ROLL_K]
axes[1].scatter(h_an.index, h_an.values, color="crimson", s=40, zorder=5)
axes[1].set_title("Hourly-averaged temperature (the real signal)")
axes[1].set_ylabel("C"); axes[1].set_xlabel("time")
save(fig, "06_raw_vs_hourly.png")
plt.show()

# %% [markdown]
# ## 4. Anomaly report (the deliverable list)

# %%
anoms = (df[df["anomaly_any"]]
         .reset_index()[["timestamp", "temperature", "rate", "roll_z", "votes"] + METHODS]
         .sort_values("votes", ascending=False))
report_path = config.OUTPUT_DIR / "anomalies.csv"
anoms.to_csv(report_path, index=False)
print(f"{len(anoms)} anomalies written to {report_path}")

# Largest sudden changes ('when did the most unusual changes occur?')
biggest = df["rate"].abs().sort_values(ascending=False).head(5)
print("\nLargest temperature jumps (deg C between readings):")
print(biggest)
anoms.head(10)

# %% [markdown]
# ## 5. Sensitivity & false positives (controlled validation)
#
# The live stream rarely contains *labelled* anomalies, so we validate the
# detector by **injecting known spikes** into a copy of the series and measuring
# how many we recover (recall) and how many clean points are wrongly flagged
# (false-positive rate). This answers the brief's questions *"How sensitive is
# the method?"* and *"How many false positives?"*.

# %%
def evaluate(k: float, n_inject: int = 12, seed: int = 42):
    rng = np.random.default_rng(seed)
    base = df["temperature"].to_numpy(dtype=float)
    truth = np.zeros(len(base), dtype=bool)
    idx = rng.choice(np.arange(ROLL_WINDOW, len(base)), size=n_inject, replace=False)
    spiked = base.copy()
    spiked[idx] += rng.choice([-1, 1], n_inject) * rng.uniform(4, 8, n_inject)
    truth[idx] = True

    s = pd.Series(spiked)
    rm = s.rolling(ROLL_WINDOW, min_periods=min_p).mean()
    rs = s.rolling(ROLL_WINDOW, min_periods=min_p).std()
    flagged = ((s - rm) / rs).abs() > k

    tp = int((flagged.to_numpy() & truth).sum())
    fp = int((flagged.to_numpy() & ~truth).sum())
    recall = tp / n_inject
    fp_rate = fp / int((~truth).sum())
    return recall, fp_rate


print(f"{'k':>4} {'recall':>8} {'false-pos rate':>16}")
for k in (2.0, 2.5, 3.0, 3.5, 4.0):
    rec, fpr = evaluate(k)
    print(f"{k:>4} {rec:>8.2f} {fpr:>16.4f}")

# %% [markdown]
# ## 6. Conclusions
#
# - **The raw 10-minute stream is noise-dominated.** The typical change between
#   two consecutive readings (~2 C) is about the same size as the standard
#   deviation of the whole series (~2 C), so the z-score, rolling-z and IQR
#   methods correctly report almost **no organic point anomalies** — on this
#   data, "no statistical outliers" is the right answer, not a bug.
# - **Detection works after aggregation.** Hourly averaging cuts the noise by
#   ~2.4x (sqrt of 6 samples per hour) and exposes the real temperature signal
#   (section 3b), so anomalies and trend-breaks become detectable. Answer to the
#   brief's *raw vs aggregated* question: **aggregated**.
# - **Validated sensitivity (section 5).** With injected spikes, the rolling
#   z-score recovers the larger anomalies at k=3 with a low false-positive rate;
#   lowering k raises recall but also false positives.
# - **Best method for a stream:** the rolling z-score, because it compares each
#   reading to its *recent* context rather than to the global mean.
#
# *Add your own numbers from `outputs/anomalies.csv` and the printed summaries
# above (e.g. the exact timestamps of the largest jumps).*
