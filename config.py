"""Central configuration for Project 1 - Temperature Anomaly Detection.

Every value can be overridden with an environment variable, so you never have
to hard-code machine-specific paths or your S3 bucket into the analysis code.
"""
import os
from pathlib import Path

# --- Weather REST API (provided by the lecturer) ---------------------------
BASE_URL = os.getenv(
    "WEATHER_BASE_URL",
    "https://e6uw49pbah.execute-api.us-east-1.amazonaws.com/dev",
)
TOKEN = os.getenv("WEATHER_TOKEN", "STUDENT_TOKEN_2026")
STATION = os.getenv("WEATHER_STATION", "GDN_01")  # GDN_01, GDN_02, GDY_01, SOP_01

# The /weather/batch endpoint returns at most ~500 records (~3.5 days at a
# 10-minute sampling interval). To build a longer history, run the collector
# repeatedly over several days in append mode (it de-duplicates by timestamp).
MAX_BATCH = 500

# --- Local storage ----------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw"   # Bronze layer (raw data)
OUTPUT_DIR = PROJECT_DIR / "outputs"      # charts + anomaly report
RAW_CSV = DATA_DIR / f"weather_{STATION}.csv"

# --- Optional AWS S3 (Bronze layer in the cloud) ---------------------------
# Leave empty to stay fully local. Set it to mirror the raw CSV to
# s3://<bucket>/<prefix>/.
S3_BUCKET = os.getenv("WEATHER_S3_BUCKET", "")
S3_PREFIX = os.getenv("WEATHER_S3_PREFIX", "bronze/weather")
S3_KEY = f"{S3_PREFIX}/{RAW_CSV.name}"  # full object key, e.g. bronze/weather/weather_GDN_01.csv


def auth_header() -> dict:
    """HTTP header carrying the bearer token required by the API."""
    return {"Authorization": f"Bearer {TOKEN}"}
