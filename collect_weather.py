"""Project 1 - Step 1: Data Collector (Bronze layer).

Pulls weather measurements from the shared REST API and stores them as a CSV.
Optionally mirrors the file to AWS S3.

Examples
--------
    # Pull the latest ~500 records for the default station (GDN_01)
    python collect_weather.py

    # A different station, appending to existing history (de-duplicated)
    python collect_weather.py --station GDN_02

    # Also upload the raw CSV to S3 (Learner Lab Bronze layer)
    python collect_weather.py --s3-bucket weather-anomaly-s212129
"""
import argparse

import pandas as pd
import requests

import config


def fetch_batch(station: str, limit: int) -> pd.DataFrame:
    """Call GET /weather/batch and return the records as a DataFrame."""
    url = f"{config.BASE_URL}/weather/batch"
    params = {"station_id": station, "limit": limit}
    resp = requests.get(url, params=params, headers=config.auth_header(), timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    records = payload.get("records", payload if isinstance(payload, list) else [])
    df = pd.DataFrame(records)
    if df.empty:
        raise SystemExit("No records returned. Check the station id and token.")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Light validation: parse timestamps, drop empties/duplicates, sort."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "temperature"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    return df.reset_index(drop=True)


def save_local(df: pd.DataFrame, append: bool) -> pd.DataFrame:
    """Write to the raw CSV, optionally merging with previously collected data."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if append and config.RAW_CSV.exists():
        old = pd.read_csv(config.RAW_CSV)
        old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True, errors="coerce")
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
        df = df.reset_index(drop=True)
    df.to_csv(config.RAW_CSV, index=False)
    return df


def upload_s3(bucket: str, prefix: str) -> None:
    """Mirror the raw CSV to S3 (boto3 imported lazily so local runs need no AWS)."""
    import boto3
    key = f"{prefix}/{config.RAW_CSV.name}"
    boto3.client("s3").upload_file(str(config.RAW_CSV), bucket, key)
    print(f"Uploaded to s3://{bucket}/{key}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect weather data (Bronze layer).")
    parser.add_argument("--station", default=config.STATION)
    parser.add_argument("--limit", type=int, default=config.MAX_BATCH,
                        help="Records to request (API caps at ~500).")
    parser.add_argument("--no-append", action="store_true",
                        help="Overwrite instead of appending to existing history.")
    parser.add_argument("--s3-bucket", default=config.S3_BUCKET)
    args = parser.parse_args()

    raw = fetch_batch(args.station, args.limit)
    cleaned = clean(raw)
    stored = save_local(cleaned, append=not args.no_append)

    span = stored["timestamp"].max() - stored["timestamp"].min()
    print(f"Fetched {len(cleaned)} records; stored total {len(stored)} rows.")
    print(f"Time span: {span}  "
          f"({stored['timestamp'].min()} -> {stored['timestamp'].max()})")
    print(f"Saved to {config.RAW_CSV}")

    if args.s3_bucket:
        upload_s3(args.s3_bucket, config.S3_PREFIX)


if __name__ == "__main__":
    main()
