# Project 1 — Temperature Anomaly Detection

## Part A — Speaker script (~10 min, ~1 min/slide)

**1. Title.** Hi everyone. This is my project on temperature anomaly detection. The idea is simple: take a live stream of weather measurements and automatically spot the temperature values that look abnormal. I'll walk through the data, the pipeline, the methods, and what I found.

**2. Goal.** The goal has two parts. First, flag unusual temperatures in a stream of readings — sudden jumps or values that don't fit the recent pattern. Second, build a small end-to-end pipeline that collects the data, stores it, and gives a clear normal-versus-anomalous verdict.

**3. Data.** The data comes from a shared weather REST API. I authenticate with a token and pull measurements for station GDN_01 in Gdańsk. Each record has temperature, humidity, pressure, wind, rain and cloud cover. Readings arrive every ten minutes, so one pull gives about 500 records — roughly three and a half days of history.

**4. Pipeline.** Here's the pipeline. I pull from the API, store the raw data as a Bronze layer — a local CSV and a copy on AWS S3. Then I clean it: parse timestamps, drop duplicates, range-check the values. The detection step runs five methods and votes. Finally I produce charts and an anomaly report.

**5. Key insight.** Now the important thing about this data. The temperature varies with a standard deviation of about 2.2 degrees over the whole window. But the typical change between two consecutive readings is about 2.6 degrees — even bigger than that. So the step-to-step noise is as large as the entire signal's spread. That means the raw ten-minute stream is dominated by noise — there are no clean outliers to find.

**6. Methods.** I used five detectors. A global Z-score compares each value to the overall mean. A rolling Z-score compares it to its recent neighbours — that's the best one for a stream. The IQR rule flags values outside the usual range. A rate-of-change rule catches sudden jumps. And an Isolation Forest is a simple machine-learning model. I combine them by majority vote — a point is flagged when at least two methods agree.

**7. Results.** The results match the insight. The three statistical methods find essentially zero anomalies — and on noisy raw data, that's the correct answer, not a bug. The jump rule and the Isolation Forest surface a handful of candidates, and I keep the ones at least two methods agree on. You can see them in red on Figure 1.

**8. Raw vs aggregated.** So how do you actually detect anything here? You aggregate. Averaging the readings into hourly values cancels the noise — the standard deviation drops from about 2.2 to 0.8 degrees, roughly the square root of six samples per hour. The real daily trend appears. So the answer to "raw or aggregated?" is clearly: aggregate first. That's Figure 2.

**9. Sensitivity.** To measure how good the detector is, I injected twelve known spikes into the data and checked how many it recovered. As I lower the threshold k, recall goes up — I catch more spikes — but false positives creep in. At k = 2 I catch half of them; tighten it and I catch fewer, with almost no false alarms. Figure 3 shows that trade-off.

**10. Takeaways.** To wrap up: the raw stream is noise-dominated, so "nothing found" on raw data is the right result. The trick is to aggregate first and use rolling context for streams. The threshold sets the trade-off between catching anomalies and false alarms. Everything is reproducible, with the raw data stored on AWS S3. Thanks — happy to take questions.

---

## Part B — What this project teaches (recap)

- **Build an end-to-end data pipeline:** ingest from an API → store in layers (Bronze on S3) → clean → analyse → visualise. This is the standard shape of a real data project.
- **Characterise the data before modelling.** Here the signal-to-noise ratio at the native 10-minute resolution is about 1, so hunting for point outliers on raw data is futile — and recognising that *is* the result.
- **Use several methods and combine them.** Statistical rules (Z-score, IQR), a domain rule (sudden jump), and an ML model (Isolation Forest), merged by majority vote, are more robust than any single detector. For a *stream*, local/rolling methods beat global ones.
- **Aggregation is denoising.** Averaging N samples cuts random noise by √N. Hourly means turn an unreadable raw stream into a clean signal.
- **Validate without ground truth.** When the data has no labelled anomalies, inject known spikes and measure recall vs false positives. The detection threshold is a tunable sensitivity dial.

## Part C — How AWS was used

- **Environment:** the whole project runs in the **AWS Academy Learner Lab** (a sandboxed AWS account, ~$50 budget, 4-hour sessions) — the platform required for the course.
- **Data source:** the shared weather API is itself hosted on AWS (**API Gateway**); my collector calls it over HTTPS with a bearer token.
- **Storage — Amazon S3 (the main AWS service I used):** the collector writes the raw measurements to **`s3://weather-anomaly-s212129/bronze/weather/`** using the **boto3** SDK. This is the cloud "Storage Layer / Bronze layer" in the architecture diagram — the raw, immutable copy of the data.
- **Credentials:** temporary Learner Lab session credentials (from *AWS Details*) placed in `~/.aws/credentials`, which boto3 picks up automatically.
- **Kept it budget-light on purpose:** compute (collection + analysis) runs locally in Python, and only S3 is used in the cloud, so the lab budget is barely touched. The same pipeline could be extended to SageMaker (notebooks), Glue/Athena (SQL on S3), or Lambda (scheduled collection) if more of AWS were required.
