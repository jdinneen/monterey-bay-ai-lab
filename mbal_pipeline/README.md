# Monterey Bay AI Lab Pipeline

Production-oriented validation and partitioning for long-format MBAL historical parquet outputs.

This tool is intentionally separate from the existing fetch and model scripts. It reads one or more harmonized parquet files, validates the data, writes a JSON report, and can write curated parquet partitioned by `station/year/month`.

## Usage

Validate only:

```powershell
python -m mbal_pipeline.cli ..\mbal_history\opendap\m1_history.parquet --report reports\m1_validation.json --overwrite
```

Validate and write partitioned parquet:

```powershell
python -m mbal_pipeline.cli ..\mbal_history\opendap\m1_history.parquet ..\mbal_history\opendap\m2_history.parquet --report reports\history_validation.json --output-dir curated_history --overwrite
```

If validation errors are present, partition writing is blocked. Use `--allow-errors` only for forensic/debug output.

## Checks

- Required long-format schema and UTC timestamp handling.
- Physical ranges for ocean, meteorological, position, depth, wind, and current fields.
- OceanSITES QC flags, including non-null values under bad QC flags.
- Duplicate `station/time/depth_m` keys.
- Time/depth coverage summaries and large cadence gaps per station/depth.
- Current velocity unit checks, including suspected cm/s values and `current_speed = sqrt(u^2 + v^2)`.

## Output Layout

Partitioned output is written as:

```text
curated_history/
  station=M1/
    year=2025/
      month=7/
        *.parquet
```

The parquet files are sorted by `station`, `time`, and `depth_m`, compressed with Zstandard, and suitable for downstream DuckDB, PyArrow, pandas, or model-training reads.

