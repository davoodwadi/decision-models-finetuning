"""CLI entrypoint for exporting covid_US_long.parquet (`python -m data.covid`)."""

from data.covid import COVID_PARQUET_PATH, export_covid_parquet

if __name__ == "__main__":
    out_df = export_covid_parquet()
    print(
        f"Wrote {COVID_PARQUET_PATH} ({len(out_df):,} rows, "
        f"{out_df['CASEID'].nunique():,} respondents, "
        f"{out_df['qid'].nunique():,} unique qids)"
    )
