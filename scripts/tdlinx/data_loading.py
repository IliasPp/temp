import polars as pl

from ..config import CLEAN_2023_PATH, DATA_PATH


def load_data() -> dict[str, pl.DataFrame]:
    sheets: dict[str, pl.DataFrame] = pl.read_excel(DATA_PATH, sheet_id=0)

    # 2023 is loaded from a separately exported clean file (exported via R),
    # because the original sheet contains a copyright disclaimer row that
    # causes Polars to infer Utf8 for all numeric columns at read time.
    sheets["2023"] = pl.read_excel(CLEAN_2023_PATH)

    processed = {}
    for year, df in sheets.items():
        if "Unternehmensgruppe" in df.columns:
            df = df.drop("Unternehmensgruppe")
        if "Gemeindekennziffer" in df.columns:
            df = df.with_columns(
                pl.col("Gemeindekennziffer")
                .cast(pl.Utf8)
                .str.zfill(8)
                .str.slice(0, 5)
                .alias("AGS5")
            )
        processed[year] = df

    return processed
