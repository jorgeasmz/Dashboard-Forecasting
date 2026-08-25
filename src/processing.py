import pandas as pd

from src.config import DATE_COLUMN_RAW, SALES_COLUMN_RAW


def prepare_for_prophet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the DataFrame for Prophet.
    Prophet strictly requires columns named 'ds' (date) and 'y' (target metric).

    Raises:
        ValueError: if the expected raw columns are missing, instead of letting
            a downstream KeyError surface with no context.
    """
    missing = [c for c in (DATE_COLUMN_RAW, SALES_COLUMN_RAW) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected '{DATE_COLUMN_RAW}' and '{SALES_COLUMN_RAW}'."
        )

    # Create a copy to avoid SettingWithCopy warnings on the original DF
    prophet_df = df.copy()

    # Rename columns
    prophet_df = prophet_df.rename(columns={
        DATE_COLUMN_RAW: 'ds',
        SALES_COLUMN_RAW: 'y'
    })

    # Ensure sorting
    prophet_df = prophet_df.sort_values(by='ds')

    return prophet_df
