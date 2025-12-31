import pandas as pd

def load_data(filepath):
    """
    Load CSV file with semicolon separator and decimal point handling,
    convert timestamp column to datetime, sort by timestamp.
    """
    # Read CSV with proper separator and decimal point
    df = pd.read_csv(
        filepath,
        sep=';',  # Semicolon separator
    )

    # Convert timestamp column if exists
    if "TIME" in df.columns:
        df["TIME"] = pd.to_datetime(df["TIME"], errors='coerce')
        df.sort_values("TIME", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df

