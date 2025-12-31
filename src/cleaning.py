
import pandas as pd


def clean_data(df, cols_to_clip=None, time_col=None):
    """
    Clean dataset by clipping invalid values and interpolating missing numeric data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    cols_to_clip : list, optional
        List of numeric columns to clip at lower=0
    time_col : str, optional
        Name of datetime column to use as index for time interpolation

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe
    """
    df = df.copy()

    # Clip negative values
    if cols_to_clip:
        for col in cols_to_clip:
            df[col] = df[col].clip(lower=0)

    # Time-based interpolation
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df.set_index(time_col, inplace=True)
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method="time")
        df.reset_index(inplace=True)

    # Drop remaining NaNs
    df.dropna(inplace=True)

    return df


# --- Outlier handling functions ---

def outlier_thresholds(df, col_name, q1=0.25, q3=0.75):
    """
    Calculate lower and upper limits for outliers based on IQR.

    Parameters
    ----------
    df : pd.DataFrame
    col_name : str
        Column name
    q1 : float
        Lower quantile (default 0.25)
    q3 : float
        Upper quantile (default 0.75)

    Returns
    -------
    low_limit, up_limit : tuple
        Lower and upper threshold for the column
    """
    quartile1 = df[col_name].quantile(q1)
    quartile3 = df[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * iqr
    up_limit = quartile3 + 1.5 * iqr
    return low_limit, up_limit


def replace_with_thresholds(df, col_name):
    """
    Replace values outside the thresholds with the threshold values.

    Parameters
    ----------
    df : pd.DataFrame
    col_name : str
        Column name

    Returns
    -------
    None (modifies dataframe in place)
    """
    low, up = outlier_thresholds(df, col_name)
    df.loc[df[col_name] < low, col_name] = low
    df.loc[df[col_name] > up, col_name] = up
