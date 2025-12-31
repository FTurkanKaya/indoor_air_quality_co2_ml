import pandas as pd

def clean_data(df, cols_to_clip=None):
    """
    Clean negative or invalid values and interpolate missing data
    """
    df = df.copy()
    if cols_to_clip:
        for col in cols_to_clip:
            df[col] = df[col].clip(lower=0)
    df.interpolate(method='time', inplace=True)
    df.dropna(inplace=True)
    return df


# Outlier handling
def outlier_thresholds(df, col_name, q1=0.25, q3=0.75):
    quartile1 = df[col_name].quantile(q1)
    quartile3 = df[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * iqr
    up_limit = quartile3 + 1.5 * iqr
    return low_limit, up_limit

def replace_with_thresholds(df, col_name):
    low, up = outlier_thresholds(df, col_name)
    df.loc[df[col_name] < low, col_name] = low
    df.loc[df[col_name] > up, col_name] = up

