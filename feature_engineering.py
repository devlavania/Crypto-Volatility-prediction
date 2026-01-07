import pandas as pd

def create_features(df):

    df['return'] = df.groupby('crypto_name')['close'].pct_change()

    df['volatility_14'] = (
        df.groupby('crypto_name')['return']
        .rolling(14)
        .std()
        .reset_index(level=0, drop=True)
    )

    df['ma_7'] = (
        df.groupby('crypto_name')['close']
        .rolling(7)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df['ma_21'] = (
        df.groupby('crypto_name')['close']
        .rolling(21)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df['liquidity_ratio'] = df['volume'] / df['marketCap']

    rolling_mean = (
        df.groupby('crypto_name')['close']
        .rolling(20)
        .mean()
        .reset_index(level=0, drop=True)
    )

    rolling_std = (
        df.groupby('crypto_name')['close']
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
    )

    df['bb_width'] = (2 * rolling_std) / rolling_mean

    df.dropna(inplace=True)

    return df
