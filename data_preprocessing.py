import pandas as pd

def load_and_preprocess_data(path):
    df = pd.read_csv(path)

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(['crypto_name', 'date'], inplace=True)

    df.fillna(method='ffill', inplace=True)
    df.drop_duplicates(inplace=True)

    return df
