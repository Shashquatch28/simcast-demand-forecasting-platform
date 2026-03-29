import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def sort_data(df):
    df = df.sort_values(by=['store', 'item', 'date'])
    df = df.reset_index(drop=True)
    return df


def create_time_features(df):
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_month'] = df['date'].dt.day
    return df


def create_lag_features(df):
    for lag in [1, 7, 14, 28, 56, 84]:
        df[f'lag_{lag}'] = df.groupby(['store', 'item'])['sales'].shift(lag)
    return df


def create_rolling_features(df):
    df['rolling_mean_7'] = df.groupby(['store', 'item'])['sales'].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )

    df['rolling_std_7'] = df.groupby(['store', 'item'])['sales'].transform(
        lambda x: x.shift(1).rolling(7).std()
    )

    df['rolling_mean_14'] = df.groupby(['store', 'item'])['sales'].transform(
        lambda x: x.shift(1).rolling(14).mean()
    )

    df['rolling_mean_28'] = df.groupby(['store', 'item'])['sales'].transform(
        lambda x: x.shift(1).rolling(28).mean()
    )

    df['rolling_max_7'] = df.groupby(['store', 'item'])['sales'].transform(
        lambda x: x.shift(1).rolling(7).max()
    )

    return df


def clean_data(df):
    df = df.dropna()
    return df


def save_data(df, path):
    df.to_csv(path, index=False)


def main():
    input_path = "data/raw/train.csv"
    output_path = "data/processed/feature_data.csv"

    df = load_data(input_path)
    df = sort_data(df)
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = clean_data(df)

    save_data(df, output_path)

    print("✅ Feature engineering complete. Saved to:", output_path)


if __name__ == "__main__":
    main()