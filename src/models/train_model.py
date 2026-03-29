import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib


def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def split_data(df, split_date="2017-01-01"):
    train = df[df['date'] < split_date]
    test = df[df['date'] >= split_date]
    return train, test


def get_features():
    return [
        'store', 'item',
        'day_of_week', 'month', 'year', 'day_of_month',
        'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_56', 'lag_84',
        'rolling_mean_7', 'rolling_std_7',
        'rolling_mean_14', 'rolling_mean_28',
        'rolling_max_7'
    ]


def train_model(train, features, target):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(train[features], train[target])
    return model


def evaluate_model(model, test, features, target):
    preds = model.predict(test[features])
    mae = mean_absolute_error(test[target], preds)

    print(f"MAE: {mae:.4f}")
    return preds


def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def main():
    data_path = "data/processed/feature_data.csv"
    model_path = "models/xgb_model.pkl"

    df = load_data(data_path)
    train, test = split_data(df)

    FEATURES = get_features()
    TARGET = "sales"

    model = train_model(train, FEATURES, TARGET)
    preds = evaluate_model(model, test, FEATURES, TARGET)

    save_model(model, model_path)


if __name__ == "__main__":
    main()