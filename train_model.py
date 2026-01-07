import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def train_model(df):

    features = [
        'open','high','low','close',
        'volume','marketCap',
        'ma_7','ma_21',
        'liquidity_ratio','bb_width'
    ]

    X = df[features]
    y = df['volatility_14']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "models/crypto_volatility_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    return X_test, y_test
