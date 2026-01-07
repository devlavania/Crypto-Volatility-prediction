import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(X_test, y_test):

    model = joblib.load("models/crypto_volatility_model.pkl")

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2 Score:", r2)
