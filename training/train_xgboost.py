import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error


# 🔥 CHANGE THIS EACH RUN
TASK = "cav"   # "nav" or "cav"


def train():

    # Load data
    X = np.load(f"data/features/{TASK}_X.npy")
    y = np.load(f"data/features/{TASK}_y.npy")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔥 STRONGER MODEL (improved)
    model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.5,
        random_state=42,
        n_jobs=-1
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_val)

    # Metrics
    r2 = r2_score(y_val, preds)
    rmse = root_mean_squared_error(y_val, preds)

    print("\n==============================")
    print(f"{TASK.upper()} RESULTS")
    print("==============================")
    print(f"R²   : {r2:.4f}")
    print(f"RMSE : {rmse:.4f}")

    # Save model
    model.save_model(f"prediction_backend/models/saved_models/xgb_{TASK}.json")
    print(f"✅ Model saved: xgb_{TASK}.json")


if __name__ == "__main__":
    train()