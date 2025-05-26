import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

# Load and preprocess data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Feature engineering
def add_features(df):
    df = df.copy()
    df["BMI"] = df["weight(kg)"] / ((df["height(cm)"] / 100) ** 2)
    df["pulse_pressure"] = df["systolic"] - df["relaxation"]
    df["eyesight_diff"] = abs(df["eyesight(left)"] - df["eyesight(right)"])
    df["hearing_diff"] = abs(df["hearing(left)"] - df["hearing(right)"])
    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

# Features and labels
X = train_df.drop(columns=["id", "smoking"])
y = train_df["smoking"]
X_test = test_df.drop(columns=["id"])

# Preprocessing: imputation
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

# Optional: scaling (XGBoost is scale-invariant, but can help in some cases)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Split for internal validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Optuna objective function
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1
    }

    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
    return scores.mean()

# Run the Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, timeout=600)  # 50 trials or 10 minutes

print("Best trial:")
print(study.best_trial)

# Train the final model with best params
best_params = study.best_trial.params
best_params["eval_metric"] = "logloss"
best_params["random_state"] = 42

final_model = XGBClassifier(**best_params)
final_model.fit(X_train, y_train)

# Evaluate
y_pred = final_model.predict(X_val)
print("\nValidation Report:")
print(classification_report(y_val, y_pred))

# Predict test set
test_preds = final_model.predict(X_test)

# Save predictions
submission = pd.read_csv("sample_submission.csv")
submission["smoking"] = test_preds
submission.to_csv("submission.csv", index=False)

# Save model
# joblib.dump(final_model, "xgb_optuna_best_model.pkl")
