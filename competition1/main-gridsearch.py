import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Feature Engineering
def add_features(df):
    df = df.copy()
    df["BMI"] = df["weight(kg)"] / ((df["height(cm)"] / 100) ** 2)
    df["pulse_pressure"] = df["systolic"] - df["relaxation"]
    df["eyesight_diff"] = abs(df["eyesight(left)"] - df["eyesight(right)"])
    df["hearing_diff"] = abs(df["hearing(left)"] - df["hearing(right)"])
    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

# Prepare features and labels
X = train_df.drop(columns=["id", "smoking"])
y = train_df["smoking"]
X_test = test_df.drop(columns=["id"])

# Pipeline: Imputation + Scaling + XGBoost
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("xgb", XGBClassifier(eval_metric='logloss', random_state=42))
])

# Hyperparameter Grid
param_grid = {
    "xgb__n_estimators": [500, 1000, 1500],
    "xgb__max_depth": [3, 4, 6],
    "xgb__learning_rate": [0.01, 0.02, 0.05],
    "xgb__subsample": [0.8, 1.0],
    "xgb__colsample_bytree": [0.8, 1.0],
}

# Train-validation split for internal validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,
    verbose=1,
    n_jobs=-1
)

# Train with best parameters
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Evaluate on validation set
y_pred = best_model.predict(X_val)
print("\nValidation Report:")
print(classification_report(y_val, y_pred))

# Predict on test set
X_test_transformed = test_df.drop(columns=["id"])
test_preds = best_model.predict(X_test_transformed)

# Save predictions
submission = pd.read_csv("sample_submission.csv")
submission["smoking"] = test_preds
submission.to_csv("submission.csv", index=False)

# Save best model
# joblib.dump(best_model, "best_xgb_model_tuned.pkl")
