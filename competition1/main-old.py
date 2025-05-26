import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Drop 'id' column
X = train_df.drop(columns=["id", "smoking"])
y = train_df["smoking"]

# Preprocessing: fill missing values if any
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)
test_X = imputer.transform(test_df.drop(columns=["id"]))

# Train/test split (optional, for validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Random Forest
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# Model: XGBoost
model = XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.02, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)


# Validation report
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))

# Predict on test data
test_preds = model.predict(test_X)

# Save predictions
submission = pd.read_csv("sample_submission.csv")
submission["smoking"] = test_preds
submission.to_csv("submission.csv", index=False)
