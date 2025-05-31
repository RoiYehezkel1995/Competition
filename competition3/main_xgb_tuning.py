import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Data cleaning function
def clean_data(df):
    df = df.drop_duplicates().copy()
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            # Cap outliers
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)
            # Fill missing with median
            df[col] = df[col].fillna(df[col].median())
        elif df[col].dtype == 'object':
            # Standardize categorical values
            df[col] = df[col].astype(str).str.strip().str.lower()
            # Fill missing with mode
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
    return df

# Read data
train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')

# Clean data
# train = clean_data(train)
# test = clean_data(test)

# Features and target
X_train = train.drop(columns=['id', 'Status'])
y_train = train['Status']
X_test = test.drop(columns=['id'])

# One-hot encode categorical features
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Align columns in test set to match training set
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Fill NaNs with 0 after one-hot encoding
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Encode target labels as integers
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# Split for validation
X_train_func, X_val, y_train_func, y_val = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42)

# Hyperparameter tuning for XGBClassifier
param_dist = {
    'n_estimators': [100, 500, 1000, 2000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [2, 3, 4, 5, 6],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.5]
}

xgb = XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1, use_label_encoder=False)
random_search = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    n_iter=20,
    scoring='balanced_accuracy',
    n_jobs=-1,
    cv=3,
    verbose=2,
    random_state=42
)

random_search.fit(X_train_func, y_train_func)
print("Best parameters found:", random_search.best_params_)
print("Best balanced accuracy (CV):", random_search.best_score_)

# Evaluate on validation set
best_xgb = random_search.best_estimator_
val_pred = best_xgb.predict(X_val)
val_score = balanced_accuracy_score(y_val, val_pred)
print("Validation balanced accuracy (holdout):", val_score)

# Retrain on full training data
best_xgb.fit(X_train, y_train_encoded)

# Predict probabilities
probs = best_xgb.predict_proba(X_test)
class_order = le.classes_

# Prepare submission DataFrame
submission = pd.DataFrame(test['id'], columns=['id'])
for class_name in ['C', 'CL', 'D']:
    col = f'Status_{class_name}'
    if class_name in class_order:
        idx = list(class_order).index(class_name)
        submission[col] = probs[:, idx]
    else:
        submission[col] = 0.0

# Clip probabilities to avoid log(0) issues
eps = 1e-15
for col in ['Status_C', 'Status_CL', 'Status_D']:
    submission[col] = submission[col].clip(eps, 1 - eps)

# Save to CSV
submission.to_csv('submission.csv', index=False) 