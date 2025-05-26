import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Read data
train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')

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

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1, max_depth=7)
model.fit(X_train, y_train_encoded)

# Predict probabilities
probs = model.predict_proba(X_test)
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
