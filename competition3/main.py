import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, StratifiedKFold


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

models = {
    "GradientBoosting": GradientBoostingClassifier(n_estimators=2000, learning_rate=0.05),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-3, max_depth=4)
}
# feature selection
# print(f"Before applying feature selection {X_train.shape[1]}")
# fs_model = XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1)
# fs_model.fit(X_train, y_train_encoded)
# selector = SelectFromModel(fs_model, threshold='median', prefit=True)  # Can change to 'mean' or a float

# X_train = selector.transform(X_train)
# X_test = selector.transform(X_test)
# print(f"Reduced to {X_train.shape[1]} features after selection.")
X_train_func, X_val, y_train_func, y_val = train_test_split(X_train, y_train_encoded, test_size=0.2)
best_model = ""
best_score = -np.inf
print("-----------------------------")
for clf_name, clf in models.items():
    cloned_clf = clone(clf)
    cloned_clf.fit(X_train_func, y_train_func)
    print("*************************")
    print(clf_name)
    score = balanced_accuracy_score(y_val, cloned_clf.predict(X_val))
    print(f"train score: {balanced_accuracy_score(y_train_func, cloned_clf.predict(X_train_func))}")
    print(f"test score: {score}")
    print("*************************")
    if(score > best_score):
        best_model = clf_name
        best_score = score
print("-----------------------------")
print(f"The best model is: {best_model}, the best score is: {best_score}")
model = models[best_model]
model.fit(X_train, y_train_encoded)
print('train',balanced_accuracy_score(y_train_encoded, cloned_clf.predict(X_train)))
print('test',balanced_accuracy_score(y_val, model.predict(X_val)))

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
