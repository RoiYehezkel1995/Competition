import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


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

models = {
    # "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "Tree_RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, min_samples_leaf=3, max_depth=10),
    "Tree_XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-3, max_depth=4)
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
print('train',balanced_accuracy_score(y_train_encoded, model.predict(X_train)))

# Train model
# model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='neg_log_loss', n_jobs=-1)

# model = RandomForestClassifier(n_estimators=1000, random_state=42, min_samples_leaf=10, max_depth=10)
# model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1, max_depth=7)
# model.fit(X_train, y_train_encoded)

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
