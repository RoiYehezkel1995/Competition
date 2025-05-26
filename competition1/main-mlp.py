import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ==== Feature Engineering ====
def add_features(df):
    df = df.copy()
    df["BMI"] = df["weight(kg)"] / ((df["height(cm)"] / 100) ** 2)
    df["pulse_pressure"] = df["systolic"] - df["relaxation"]
    df["eyesight_diff"] = abs(df["eyesight(left)"] - df["eyesight(right)"])
    df["hearing_diff"] = abs(df["hearing(left)"] - df["hearing(right)"])
    return df

# ==== Load Data ====
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
submission_df = pd.read_csv("sample_submission.csv")

train_df = add_features(train_df)
test_df = add_features(test_df)

X = train_df.drop(columns=["id", "smoking"])
y = train_df["smoking"]
X_test = test_df.drop(columns=["id"])

# ==== Preprocess ====
imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# ==== Build Model ====
def build_model(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(1024, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[keras.metrics.AUC(name="auc")])
    return model

# ==== Train with Stratified K-Fold ====
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds_all = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}")
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = build_model(X.shape[1])
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=128,
        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0
    )

    val_preds = model.predict(X_val).flatten()
    val_auc = roc_auc_score(y_val, val_preds)
    print(f"Validation AUC: {val_auc:.4f}")

    test_preds = model.predict(X_test).flatten()
    test_preds_all.append(test_preds)

# ==== Average Test Predictions and Export ====
ensemble_preds = np.mean(test_preds_all, axis=0)
final_preds = (ensemble_preds > 0.5).astype(int)

submission_df["smoking"] = final_preds
submission_df.to_csv("submission.csv", index=False)
