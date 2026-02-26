import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

# ---- CNN deps ----
try:
    from tensorflow import keras
    from scikeras.wrappers import KerasClassifier
except Exception:
    keras = None
    KerasClassifier = None


# ======================
# CONFIG
# ======================
OUT_TRAIN = "train_split.csv"
OUT_TEST = "test_split.csv"
DATA_CSV = "dataset_balanced_smotetomek.csv"
RANKING_CSV = "information_gain_ranking.csv"
TARGET = "phishing"

TOP_K = 40
TEST_SIZE = 0.2
RANDOM_STATE = 42

OUT_RESULTS = "results_baseline.csv"

FAST_MODE = True
SAMPLE_FRAC = 1.0


# ======================
# UTIL
# ======================
def load_topk_features(ranking_csv: str, top_k: int) -> list:
    ig = pd.read_csv(ranking_csv)
    ig.columns = [c.strip().lower() for c in ig.columns]
    if "feature" not in ig.columns:
        raise ValueError(f"File ranking phải có cột 'feature'. Columns hiện có: {list(ig.columns)}")
    return ig["feature"].head(top_k).astype(str).tolist()

def build_linear(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", model)
    ])

def build_tree(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", model)
    ])

def fit_with_timing(est, X_train, y_train):
    t0 = time.time()
    est.fit(X_train, y_train)
    return est, time.time() - t0

def predict_with_timing(est, X_test):
    t0 = time.time()
    _ = est.predict(X_test)
    return time.time() - t0

def evaluate_model(name, est, X_test, y_test, fit_time, pred_time) -> dict:
    y_pred = est.predict(X_test)

    y_score = None
    if hasattr(est, "predict_proba"):
        try:
            y_score = est.predict_proba(X_test)[:, 1]
        except Exception:
            y_score = None
    if y_score is None and hasattr(est, "decision_function"):
        try:
            y_score = est.decision_function(X_test)
        except Exception:
            y_score = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    auc = np.nan
    if y_score is not None:
        try:
            auc = roc_auc_score(y_test, y_score)
        except Exception:
            auc = np.nan

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fit_time_sec": fit_time,
        "predict_time_sec": pred_time
    }


# ======================
# CNN (tabular 1D-CNN)
# ======================
def _reshape_for_cnn(X):
    X = np.asarray(X, dtype=np.float32)
    return X.reshape((X.shape[0], X.shape[1], 1))

def make_cnn_model(n_features: int, lr: float = 1e-3, dropout: float = 0.2):
    inputs = keras.Input(shape=(n_features, 1))
    x = keras.layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")]
    )
    return model

def build_cnn(n_features: int):
    if KerasClassifier is None:
        raise ImportError("Thiếu scikeras/tensorflow. Cài: pip install -U scikeras tensorflow")

    epochs = 8 if FAST_MODE else 20
    batch_size = 256 if FAST_MODE else 128

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    cnn_clf = KerasClassifier(
        model=make_cnn_model,
        model__n_features=n_features,
        model__lr=1e-3,
        model__dropout=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_split=0.2,
        callbacks=callbacks,
        random_state=RANDOM_STATE
    )

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reshape", FunctionTransformer(_reshape_for_cnn, validate=False)),
        ("clf", cnn_clf)
    ])


# ======================
# MAIN
# ======================
def main():
    if XGBClassifier is None:
        raise ImportError("Bạn chưa cài xgboost. Chạy: pip install xgboost")

    print("=== BASELINE: LR + RF + KNN + CNN + XGB ===")
    print("FAST_MODE:", FAST_MODE, "| SAMPLE_FRAC:", SAMPLE_FRAC)

    df = pd.read_csv(DATA_CSV)
    if TARGET not in df.columns:
        raise ValueError(f"Không tìm thấy cột target '{TARGET}' trong {DATA_CSV}")

    if SAMPLE_FRAC < 1.0:
        df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE).reset_index(drop=True)

    top_features = load_topk_features(RANKING_CSV, TOP_K)
    missing = [c for c in top_features if c not in df.columns]
    if missing:
        raise ValueError(f"Các feature sau có trong ranking nhưng không có trong data: {missing[:10]} ...")

    X = df[top_features].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # save splits
    train_df = X_train.copy(); train_df[TARGET] = y_train.values
    test_df = X_test.copy();  test_df[TARGET] = y_test.values
    train_df.to_csv(OUT_TRAIN, index=False)
    test_df.to_csv(OUT_TEST, index=False)

    n_features = X_train.shape[1]

    lr = build_linear(LogisticRegression(max_iter=2000, n_jobs=-1))

    rf_estimators = 200 if FAST_MODE else 300
    rf = build_tree(RandomForestClassifier(
        n_estimators=rf_estimators, random_state=RANDOM_STATE, n_jobs=-1
    ))

    knn_k = 7 if FAST_MODE else 5
    knn = build_linear(KNeighborsClassifier(n_neighbors=knn_k))

    cnn = build_cnn(n_features=n_features)

    xgb_estimators = 200 if FAST_MODE else 500
    xgb_depth = 4 if FAST_MODE else 6
    xgb = build_tree(XGBClassifier(
        n_estimators=xgb_estimators,
        learning_rate=0.05,
        max_depth=xgb_depth,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist"
    ))

    models = {
        "LR_baseline": lr,
        "RF_baseline": rf,
        "KNN_baseline": knn,
        "CNN_baseline": cnn,
        "XGB_baseline": xgb
    }

    results = []
    for name, est in models.items():
        print(f"\n--- Training {name} ---")
        fitted, fit_time = fit_with_timing(est, X_train, y_train)
        pred_time = predict_with_timing(fitted, X_test)
        row = evaluate_model(name, fitted, X_test, y_test, fit_time, pred_time)
        results.append(row)
        print(f"{name} | fit={fit_time:.2f}s | pred={pred_time:.2f}s | F1={row['f1']:.4f} | AUC={row['roc_auc']:.4f}")

    res_df = pd.DataFrame(results).sort_values(by="f1", ascending=False)
    res_df.to_csv(OUT_RESULTS, index=False)

    print("\n=== DONE: BASELINE ===")
    print("Saved:", os.path.abspath(OUT_RESULTS))
    print(res_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc",
                  "fit_time_sec", "predict_time_sec"]].to_string(index=False))


if __name__ == "__main__":
    main()