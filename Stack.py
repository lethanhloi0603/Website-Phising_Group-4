import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingClassifier
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
TRAIN_CSV = "train_split.csv"
TEST_CSV  = "test_split.csv"

RANKING_CSV = "information_gain_ranking.csv"
TARGET = "phishing"

TOP_K = 40
RANDOM_STATE = 42

OUT_STACKING = "results_stacking.csv"
N_SPLITS_STACK = 3
FAST_MODE = True


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
        loss="binary_crossentropy"
    )
    return model

def build_cnn(n_features: int):
    if KerasClassifier is None:
        raise ImportError("Thiếu scikeras/tensorflow. Cài: pip install -U scikeras tensorflow")

    epochs = 6 if FAST_MODE else 15
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

def align_train_test_columns(X_train: pd.DataFrame, X_test: pd.DataFrame):
    # đảm bảo cùng tập cột + cùng thứ tự
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    missing_in_test = [c for c in X_train.columns if c not in X_test.columns]
    missing_in_train = [c for c in X_test.columns if c not in X_train.columns]

    if missing_in_test:
        print(f"[WARN] Test thiếu {len(missing_in_test)} cột so với train. Ví dụ: {missing_in_test[:10]}")
    if missing_in_train:
        print(f"[WARN] Train thiếu {len(missing_in_train)} cột so với test. Ví dụ: {missing_in_train[:10]}")

    if len(common_cols) == 0:
        raise ValueError("Train và Test không có cột feature chung nào để chạy.")

    return X_train[common_cols].copy(), X_test[common_cols].copy(), common_cols


# ======================
# MAIN
# ======================
def main():
    print("=== STACKING: (LR + RF + KNN + CNN) -> XGB meta ===")
    print("N_SPLITS_STACK:", N_SPLITS_STACK)

    if XGBClassifier is None:
        raise ImportError("Bạn chưa cài xgboost. Chạy: pip install xgboost")

    # 1) Load đúng tập train/test đã split từ baseline
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    if TARGET not in train_df.columns or TARGET not in test_df.columns:
        raise ValueError(f"Không tìm thấy cột target '{TARGET}' trong train/test CSV.")

    # 2) Lấy TOP_K features theo IG (nhưng chỉ giữ những feature thật sự có trong dữ liệu)
    top_features = load_topk_features(RANKING_CSV, TOP_K)

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    selected_train = [c for c in top_features if c in train_cols and c != TARGET]
    selected_test  = [c for c in top_features if c in test_cols and c != TARGET]

    if len(selected_train) == 0 or len(selected_test) == 0:
        raise ValueError("Không chọn được feature nào từ ranking có trong train/test. Kiểm tra tên cột.")

    # ưu tiên giao nhau để tránh lệch cột
    selected = [c for c in selected_train if c in selected_test]
    if len(selected) == 0:
        raise ValueError("Train và test không có feature giao nhau theo ranking. Kiểm tra train_split/test_split/ranking.")

    if len(selected) < TOP_K:
        print(f"[INFO] TOP_K={TOP_K} nhưng chỉ dùng được {len(selected)} feature (do thiếu cột trong train/test).")

    X_train = train_df[selected].copy()
    y_train = train_df[TARGET].astype(int).values

    X_test = test_df[selected].copy()
    y_test = test_df[TARGET].astype(int).values

    # 3) Align lần nữa cho chắc (thứ tự cột)
    X_train, X_test, used_cols = align_train_test_columns(X_train, X_test)
    n_features = X_train.shape[1]
    print(f"[INFO] Using n_features = {n_features}")

    # 4) Build base learners
    lr = build_linear(LogisticRegression(max_iter=2000))
    rf = build_tree(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))
    knn = build_linear(KNeighborsClassifier(n_neighbors=7))
    cnn = build_cnn(n_features=n_features)

    cv = StratifiedKFold(n_splits=N_SPLITS_STACK, shuffle=True, random_state=RANDOM_STATE)

    meta = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist"
    )

    stack = StackingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("knn", knn), ("cnn", cnn)],
        final_estimator=meta,
        stack_method="predict_proba",
        cv=cv,
        n_jobs=-1
    )

    # 5) Fit + eval
    t0 = time.time()
    stack.fit(X_train, y_train)
    fit_time = time.time() - t0

    pred_time = predict_with_timing(stack, X_test)
    row = evaluate_model("Stacking_XGB_meta_(LR+RF+KNN+CNN)", stack, X_test, y_test, fit_time, pred_time)

    pd.DataFrame([row]).to_csv(OUT_STACKING, index=False)

    print("\n=== DONE: STACKING ===")
    print("Train file:", os.path.abspath(TRAIN_CSV))
    print("Test  file:", os.path.abspath(TEST_CSV))
    print("Saved:", os.path.abspath(OUT_STACKING))
    print(row)


if __name__ == "__main__":
    main()