# ============================================================
# PHISHING DETECTION PIPELINE
# FLOW:
# Raw Dataset
# -> Preprocessing
# -> Train/Test Split
# -> Branch 1: No SMOTE
# -> Branch 2: SMOTE-Tomek (train only)
# -> Information Gain (sau split; với branch SMOTE thì sau SMOTE)
# -> Model Training
# -> Evaluation
# -> Stacking
# -> SHAP Explainability
# ============================================================

import os
import time
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone

from imblearn.combine import SMOTETomek

# ============================================================
# OPTIONAL LIBRARIES
# ============================================================
XGB_OK = False
try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

TF_OK = False
SCIKERAS_OK = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_OK = True
except Exception:
    TF_OK = False

try:
    from scikeras.wrappers import KerasClassifier
    SCIKERAS_OK = True
except Exception:
    SCIKERAS_OK = False

SHAP_OK = False
try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False


# ============================================================
# CONFIG
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_K_FEATURES = 40

DATA_PATH = "dataset_full.csv"   # sửa lại nếu cần
TARGET_COL = "phishing"          # sửa lại nếu cần

RESULT_FILE = "model_results_comparison.csv"
IG_FILE_NO_SMOTE = "ig_no_smote_train.csv"
IG_FILE_SMOTE = "ig_smote_train.csv"

CNN_EPOCHS = 10
CNN_BATCH_SIZE = 128
CNN_VERBOSE = 0

# ===== NEW OUTPUT PATHS =====
TRAIN_TEST_DIR = "train_test_exports"
CONFUSION_DIR = "confusion_matrix_plots"
TIME_PLOT_DIR = "time_comparison_plots"
SHAP_DIR = "shap_outputs"


# ============================================================
# REPRODUCIBILITY
# ============================================================
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
if TF_OK:
    tf.random.set_seed(RANDOM_STATE)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def print_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_filename(text):
    invalid_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for ch in invalid_chars:
        text = text.replace(ch, "_")
    text = text.replace(" ", "_")
    return text


def load_data(path, target_col):
    df = pd.read_csv(path)

    if target_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột target '{target_col}' trong file dữ liệu.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Chỉ giữ numeric
    X = X.select_dtypes(include=[np.number]).copy()

    print_header("LOAD DATA")
    print(f"Dataset shape: {df.shape}")
    print(f"Numeric feature shape: {X.shape}")
    print("Class distribution:")
    print(y.value_counts())

    return df, X, y


def preprocess_before_split(X):
    """
    Theo yêu cầu của bạn:
    preprocess/scaling trước rồi mới split.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, imputer, scaler


def save_train_test_files(X_train, X_test, y_train, y_test, feature_names):
    """
    Lưu thêm file train/test sau khi preprocessing + split.
    """
    ensure_dir(TRAIN_TEST_DIR)

    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    y_train_df = pd.DataFrame({TARGET_COL: y_train.values})
    y_test_df = pd.DataFrame({TARGET_COL: y_test.values})

    train_df = X_train_df.copy()
    train_df[TARGET_COL] = y_train.values

    test_df = X_test_df.copy()
    test_df[TARGET_COL] = y_test.values

    X_train_df.to_csv(os.path.join(TRAIN_TEST_DIR, "X_train_processed.csv"), index=False, encoding="utf-8-sig")
    X_test_df.to_csv(os.path.join(TRAIN_TEST_DIR, "X_test_processed.csv"), index=False, encoding="utf-8-sig")
    y_train_df.to_csv(os.path.join(TRAIN_TEST_DIR, "y_train.csv"), index=False, encoding="utf-8-sig")
    y_test_df.to_csv(os.path.join(TRAIN_TEST_DIR, "y_test.csv"), index=False, encoding="utf-8-sig")
    train_df.to_csv(os.path.join(TRAIN_TEST_DIR, "train_processed_full.csv"), index=False, encoding="utf-8-sig")
    test_df.to_csv(os.path.join(TRAIN_TEST_DIR, "test_processed_full.csv"), index=False, encoding="utf-8-sig")

    print(f"\nSaved train/test files to folder: {TRAIN_TEST_DIR}")


def compute_information_gain(X_train_processed, y_train_processed, feature_names, top_k=40):
    """
    IG / Mutual Information tính trên TRAIN của từng branch.
    Với nhánh SMOTE thì tính sau SMOTE.
    """
    ig_scores = mutual_info_classif(
        X_train_processed,
        y_train_processed,
        discrete_features=False,
        random_state=RANDOM_STATE
    )

    ig_df = pd.DataFrame({
        "feature": feature_names,
        "information_gain": ig_scores
    }).sort_values(by="information_gain", ascending=False).reset_index(drop=True)

    selected_features = ig_df.head(top_k)["feature"].tolist()
    selected_indices = [feature_names.index(f) for f in selected_features]

    return ig_df, selected_features, selected_indices


def select_features(X, selected_indices):
    return X[:, selected_indices]


def build_cnn_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation="relu", padding="same"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def make_cnn_classifier(input_dim):
    if not (TF_OK and SCIKERAS_OK):
        return None

    return KerasClassifier(
        model=build_cnn_model,
        input_dim=input_dim,
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        verbose=CNN_VERBOSE
    )


def get_models(input_dim):
    models = {}

    models["Logistic Regression"] = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    models["KNN"] = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    )

    if XGB_OK:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    cnn_model = make_cnn_classifier(input_dim)
    if cnn_model is not None:
        models["CNN"] = cnn_model

    return models


def fit_and_predict(model, model_name, X_train, y_train, X_test):
    start_train = time.time()

    if model_name == "CNN":
        X_train_input = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_input = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        model.fit(X_train_input, y_train)
    else:
        X_train_input = X_train
        X_test_input = X_test
        model.fit(X_train_input, y_train)

    train_time = time.time() - start_train

    start_pred = time.time()
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_input)[:, 1]
    else:
        y_prob = model.predict(X_test_input).astype(float)
    pred_time = time.time() - start_pred

    y_pred = (y_prob >= 0.5).astype(int)

    return y_pred, y_prob, train_time, pred_time, model


def evaluate_model(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "ROC-AUC": auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


def clone_model_for_stacking(model_name, model, input_dim):
    if model_name == "CNN":
        return make_cnn_classifier(input_dim)
    return clone(model)


def manual_stacking(base_models_dict, meta_model, X_train, y_train, X_test, n_splits=5):
    """
    Custom stacking để dùng được cả CNN.
    Base learners tạo OOF predictions.
    Meta learner học từ OOF predictions.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    base_names = list(base_models_dict.keys())
    input_dim = X_train.shape[1]

    oof_preds = np.zeros((X_train.shape[0], len(base_names)))
    test_meta = np.zeros((X_test.shape[0], len(base_names)))

    total_train_time = 0.0
    total_pred_time = 0.0

    for j, base_name in enumerate(base_names):
        print(f"  -> Base learner: {base_name}")
        fold_test_preds = []

        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            model = clone_model_for_stacking(base_name, base_models_dict[base_name], input_dim)

            start_train = time.time()
            if base_name == "CNN":
                model.fit(X_tr.reshape((X_tr.shape[0], X_tr.shape[1], 1)), y_tr)
            else:
                model.fit(X_tr, y_tr)
            total_train_time += time.time() - start_train

            start_pred = time.time()
            if base_name == "CNN":
                val_prob = model.predict_proba(
                    X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
                )[:, 1]
                test_prob = model.predict_proba(
                    X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                )[:, 1]
            else:
                val_prob = model.predict_proba(X_val)[:, 1]
                test_prob = model.predict_proba(X_test)[:, 1]
            total_pred_time += time.time() - start_pred

            oof_preds[val_idx, j] = val_prob
            fold_test_preds.append(test_prob)

        test_meta[:, j] = np.mean(fold_test_preds, axis=0)

    start_train = time.time()
    meta_model.fit(oof_preds, y_train)
    total_train_time += time.time() - start_train

    start_pred = time.time()
    if hasattr(meta_model, "predict_proba"):
        y_prob = meta_model.predict_proba(test_meta)[:, 1]
    else:
        y_prob = meta_model.predict(test_meta).astype(float)
    total_pred_time += time.time() - start_pred

    y_pred = (y_prob >= 0.5).astype(int)

    return y_pred, y_prob, total_train_time, total_pred_time, meta_model


def plot_confusion_matrix_from_counts(tn, fp, fn, tp, title, save_path):
    """
    Vẽ confusion matrix từ TN, FP, FN, TP đã có trong bảng kết quả.
    """
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    classes = ["Non-Phishing (0)", "Phishing (1)"]
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title=title
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
                fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_all_confusion_matrix_plots(result_df):
    """
    Vẽ confusion matrix cho toàn bộ mô hình từ bảng kết quả cuối cùng.
    """
    ensure_dir(CONFUSION_DIR)

    for _, row in result_df.iterrows():
        branch = row["Branch"]
        model = row["Model"]
        tn = int(row["TN"])
        fp = int(row["FP"])
        fn = int(row["FN"])
        tp = int(row["TP"])

        title = f"Confusion Matrix - {branch} - {model}"
        filename = safe_filename(f"{branch}_{model}_confusion_matrix.png")
        save_path = os.path.join(CONFUSION_DIR, filename)

        plot_confusion_matrix_from_counts(
            tn=tn, fp=fp, fn=fn, tp=tp,
            title=title,
            save_path=save_path
        )

    print(f"\nSaved confusion matrix plots to folder: {CONFUSION_DIR}")


def plot_time_comparison(result_df):
    """
    Vẽ biểu đồ so sánh thời gian train và predict giữa các mô hình.
    """
    ensure_dir(TIME_PLOT_DIR)

    df_plot = result_df.copy()
    df_plot["Model_Branch"] = df_plot["Branch"] + " | " + df_plot["Model"]

    # --------------------------
    # Train time plot
    # --------------------------
    df_train = df_plot.sort_values(by="Train_Time_sec", ascending=False)

    plt.figure(figsize=(14, 7))
    plt.bar(df_train["Model_Branch"], df_train["Train_Time_sec"])
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Train Time (seconds)")
    plt.title("Model Training Time Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(TIME_PLOT_DIR, "training_time_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------
    # Predict time plot
    # --------------------------
    df_pred = df_plot.sort_values(by="Predict_Time_sec", ascending=False)

    plt.figure(figsize=(14, 7))
    plt.bar(df_pred["Model_Branch"], df_pred["Predict_Time_sec"])
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Predict Time (seconds)")
    plt.title("Model Prediction Time Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(TIME_PLOT_DIR, "prediction_time_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------
    # Combined grouped plot
    # --------------------------
    df_combined = df_plot.sort_values(by=["Branch", "Model"]).reset_index(drop=True)
    x = np.arange(len(df_combined))
    width = 0.38

    plt.figure(figsize=(16, 7))
    plt.bar(x - width / 2, df_combined["Train_Time_sec"], width, label="Train Time")
    plt.bar(x + width / 2, df_combined["Predict_Time_sec"], width, label="Predict Time")
    plt.xticks(x, df_combined["Model_Branch"], rotation=75, ha="right")
    plt.ylabel("Time (seconds)")
    plt.title("Training vs Prediction Time Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(TIME_PLOT_DIR, "train_vs_predict_time_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved time comparison plots to folder: {TIME_PLOT_DIR}")


def run_branch(branch_name, X_train, y_train, X_test, y_test, feature_names, use_smote=False, ig_file_name="ig.csv"):
    """
    Mỗi branch:
    - Nếu use_smote=True: SMOTE-Tomek trên train only
    - Sau đó IG trên train của branch
    - Chọn top-k features
    - Train models
    - Evaluate trên test
    """
    print_header(f"RUN BRANCH: {branch_name}")

    print("Train distribution BEFORE branch processing:")
    print(pd.Series(y_train).value_counts())

    # --------------------------------------------------------
    # SMOTE-Tomek on TRAIN ONLY
    # --------------------------------------------------------
    if use_smote:
        smote_tomek = SMOTETomek(random_state=RANDOM_STATE)
        X_train_branch, y_train_branch = smote_tomek.fit_resample(X_train, y_train)

        print("\nApplied SMOTE-Tomek on TRAIN ONLY")
        print("Train distribution AFTER SMOTE-Tomek:")
        print(pd.Series(y_train_branch).value_counts())
    else:
        X_train_branch = X_train.copy()
        y_train_branch = y_train.copy()

    # --------------------------------------------------------
    # Information Gain AFTER branch processing
    # --------------------------------------------------------
    ig_df, selected_features, selected_indices = compute_information_gain(
        X_train_processed=X_train_branch,
        y_train_processed=y_train_branch,
        feature_names=feature_names,
        top_k=TOP_K_FEATURES
    )

    ig_df.to_csv(ig_file_name, index=False, encoding="utf-8-sig")
    print(f"\nSaved IG ranking: {ig_file_name}")
    print("\nTop selected features:")
    print(ig_df.head(TOP_K_FEATURES))

    # --------------------------------------------------------
    # Feature selection
    # --------------------------------------------------------
    X_train_selected = select_features(X_train_branch, selected_indices)
    X_test_selected = select_features(X_test, selected_indices)

    print(f"\nX_train_selected shape: {X_train_selected.shape}")
    print(f"X_test_selected shape : {X_test_selected.shape}")

    # --------------------------------------------------------
    # Train single models
    # --------------------------------------------------------
    models = get_models(input_dim=X_train_selected.shape[1])
    results = []
    fitted_models = {}

    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")

        y_pred, y_prob, train_time, pred_time, fitted_model = fit_and_predict(
            model=model,
            model_name=model_name,
            X_train=X_train_selected,
            y_train=y_train_branch,
            X_test=X_test_selected
        )

        metrics = evaluate_model(y_test, y_pred, y_prob)

        row = {
            "Branch": branch_name,
            "Model": model_name,
            "Train_Time_sec": train_time,
            "Predict_Time_sec": pred_time,
            **metrics
        }
        results.append(row)
        fitted_models[model_name] = fitted_model

        print(pd.Series(row))

    # --------------------------------------------------------
    # Stacking
    # Base learners: LR + RF + KNN + CNN (nếu có)
    # Meta learner: XGBoost, nếu không có thì LogisticRegression
    # --------------------------------------------------------
    stacking_base_models = {}
    for name in ["Logistic Regression", "Random Forest", "KNN", "CNN"]:
        if name in models:
            stacking_base_models[name] = models[name]

    if len(stacking_base_models) >= 3:
        if XGB_OK:
            meta_model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            meta_name = "XGBoost"
        else:
            meta_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            meta_name = "Logistic Regression"

        print(f"\nTraining model: Stacking ({'+'.join(stacking_base_models.keys())} -> {meta_name})")

        y_pred, y_prob, train_time, pred_time, fitted_meta_model = manual_stacking(
            base_models_dict=stacking_base_models,
            meta_model=meta_model,
            X_train=X_train_selected,
            y_train=y_train_branch,
            X_test=X_test_selected,
            n_splits=5
        )

        metrics = evaluate_model(y_test, y_pred, y_prob)

        row = {
            "Branch": branch_name,
            "Model": f"Stacking ({'+'.join(stacking_base_models.keys())} -> {meta_name})",
            "Train_Time_sec": train_time,
            "Predict_Time_sec": pred_time,
            **metrics
        }
        results.append(row)
        fitted_models["Stacking_Meta"] = fitted_meta_model

        print(pd.Series(row))
    else:
        print("\nKhông đủ base learners để chạy stacking.")

    result_df = pd.DataFrame(results)

    return result_df, fitted_models, selected_features, X_test_selected


def run_shap_explainability(model, X_test_selected, selected_features, model_name, max_samples=200):
    """
    SHAP chạy sau khi train và evaluate xong.
    Ưu tiên model cây như XGBoost hoặc Random Forest.
    Giữ nguyên phần hình, chỉ thêm bảng SHAP importance.
    Sửa để waterfall plot hiển thị tên feature thật.
    """
    if not SHAP_OK:
        print("\nSHAP chưa được cài. Bỏ qua bước Explainability.")
        return

    ensure_dir(SHAP_DIR)

    print_header(f"RUN SHAP EXPLAINABILITY - {model_name}")

    sample_n = min(max_samples, len(X_test_selected))

    # =====================================================
    # Đổi sang DataFrame để giữ đúng tên feature
    # =====================================================
    X_sample_df = pd.DataFrame(
        X_test_selected[:sample_n],
        columns=selected_features
    )

    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_sample_df)

        # ==============================
        # SHAP TABLE
        # ==============================
        if hasattr(shap_values, "values"):
            shap_array = shap_values.values

            # Trường hợp output 3 chiều thì lấy class 1
            if len(shap_array.shape) == 3:
                shap_array = shap_array[:, :, 1]

            mean_abs_shap = np.abs(shap_array).mean(axis=0)

            shap_importance_df = pd.DataFrame({
                "feature": selected_features,
                "mean_abs_shap": mean_abs_shap
            }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

            shap_csv_path = os.path.join(SHAP_DIR, "shap_feature_importance_table.csv")
            shap_xlsx_path = os.path.join(SHAP_DIR, "shap_feature_importance_table.xlsx")

            shap_importance_df.to_csv(shap_csv_path, index=False, encoding="utf-8-sig")
            try:
                shap_importance_df.to_excel(shap_xlsx_path, index=False)
            except Exception:
                pass

            print("\nSaved SHAP importance table:")
            print(shap_importance_df.head(20))
            print(f"CSV : {shap_csv_path}")
            print(f"XLSX: {shap_xlsx_path}")

        # ==============================
        # SHAP SUMMARY PLOT
        # ==============================
        print("Hiển thị SHAP summary plot...")
        plt.figure()
        shap.summary_plot(
            shap_values,
            X_sample_df,
            feature_names=selected_features,
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, "shap_summary_plot.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # ==============================
        # SHAP WATERFALL PLOT
        # ==============================
        print("Hiển thị SHAP waterfall plot cho mẫu đầu tiên...")

        # Lấy values và base_values an toàn hơn cho nhiều kiểu output
        if hasattr(shap_values, "values"):
            first_values = shap_values.values[0]
            if len(np.array(first_values).shape) > 1:
                first_values = first_values[:, 1] if np.array(first_values).shape[-1] > 1 else first_values[:, 0]
        else:
            raise ValueError("shap_values không có thuộc tính 'values'.")

        base_vals = shap_values.base_values
        if np.ndim(base_vals) == 0:
            first_base_value = base_vals
        else:
            first_base_value = base_vals[0]
            if isinstance(first_base_value, (list, np.ndarray)) and np.ndim(first_base_value) > 0:
                first_base_value = first_base_value[1] if len(first_base_value) > 1 else first_base_value[0]

        first_explanation = shap.Explanation(
            values=first_values,
            base_values=first_base_value,
            data=X_sample_df.iloc[0].values,
            feature_names=selected_features
        )

        plt.figure()
        shap.plots.waterfall(first_explanation, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, "shap_waterfall_first_sample.png"), dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"Không chạy được SHAP cho model {model_name}: {e}")


def choose_best_model_for_shap(result_df, fitted_models):
    """
    Ưu tiên XGBoost -> Random Forest để giải thích bằng SHAP.
    """
    preferred_order = ["XGBoost", "Random Forest"]

    for model_name in preferred_order:
        if model_name in fitted_models and model_name in result_df["Model"].values:
            return model_name, fitted_models[model_name]

    return None, None


# ============================================================
# MAIN
# ============================================================
def main():
    # --------------------------------------------------------
    # 1) Load data
    # --------------------------------------------------------
    df, X, y = load_data(DATA_PATH, TARGET_COL)
    feature_names = list(X.columns)

    # --------------------------------------------------------
    # 2) Preprocessing BEFORE split
    # --------------------------------------------------------
    print_header("PREPROCESSING BEFORE SPLIT")
    X_processed, imputer, scaler = preprocess_before_split(X)
    print(f"Processed data shape: {X_processed.shape}")

    # --------------------------------------------------------
    # 3) Train/Test split
    # --------------------------------------------------------
    print_header("TRAIN / TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")

    print("\nTrain distribution:")
    print(pd.Series(y_train).value_counts())

    print("\nTest distribution:")
    print(pd.Series(y_test).value_counts())

    # --------------------------------------------------------
    # Save train/test files
    # --------------------------------------------------------
    save_train_test_files(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names
    )

    # --------------------------------------------------------
    # 4) Branch 1: No SMOTE
    # --------------------------------------------------------
    result_no_smote, fitted_no_smote, selected_features_no_smote, X_test_no_smote_selected = run_branch(
        branch_name="No_SMOTE",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        use_smote=False,
        ig_file_name=IG_FILE_NO_SMOTE
    )

    # --------------------------------------------------------
    # 5) Branch 2: SMOTE-Tomek (train only)
    # --------------------------------------------------------
    result_smote, fitted_smote, selected_features_smote, X_test_smote_selected = run_branch(
        branch_name="SMOTE_Tomek",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        use_smote=True,
        ig_file_name=IG_FILE_SMOTE
    )

    # --------------------------------------------------------
    # 6) Combine results
    # --------------------------------------------------------
    final_result_df = pd.concat([result_no_smote, result_smote], axis=0).reset_index(drop=True)

    final_result_df = final_result_df.sort_values(
        by=["F1-score", "Recall", "ROC-AUC", "Accuracy"],
        ascending=False
    ).reset_index(drop=True)

    print_header("FINAL RESULTS")
    print(final_result_df)

    final_result_df.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSaved result file: {RESULT_FILE}")

    # --------------------------------------------------------
    # Confusion matrix plots
    # --------------------------------------------------------
    save_all_confusion_matrix_plots(final_result_df)

    # --------------------------------------------------------
    # Time comparison plots
    # --------------------------------------------------------
    plot_time_comparison(final_result_df)

    # --------------------------------------------------------
    # 7) SHAP on best model from SMOTE branch
    # --------------------------------------------------------
    print_header("CHOOSE BEST MODEL FOR SHAP (SMOTE BRANCH)")
    model_name_for_shap, model_for_shap = choose_best_model_for_shap(result_smote, fitted_smote)

    if model_name_for_shap is not None:
        print(f"Model selected for SHAP: {model_name_for_shap}")
        run_shap_explainability(
            model=model_for_shap,
            X_test_selected=X_test_smote_selected,
            selected_features=selected_features_smote,
            model_name=model_name_for_shap,
            max_samples=200
        )
    else:
        print("Không có model phù hợp để chạy SHAP.")

    print_header("DONE")


if __name__ == "__main__":
    main()