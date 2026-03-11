import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
INPUT_FILE = "model_results_comparison.csv"
OUTPUT_DIR = "time_plots_2fig_vn"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# RÚT GỌN TÊN MÔ HÌNH
# ============================================================
def shorten_model_name(model_name):
    name_map = {
        "Logistic Regression": "LR",
        "Random Forest": "RF",
        "KNN": "KNN",
        "XGBoost": "XGB",
        "CNN": "CNN"
    }

    if model_name in name_map:
        return name_map[model_name]

    if model_name.startswith("Stacking"):
        return "Stacking"

    return model_name


# ============================================================
# CHUẨN BỊ DỮ LIỆU
# ============================================================
def prepare_labels(df):
    df = df.copy()
    df["Model_Short"] = df["Model"].apply(shorten_model_name)
    return df


# ============================================================
# VẼ BIỂU ĐỒ 1 NHÁNH
# ============================================================
def plot_branch_train_predict(df_branch, title, output_name):
    # Thứ tự hiển thị mong muốn
    desired_order = ["LR", "RF", "KNN", "XGB", "CNN", "Stacking"]

    df_branch = df_branch.copy()
    df_branch["order"] = df_branch["Model_Short"].apply(
        lambda x: desired_order.index(x) if x in desired_order else 999
    )
    df_branch = df_branch.sort_values(by="order").reset_index(drop=True)

    x = np.arange(len(df_branch))
    width = 0.36

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, df_branch["Train_Time_sec"], width, label="Huấn luyện")
    plt.bar(x + width/2, df_branch["Predict_Time_sec"], width, label="Dự đoán")

    plt.xticks(x, df_branch["Model_Short"], rotation=0)
    plt.ylabel("Thời gian (giây)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, output_name),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_cols = ["Branch", "Model", "Train_Time_sec", "Predict_Time_sec"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Thiếu cột bắt buộc: {col}")

    df = prepare_labels(df)

    # Nhánh không SMOTE
    df_no_smote = df[df["Branch"] == "No_SMOTE"].copy()

    # Nhánh SMOTE-Tomek
    df_smote = df[df["Branch"] == "SMOTE_Tomek"].copy()

    # Hình 1: Không SMOTE
    plot_branch_train_predict(
        df_branch=df_no_smote,
        title="So sánh thời gian huấn luyện và dự đoán giữa các mô hình (Không dùng SMOTE)",
        output_name="thoi_gian_khong_smote.png"
    )

    # Hình 2: SMOTE-Tomek
    plot_branch_train_predict(
        df_branch=df_smote,
        title="So sánh thời gian huấn luyện và dự đoán giữa các mô hình (Dùng SMOTE-Tomek)",
        output_name="thoi_gian_smote_tomek.png"
    )

    print(f"Đã lưu 2 biểu đồ vào thư mục: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()