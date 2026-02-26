import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# Load dữ liệu
# ===============================
df_base = pd.read_csv("results_baseline.csv")
df_stack = pd.read_csv("results_stacking.csv")
df = pd.concat([df_base, df_stack], ignore_index=True)

df["fit_time_sec"] = pd.to_numeric(df["fit_time_sec"])
df["predict_time_sec"] = pd.to_numeric(df["predict_time_sec"])

# ===============================
# Rút gọn tên model
# ===============================
def shorten(name):
    s = str(name).lower()
    if "stack" in s:
        return "STACK"
    if "cnn" in s:
        return "CNN"
    if "xgb" in s:
        return "XGB"
    if "knn" in s:
        return "KNN"
    if "rf" in s:
        return "RF"
    if "lr" in s:
        return "LR"
    return name.upper()

df["alg"] = df["model"].apply(shorten)

order = ["LR", "RF", "KNN", "XGB", "CNN", "STACK"]
df["alg"] = pd.Categorical(df["alg"], categories=order, ordered=True)
df = df.sort_values("alg")
df = df.drop_duplicates(subset=["alg"], keep="first")

# ===============================
# Chuẩn bị dữ liệu vẽ
# ===============================
algs = df["alg"].astype(str).tolist()
train_t = df["fit_time_sec"].values
test_t = df["predict_time_sec"].values

x = np.arange(len(algs))
width = 0.35

# ===============================
# VẼ BIỂU ĐỒ
# ===============================
plt.figure(figsize=(10,5.5))

# CÙNG MỘT MÀU cho tất cả train
bars_train = plt.bar(x - width/2, train_t,
                     width,
                     color="steelblue",
                     label="Huấn luyện")

# CÙNG MỘT MÀU cho tất cả test (khác train)
bars_test = plt.bar(x + width/2, test_t,
                    width,
                    color="darkorange",
                    label="Dự đoán")

plt.xticks(x, algs)
plt.ylabel("Thời gian (giây)")
plt.legend()

# Ghi số trên đầu cột
for bar in bars_train:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             h,
             f"{h:.2f}",
             ha='center', va='bottom', fontsize=9)

for bar in bars_test:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             h,
             f"{h:.4f}",
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("time_train_vs_test.png", dpi=300)
plt.show()