# ============================================================
# 3.2.2 Class Distribution & Imbalance Handling using SMOTE-Tomek
# ============================================================


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek


# ------------------------------------------------
# 1. Load dataset
# ------------------------------------------------
DATA_PATH = "dataset_full.csv"
TARGET_COL = "phishing"   # target của bạn


df = pd.read_csv(DATA_PATH)


if TARGET_COL not in df.columns:
   raise ValueError(f"Không tìm thấy cột '{TARGET_COL}'")


X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]


# ------------------------------------------------
# 2. Phân bố lớp ban đầu
# ------------------------------------------------
print("\n=== Class distribution BEFORE balancing ===")
class_counts = y.value_counts().sort_index()
print(class_counts)


plt.figure(figsize=(6,4))
plt.bar(["Legitimate (0)", "Phishing (1)"],
       class_counts.values,
       color=["#1f77b4", "#d62728"])
plt.title("Class Distribution Before SMOTE-Tomek")
plt.ylabel("Samples")
plt.tight_layout()
plt.show()


# ------------------------------------------------
# 3. Tiền xử lý toàn bộ dữ liệu
# (SMOTE cần dữ liệu numeric + không missing)
# ------------------------------------------------
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()


X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)


# ------------------------------------------------
# 4. Áp dụng SMOTE-Tomek
# ------------------------------------------------
print("\nApplying SMOTE-Tomek...")


smt = SMOTETomek(random_state=42)
X_balanced, y_balanced = smt.fit_resample(X_scaled, y)


# ------------------------------------------------
# 5. Phân bố lớp sau cân bằng
# ------------------------------------------------
print("\n=== Class distribution AFTER SMOTE-Tomek ===")
after_counts = pd.Series(y_balanced).value_counts().sort_index()
print(after_counts)


plt.figure(figsize=(6,4))
plt.bar(["Legitimate (0)", "Phishing (1)"],
       after_counts.values,
       color=["#1f77b4", "#d62728"])
plt.title("Class Distribution After SMOTE-Tomek")
plt.ylabel("Samples")
plt.tight_layout()
plt.show()


# ------------------------------------------------
# 6. Tạo dataset mới đã cân bằng
# ------------------------------------------------
X_balanced_df = pd.DataFrame(X_balanced, columns=X.columns)
df_balanced = X_balanced_df.copy()
df_balanced[TARGET_COL] = y_balanced


OUTPUT_PATH = "dataset_balanced_smotetomek.csv"
df_balanced.to_csv(OUTPUT_PATH, index=False)


print(f"\n Dataset balanced saved to: {OUTPUT_PATH}")
print("New dataset shape:", df_balanced.shape)

