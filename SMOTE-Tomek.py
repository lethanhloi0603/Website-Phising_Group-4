import pandas as pd
from imblearn.combine import SMOTETomek
import os

# =====================================================
# 1. Load dữ liệu train trước khi cân bằng
# =====================================================
X_train = pd.read_csv("train_test_exports/X_train_processed.csv")
y_train = pd.read_csv("train_test_exports/y_train.csv")["phishing"]

print("===== CLASS DISTRIBUTION BEFORE BALANCING =====")
print(y_train.value_counts())

# =====================================================
# 2. Áp dụng SMOTE-Tomek
# =====================================================
sm = SMOTETomek(random_state=42)
X_sm, y_sm = sm.fit_resample(X_train, y_train)

print("\n===== CLASS DISTRIBUTION AFTER SMOTE-TOMEK =====")
print(pd.Series(y_sm).value_counts())

# =====================================================
# 3. Tạo bảng so sánh trước và sau cân bằng
# =====================================================
before_counts = y_train.value_counts().sort_index()
after_counts = pd.Series(y_sm).value_counts().sort_index()

summary = pd.DataFrame({
    "Before_SMOTE": before_counts,
    "After_SMOTE_Tomek": after_counts
})

summary["Change"] = summary["After_SMOTE_Tomek"] - summary["Before_SMOTE"]

print("\n===== COMPARISON TABLE =====")
print(summary)

# =====================================================
# 4. Tạo thư mục output nếu chưa tồn tại
# =====================================================
output_dir = "train_test_exports"
os.makedirs(output_dir, exist_ok=True)

# =====================================================
# 5. Lưu bảng so sánh
# =====================================================
summary.to_csv(
    f"{output_dir}/balance_comparison.csv",
    encoding="utf-8-sig"
)

# =====================================================
# 6. Lưu dữ liệu sau SMOTE
# =====================================================
X_sm_df = pd.DataFrame(X_sm, columns=X_train.columns)
y_sm_df = pd.DataFrame({"phishing": y_sm})

X_sm_df.to_csv(
    f"{output_dir}/X_train_smote_tomek.csv",
    index=False,
    encoding="utf-8-sig"
)

y_sm_df.to_csv(
    f"{output_dir}/y_train_smote_tomek.csv",
    index=False,
    encoding="utf-8-sig"
)

# train full
train_sm_df = X_sm_df.copy()
train_sm_df["phishing"] = y_sm

train_sm_df.to_csv(
    f"{output_dir}/train_smote_tomek_full.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== FILES SAVED =====")
print("balance_comparison.csv")
print("X_train_smote_tomek.csv")
print("y_train_smote_tomek.csv")
print("train_smote_tomek_full.csv")