import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from scipy.stats import entropy


INPUT_CSV = "dataset_balanced_smotetomek.csv"
OUT_DIR = "."
TARGET = "phishing"
N_BINS = 5
TOP_SHOW = 40


def compute_entropy(series: pd.Series) -> float:
    probs = series.value_counts(normalize=True)
    return float(entropy(probs, base=2))


def compute_information_gain_entropy(
    input_csv: str = INPUT_CSV,
    out_dir: str = OUT_DIR,
    target_col: str = TARGET,
    n_bins: int = N_BINS,
    top_show: int = TOP_SHOW,
    treat_minus_one_as_missing: bool = False
) -> pd.DataFrame:

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    if target_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột target '{target_col}' trong file {input_csv}")

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    if treat_minus_one_as_missing:
        X = X.replace(-1, np.nan)

    # Impute
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Discretize để tính entropy/IG
    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy="uniform"
    )
    X_binned = discretizer.fit_transform(X_scaled)
    X_binned = pd.DataFrame(X_binned, columns=X.columns)

    H_y = compute_entropy(y)

    ig_scores = []
    for col in X_binned.columns:
        H_y_given_x = 0.0

        # duyệt từng giá trị rời rạc (bin)
        for v in X_binned[col].unique():
            mask = (X_binned[col] == v)
            subset = y[mask]

            weight = len(subset) / len(y)
            H_y_given_x += weight * compute_entropy(subset)

        ig_scores.append(H_y - H_y_given_x)

    ig_df = pd.DataFrame({
        "feature": X.columns,
        "information_gain": ig_scores
    }).sort_values(by="information_gain", ascending=False).reset_index(drop=True)

    out_path = os.path.join(out_dir, "information_gain_ranking.csv")
    ig_df.to_csv(out_path, index=False)

    print("Đã tạo ranking IG (entropy-based).")
    print("File:", os.path.abspath(out_path))
    print(f"\nTop {top_show} features theo IG:")
    print(ig_df.head(top_show).to_string(index=False))

    return ig_df


if __name__ == "__main__":
    compute_information_gain_entropy()