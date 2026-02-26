import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

BASELINE_CSV = "results_baseline.csv"
STACKING_CSV = "results_stacking.csv"
OUT_DIR = "plots_cm"
os.makedirs(OUT_DIR, exist_ok=True)

df_base = pd.read_csv(BASELINE_CSV)
df_stack = pd.read_csv(STACKING_CSV)
df = pd.concat([df_base, df_stack], ignore_index=True)

# đảm bảo là số nguyên
for c in ["tn", "fp", "fn", "tp"]:
    df[c] = pd.to_numeric(df[c]).astype(int)

for _, r in df.iterrows():
    name = str(r["model"])
    tn, fp, fn, tp = r["tn"], r["fp"], r["fn"], r["tp"]

    cm = np.array([[tn, fp],
                   [fn, tp]])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["0", "1"]  # chỉ để số, bạn tự chú thích sau
    )
    disp.plot(ax=ax, values_format="d")

    ax.set_title("")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=220)
    plt.close(fig)

print("Đã xuất xong các ma trận.")