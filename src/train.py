import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# ===============================
# CONFIG
# ===============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "balanced_missense_2000_clean.tsv")
MODEL_PATH = "../models/final_rf_model.pkl"
RANDOM_STATE = 42
N_SPLITS = 5

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv(DATA_PATH, sep="\t")

df["Label"] = df["Label"].map({
    "Pathogenic": 1,
    "Benign": 0
})

y = df["Label"].values
X = pd.get_dummies(df.drop(columns=["Label"])).values

# ===============================
# MODEL
# ===============================

rf = RandomForestClassifier(
    n_estimators=400,
    min_samples_split=5,
    random_state=RANDOM_STATE
)

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_scores = []
best_thresholds = []

# ===============================
# CROSS VALIDATION + THRESHOLD OPT
# ===============================

for train_idx, val_idx in skf.split(X, y):

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    rf.fit(X_train, y_train)
    probs = rf.predict_proba(X_val)[:, 1]

    best_f1 = 0
    best_t = 0.5

    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs > t).astype(int)
        f1 = f1_score(y_val, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    fold_scores.append(best_f1)
    best_thresholds.append(best_t)

print("Fold F1 Scores:", fold_scores)
print("Mean CV F1:", round(np.mean(fold_scores), 4))
print("Optimal Threshold (mean):", round(np.mean(best_thresholds), 2))

# ===============================
# TRAIN FINAL MODEL ON ALL DATA
# ===============================

rf.fit(X, y)
joblib.dump(rf, MODEL_PATH)

print("\nFinal model saved to:", MODEL_PATH)