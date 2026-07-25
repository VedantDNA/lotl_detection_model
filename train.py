"""
train_model.py
================
Trains an Isolation Forest anomaly detector on your windowed Procmon
feature CSVs, and evaluates how well it separates "attack" from "normal".

This script does exactly what we walked through step by step in chat --
nothing hidden, nothing extra. Read it top to bottom; each STEP comment
block matches a step we discussed.

HOW TO RUN:
    python3 train_model.py

REQUIRES (install once):
    pip install pandas scikit-learn
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


# ---------------------------------------------------------------------------
# CONFIG -- change these if your filenames/PIDs are different
# ---------------------------------------------------------------------------

NOTEPAD_CSV = "windowed_features.csv"          # every row here = attack
SVCHOST_CSV = "windowed_features_svchost.csv"  # mix of attack + normal
ATTACK_PID = 952                                # the injected process's PID in the svchost file

CONTAMINATION = 0.05   # our guess at what % of real-world traffic is anomalous
N_ESTIMATORS = 200     # how many random trees the Isolation Forest builds
TEST_SIZE = 0.20       # fraction of NORMAL data held back purely for testing
RANDOM_STATE = 42      # fixes randomness so results are reproducible


def step1_load_data():
    """STEP 1: Read the two CSVs into memory as tables (DataFrames)."""
    notepad = pd.read_csv(NOTEPAD_CSV)
    svchost = pd.read_csv(SVCHOST_CSV)
    print(f"[Step 1] Loaded notepad: {notepad.shape}, svchost: {svchost.shape}")
    return notepad, svchost


def step2_label_and_combine(notepad, svchost):
    """
    STEP 2: Attach the ground-truth answer key.
    label = 1 means "this window is part of the attack"
    label = 0 means "this window is normal/background"

    We know this because WE set up the attack -- the model never sees
    this column during training.
    """
    notepad = notepad.copy()
    svchost = svchost.copy()

    notepad["label"] = 1  # the entire notepad capture is attack traffic

    svchost["label"] = (svchost["pid"] == ATTACK_PID).astype(int)

    combined = pd.concat([notepad, svchost], ignore_index=True)
    print(f"[Step 2] Combined shape: {combined.shape}")
    print(f"[Step 2] Label counts:\n{combined['label'].value_counts()}\n")
    return combined


def step3_select_features(df):
    """
    STEP 3: Split the table into X (inputs the model sees) and y (the answer key).

    We drop 'pid', timestamps, and 'label' from X -- if we left pid in,
    the model could "cheat" by memorizing which PID is bad instead of
    learning what BEHAVIOR is bad.
    """
    exclude_cols = {"pid", "window_start_t", "window_end_t", "label"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].fillna(0.0)   # replace any missing values with 0
    y = df["label"]

    print(f"[Step 3] Using {len(feature_cols)} features: {feature_cols}\n")
    return X, y, feature_cols


def step4_train_test_split(X, y):
    """
    STEP 4: Hold back some NORMAL data purely for testing.

    We train only on normal (y==0) rows -- this is what makes it
    "unsupervised": the model never sees a single attack example
    during training. It only learns what "normal" looks like.

    We hold back 30% of normal data to check: does the model correctly
    call BRAND NEW normal data "normal", or does it get confused?
    """
    X_normal = X[y == 0]
    X_train, X_test_normal = train_test_split(
        X_normal, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[Step 4] Training rows (normal only): {X_train.shape[0]}")
    print(f"[Step 4] Held-out normal test rows: {X_test_normal.shape[0]}\n")
    return X_train


def step5_train_model(X_train):
    """STEP 5: Create the model object and fit it on normal data only."""
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train)
    print(f"[Step 5] Model trained on {X_train.shape[0]} rows.\n")
    return model


def step6_score_everything(model, X, df):
    """
    STEP 6: Use the trained model to score every window (including
    windows it never saw during training -- both normal and attack).

    score_samples() gives a raw number per row (sklearn convention:
    LOWER = more anomalous). We flip the sign so HIGHER = more anomalous,
    since that's more intuitive to read.

    predict() applies the contamination threshold for a hard yes/no call.
    """
    df = df.copy()
    df["anomaly_score"] = -model.score_samples(X)
    df["predicted_attack"] = (model.predict(X) == -1).astype(int)
    print("[Step 6] Scored all rows.\n")
    return df


def step7_evaluate(df):
    """
    STEP 7: Compare predictions against the true labels.

    ROC-AUC: if you picked one random attack row and one random normal
    row, what's the probability the attack row scores higher?
    0.5 = coin flip (useless). 1.0 = perfect separation.
    """
    auc = roc_auc_score(df["label"], df["anomaly_score"])
    print(f"[Step 7] ROC-AUC: {auc:.3f}  (0.5=random guessing, 1.0=perfect)\n")

    cm = confusion_matrix(df["label"], df["predicted_attack"])
    print("[Step 7] Confusion matrix (rows=truth, columns=predicted):")
    print("                 predicted_normal  predicted_attack")
    print(f"actually_normal       {cm[0][0]:5d}            {cm[0][1]:5d}")
    print(f"actually_attack       {cm[1][0]:5d}            {cm[1][1]:5d}\n")

    print("[Step 7] Full report:")
    print(classification_report(df["label"], df["predicted_attack"],
                                 target_names=["normal", "attack"]))
    return auc


def main():
    notepad, svchost = step1_load_data()
    df = step2_label_and_combine(notepad, svchost)
    X, y, feature_cols = step3_select_features(df)
    X_train = step4_train_test_split(X, y)
    model = step5_train_model(X_train)
    df = step6_score_everything(model, X, df)
    step7_evaluate(df)

    df.to_csv("scored_output.csv", index=False)
    print("Saved full results (scores + predictions) to scored_output.csv")


if __name__ == "__main__":
    main()