# fusion/dynamic_weights.py
import pandas as pd
import os
from config import TIMEFRAMES, LOG_DIR, INIT_WEIGHTS

WEIGHT_FILE = f"{LOG_DIR}/weight_history.csv"


def update_weight(tf, correct):
    if not os.path.exists(WEIGHT_FILE):
        pd.DataFrame(columns=["timestamp"] + TIMEFRAMES).to_csv(
            WEIGHT_FILE, index=False
        )

    df = pd.read_csv(WEIGHT_FILE)
    row = {"timestamp": pd.Timestamp.now()}
    row.update({t: 1 if t == tf and correct else 0 for t in TIMEFRAMES})
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(WEIGHT_FILE, index=False)


def get_dynamic_weights(window=50):
    if not os.path.exists(WEIGHT_FILE):
        return INIT_WEIGHTS.copy()

    df = pd.read_csv(WEIGHT_FILE).tail(window)
    if len(df) == 0:
        return INIT_WEIGHTS.copy()

    acc = df[TIMEFRAMES].mean()
    weights = acc / acc.sum()
    return weights.fillna(0.25).to_dict()
