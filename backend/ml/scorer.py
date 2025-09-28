import sys, json, os
import pandas as pd
import numpy as np
from joblib import load

MODEL_BUNDLE_PATH = os.environ.get("MODEL_BUNDLE_PATH", "ml/models/fraud_ensemble_bundle.joblib")
THRESHOLD = float(0.45)

# lazy-load (요청당 실행이면 큰 의미 없지만, worker 모드에서 이점)
_bundle = load(MODEL_BUNDLE_PATH)
pipe_old = _bundle["pipe_old"]
pipe_local = _bundle["pipe_local"]
threshold = float(_bundle.get("threshold", THRESHOLD))  # 번들 값 우선, env로 override 가능
MODEL_VERSION = "ensemble_v1"
SCHEMA_VERSION = "v1.0.0"

LEAK_COLS = ["MonthClaimed", "DayOfWeekClaimed", "WeekOfMonthClaimed"]

def to_df(payload):
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    elif isinstance(payload, list):
        return pd.DataFrame(payload)
    else:
        raise ValueError("Payload must be object or list of objects")

def proba(pipe, X):
    clf = pipe.named_steps["clf"]
    return pipe.predict_proba(X)[:,1] if hasattr(clf, "predict_proba") else pipe.decision_function(X)

def score(df):
    for c in LEAK_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
    p1 = proba(pipe_old, df)
    p2 = proba(pipe_local, df)
    p = (p1 + p2) / 2.0
    d = (p >= threshold).astype(int)
    out = []
    for pi, di in zip(p, d):
        out.append({
            "proba": float(pi),
            "decision": int(di),
            "threshold": threshold,
            "model_version": MODEL_VERSION,
            "schema_version": SCHEMA_VERSION
        })
    return out

def main():
    raw = sys.stdin.read()
    payload = json.loads(raw)
    df = to_df(payload)
    out = score(df)
    sys.stdout.write(json.dumps(out))

if __name__ == "__main__":
    main()
