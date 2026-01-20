import os
import pandas as pd

from src.experts.experts_classes.expert_RandomForest import RandomForestExpert
from src.experts.experts_classes.expert_LGBM import LGBMExpert
from src.experts.experts_classes.expert_ElasticNet import ElasticNetExpert

DATA_PATH = "data/cleaned/data_engineering_belgique.csv"
OUT_PATH = "data/output1/expertsvs2.csv"

DT_COL = "Date_Heure"
TARGET_COL = "Eolien_MW"
CUT_HOUR = 10

# simulate only last ~9 months of provider delivery
PROVIDER_MONTHS = 9

KEEP_ONLY_FULL_DAYS = True


def _day_ahead_index(issue_dt) -> pd.DatetimeIndex:
    issue_dt = pd.Timestamp(issue_dt)
    next_day = issue_dt.normalize() + pd.Timedelta(days=1)
    return pd.date_range(start=next_day, periods=24, freq="H")


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df[DT_COL] = pd.to_datetime(df[DT_COL])
    df = df.sort_values(DT_COL).reset_index(drop=True)

    if df[DT_COL].duplicated().any():
        dup = df.loc[df[DT_COL].duplicated(), DT_COL].iloc[0]
        raise ValueError(f"Duplicate timestamp found: {dup}")

    df = df.set_index(DT_COL)

    # --- define provider period (last 9 months of issue_datetimes) ---
    end_time = df.index.max()
    forecast_start = end_time - pd.DateOffset(months=PROVIDER_MONTHS)

    # issue times = daily at 10:00, but only in [forecast_start, end_time]
    issue_times = df.index[(df.index.hour == CUT_HOUR) & (df.index.minute == 0) & (df.index.second == 0)]
    issue_times = sorted(pd.unique(issue_times))
    issue_times = [pd.Timestamp(t) for t in issue_times if pd.Timestamp(t) >= forecast_start]

    if len(issue_times) == 0:
        raise ValueError("No issue_times found >= forecast_start. Check timestamps / CUT_HOUR.")

    # --- fit once (blackbox provider model already trained) ---
    train_df = df.loc[df.index < issue_times[0]]
    if train_df.empty:
        raise ValueError("Train set is empty before forecast_start. Increase history or move forecast_start earlier.")

    X_train = train_df
    y_train = train_df[TARGET_COL]

    models = {
        "randomforest": RandomForestExpert(),
        "lgbm": LGBMExpert(),
        "elasticnet": ElasticNetExpert(),
    }

    print(f"Fitting models once on history: n={len(train_df)} rows, end_train={issue_times[0]}")
    for name, model in models.items():
        model.fit(X_train, y_train)

    out_rows = []
    total = len(issue_times)

    for k, issue_dt in enumerate(issue_times, start=1):
        if k % 20 == 0:
            print(f"Run {k}/{total} | issue={issue_dt}")

        target_times = _day_ahead_index(issue_dt)

        if KEEP_ONLY_FULL_DAYS and not target_times.isin(df.index).all():
            continue

        future_df = df.loc[df.index.intersection(target_times)]
        if len(future_df) != 24:
            continue

        preds = {}
        for name, model in models.items():
            y_pred = model.predict(future_df)
            if len(y_pred) != 24:
                raise ValueError(f"{name} predicted {len(y_pred)} instead of 24 at issue={issue_dt}")
            preds[name] = y_pred

        y_true_block = future_df[TARGET_COL].values
        for i, target_dt in enumerate(target_times):
            out_rows.append({
                "issue_datetime": issue_dt,
                "target_datetime": target_dt,
                "y_true": float(y_true_block[i]),
                "randomforest": float(preds["randomforest"][i]),
                "lgbm": float(preds["lgbm"][i]),
                "elasticnet": float(preds["elasticnet"][i]),
            })

    out = pd.DataFrame(out_rows)
    if not out.empty:
        out = out.sort_values(["issue_datetime", "target_datetime"]).reset_index(drop=True)

    out.to_csv(OUT_PATH, index=False)
    print(f"[OK] Saved {OUT_PATH} | rows={len(out)} | runs={out['issue_datetime'].nunique() if not out.empty else 0}")


if __name__ == "__main__":
    main()
