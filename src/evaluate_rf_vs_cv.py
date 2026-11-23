"""RF alone vs the Trial17 + store CV LightGBM model, and CV 90% + RF 10% (log space).

Same 25 Trial17 features + store_sales_cv_28d, same sales_log target, same train cutoffs.
RF itself comes from modeling.candidate_log_predictions("rf", ...).
CV predictions are the saved ones used in the CV vs Ridge comparison.

Run from the project root:
    py "src/evaluate_rf_vs_cv.py"            # all five splits
    py "src/evaluate_rf_vs_cv.py" fold2      # one split
"""
from datetime import datetime
from pathlib import Path
import gc
import os
import sys

import numpy as np
import pandas as pd

ROOT = Path(r"C:\projects\Store Sales-ML")
sys.path.insert(0, str(ROOT / "src"))
import modeling as m  # noqa: E402

R = ROOT / "results"
SPLITS = {  # valid_start, valid_end, saved CV predictions
    "fold2": ("2017-07-15", "2017-07-30", R / "ridge_cv_fold23/20261120_164447_208798/fold2_predictions.csv"),
    "fold3": ("2017-07-31", "2017-08-15", R / "ridge_cv_fold23/20261120_164447_208798/fold3_predictions.csv"),
    "backtest1": ("2016-07-31", "2016-08-15", R / "ridge_10pct_backtests/20261120_170133_640280/backtest1_predictions.csv"),
    "backtest3": ("2016-09-01", "2016-09-16", R / "ridge_10pct_backtests/20261120_170133_640280/backtest3_predictions.csv"),
    "backtest2": ("2016-08-16", "2016-08-31", R / "trial17_fold1_backtest_results/20261111_140112_017715/backtest_paired_predictions.csv"),
}


def main():
    os.chdir(ROOT)
    wanted = sys.argv[1:] or list(SPLITS)
    raw, test, holidays = m.original_pipeline()
    del test, holidays
    df = m.prepare_features(raw)
    del raw
    gc.collect()
    daily = df.groupby(["date", "store_nbr"], observed=True).sales.sum().unstack().asfreq("D")
    window = daily.shift(1).rolling(28, min_periods=28)
    mean = window.mean()
    cv = window.std(ddof=1).div(mean.where(mean > 0))
    df[m.CV] = cv.stack().reindex(pd.MultiIndex.from_frame(df[["date", "store_nbr"]])).to_numpy()

    out = R / "rf_vs_cv" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    scores = []
    for split in wanted:
        start, end, path = SPLITS[split]
        train = df.loc[df.date.lt(start)]
        valid = df.loc[df.date.between(start, end)].copy()
        valid[m.CV] = valid.store_nbr.map(cv.loc[start]).astype(float)  # frozen at origin
        ref = pd.read_csv(path, parse_dates=["date"], encoding="utf-8-sig")
        if "experiment_pred_sales" in ref:
            ref = ref.rename(columns={"experiment_pred_sales": "cv"})
        d = valid[m.ROW_KEYS + ["sales"]].merge(ref[m.ROW_KEYS + ["sales", "cv"]], on=m.ROW_KEYS,
                                                how="left", validate="one_to_one", suffixes=("", "_ref"))
        assert len(d) == 28512 and d.cv.notna().all()
        np.testing.assert_allclose(d.sales, d.sales_ref)

        rf_log = m.candidate_log_predictions("rf", train, valid, split)
        cv_log = np.log1p(d.cv.to_numpy())
        mix_log = cv_log + m.CANDIDATE_WEIGHT * (rf_log - cv_log)
        y = np.log1p(d.sales.to_numpy())
        r = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
        scores.append(dict(split=split, cv=r(cv_log), rf=r(rf_log), rf_minus_cv=r(rf_log) - r(cv_log),
                           cv_rf10=r(mix_log), cv_rf10_minus_cv=r(mix_log) - r(cv_log)))
        d["rf"] = np.expm1(rf_log)
        m.save_csv(d[m.ROW_KEYS + ["sales", "cv", "rf"]], out / f"{split}_predictions.csv")
        m.save_csv(pd.DataFrame(scores), out / "scores.csv")
        print(scores[-1], flush=True)
        del train, valid
        gc.collect()
    print(pd.DataFrame(scores).to_string(index=False))
    print(out)


if __name__ == "__main__":
    main()
