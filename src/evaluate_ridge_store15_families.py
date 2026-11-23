from pathlib import Path
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(r"C:\projects\Store Sales-ML")
sys.path.insert(0, str(ROOT / "src"))
import modeling as m

FAMILIES = ["GROCERY II", "EGGS", "PET SUPPLIES", "POULTRY"]


def backtest2():
    raw, test, holidays = m.original_pipeline()
    del test, holidays
    frame = m.prepare_features(raw)
    del raw
    frame = frame.loc[frame.date.le("2016-08-31")].copy()
    daily = frame.groupby(["date", "store_nbr"], observed=True).sales.sum().unstack().asfreq("D")
    window = daily.shift(1).rolling(28, min_periods=28)
    mean = window.mean()
    cv = window.std(ddof=1).div(mean.where(mean > 0))
    frame[m.CV] = cv.stack().reindex(pd.MultiIndex.from_frame(frame[["date", "store_nbr"]])).to_numpy()
    train = frame.loc[frame.date.lt("2016-08-16")]
    valid = frame.loc[frame.date.ge("2016-08-16")].copy()
    valid[m.CV] = valid.store_nbr.map(cv.loc["2016-08-16"]).astype(float)
    path = ROOT / "results/trial17_fold1_backtest_results/20261111_140112_017715/backtest_paired_predictions.csv"
    d = valid[m.ROW_KEYS + ["sales", m.CV]].merge(
        pd.read_csv(path, parse_dates=["date"]), on=m.ROW_KEYS, how="left",
        validate="one_to_one", suffixes=("", "_ref"), sort=False)
    np.testing.assert_allclose(d.sales, d.sales_ref)
    np.testing.assert_allclose(d[m.CV], d[m.CV + "_ref"], atol=1e-8)
    d["ridge_sales"] = np.expm1(m.candidate_log_predictions("ridge", train, valid, "backtest2"))
    return d.rename(columns={"baseline_pred_sales": "trial17", "experiment_pred_sales": "cv"})


def main():
    os.chdir(ROOT)
    cfg = json.loads((m.RF_REFERENCE / "config.json").read_text())
    r = ROOT / "results"
    paths = {
        "fold1": r / "ridge_cv_fold1_selected/20261120_165234_245585/predictions.csv",
        **{f: r / f"ridge_cv_fold23/20261120_164447_208798/{f}_predictions.csv" for f in ("fold2", "fold3")},
        **{f: r / f"ridge_10pct_backtests/20261120_170133_640280/{f}_predictions.csv" for f in ("backtest1", "backtest3")},
        "backtest2": None,
    }
    out = r / "ridge_store15_four_families" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    scores = []
    for split, path in paths.items():
        d = pd.read_csv(path) if path else backtest2()
        assert len(d) == 28512 and not d.duplicated(m.ROW_KEYS).any()
        store = d.store_nbr.eq(15)
        family = d.family.isin(FAMILIES)
        rate = np.select([d.family.eq("GROCERY II"), store & family, store, family],
                         [0.10, 0.315, 0.53, 0.10], default=0.0)
        s = d.store_nbr.isin(cfg["store_cv"])
        f = d.family.isin(cfg["family_cv"])
        share = cfg["store_weight"] * s + cfg["family_weight"] * f
        weight = rate * share
        active = weight.gt(0)
        if path:
            col = "blend_original" if "blend_original" in d else "baseline_sales"
            b = np.log1p(d[col])
        else:
            # The uplift prediction is unavailable here. All modified rows have zero uplift share.
            assert not (active & d.family.isin(cfg["family_uplift"])).any()
            b = (1-share)*np.log1p(d.trial17) + share*np.log1p(d.cv)
            b = b.where(active)
        pred = b + weight * (np.log1p(d.ridge_sales) - np.log1p(d.cv))
        assert pred[active].notna().all()
        y = np.log1p(d.sales)
        gain = ((b[active]-y[active])**2 - (pred[active]-y[active])**2).sum()/len(d)
        for scope, mask in [("affected", active), ("all", pd.Series(True, index=d.index))]:
            if scope == "all" and not path:
                baseline = mixed = np.nan
            else:
                baseline = np.sqrt(np.mean((b[mask]-y[mask])**2))
                mixed = np.sqrt(np.mean((pred[mask]-y[mask])**2))
            scores.append([split, scope, active.sum(), baseline, mixed, baseline-mixed, gain])
        rows = d.loc[active,m.ROW_KEYS + ["sales"]].copy()
        rows["cv_ridge_rate"] = rate[active]
        rows["final_ridge_weight"] = weight[active]
        rows["baseline_sales"] = np.expm1(b[active])
        rows["mixed_sales"] = np.expm1(pred[active])
        m.save_csv(rows,out/f"{split}_affected.csv")
    table = pd.DataFrame(scores,columns=["split","scope","affected","baseline","mixed","rmsle_gain","overall_msle_gain"])
    m.save_csv(table,out/"scores.csv")
    print(table.to_string(index=False)); print(out)


if __name__ == "__main__":
    main()
