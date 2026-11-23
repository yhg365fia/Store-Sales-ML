"""Apply a frozen 26-day store inactivity rule to the selected Ridge blend."""
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(r"C:\projects\Store Sales-ML")
KEY = ["date", "store_nbr", "family"]


def zero_mask(daily, forecast):
    start = forecast.date.min()
    history = daily.reindex(pd.date_range(end=start-pd.Timedelta(days=1), periods=26))
    inactive = history.columns[history.eq(0).all()]
    eligible = forecast.store_nbr.isin(inactive)
    return eligible & forecast.onpromotion.eq(0), eligible, list(inactive)


def main():
    r = ROOT / "results"
    blend = r/"ridge_store15_four_families/20261120_171437_308643"
    sources = {
        "fold1": r/"ridge_cv_fold1_selected/20261120_165234_245585/predictions.csv",
        **{s:r/f"ridge_cv_fold23/20261120_164447_208798/{s}_predictions.csv" for s in ("fold2","fold3")},
        **{s:r/f"ridge_10pct_backtests/20261120_170133_640280/{s}_predictions.csv" for s in ("backtest1","backtest3")},
        "backtest2": r/"trial17_fold1_backtest_results/20261111_140112_017715/backtest_paired_predictions.csv",
    }
    cfg = json.loads((r/"trial17_cv_family721_oilraw_results/20261119_164707_823423/config.json").read_text())
    raw = pd.read_csv(ROOT/"data/train.csv",usecols=KEY+["sales","onpromotion"],parse_dates=["date"])
    assert not raw.duplicated(KEY).any()
    groups = raw.groupby(["date","store_nbr"]).sales
    daily = groups.sum().where(groups.count().eq(33)).unstack()
    out = r/"ridge_store_zero26"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    summary = []
    for split,path in sources.items():
        d = pd.read_csv(path,parse_dates=["date"])
        if split == "backtest2":
            store = d.store_nbr.isin(cfg["store_cv"])
            fam = d.family.isin(cfg["family_cv"])
            b = np.log1p(d.baseline_pred_sales)
            c = np.log1p(d.experiment_pred_sales)
            d["before"] = np.expm1(cfg["store_weight"]*np.where(store,c,b)
                                  +cfg["family_weight"]*np.where(fam,c,b))
            d.loc[d.family.isin(cfg["family_uplift"]),"before"] = np.nan
        else:
            d["before"] = d["blend_original" if "blend_original" in d else "baseline_sales"]
        start,end = d.date.min(),d.date.max()
        promo = raw.loc[raw.date.between(start,end),KEY+["onpromotion"]]
        d = d[KEY+["sales","before"]].merge(promo,on=KEY,validate="one_to_one",how="left")
        assert len(d)==28512 and d.onpromotion.notna().all()
        mixed = pd.read_csv(blend/f"{split}_affected.csv",parse_dates=["date"])
        d = d.merge(mixed[KEY+["mixed_sales"]],on=KEY,how="left",validate="one_to_one")
        assert d.mixed_sales.notna().sum()==len(mixed)
        d["before"] = d.mixed_sales.combine_first(d.before)
        mask,eligible,stores = zero_mask(daily,d)
        d["after"] = d.before.mask(mask,0)
        d["zero_rule"] = mask
        known = d.before.notna()
        def rmsle(pred, subset):
            return np.sqrt(np.mean((np.log1p(pred[subset])-np.log1p(d.sales[subset]))**2))
        summary.append([split,",".join(map(str,stores)),mask.sum(),(eligible & ~mask).sum(),
                        (mask & d.sales.gt(0)).sum(),known.sum(),rmsle(d.before,known),rmsle(d.after,known)])
        d.drop(columns="mixed_sales").to_csv(out/f"{split}_predictions.csv",index=False)
    scores=pd.DataFrame(summary,columns=["split","stores","zero_rows","promo_exceptions","positive_actual_zeroed","paired_rows","before","after"])
    scores["gain"]=scores.before-scores.after
    scores.to_csv(out/"scores.csv",index=False)
    print(scores.to_string(index=False));print(out)


if __name__ == "__main__":
    main()
