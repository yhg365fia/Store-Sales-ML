"""Apply the Experiment 44 Zero Sales 28-day + promo-safe rule to the best 80:20 submission.

Rule (same functions as src/experiment.py): a store/family whose 28 calendar days
before 2017-08-16 were all observed as 0 in train.csv, and which has no promotion on
any of the 16 test days, gets all 16 predictions set to 0. Other rows are unchanged.

Run from the project root:
    py "src/make_submission_zero28.py"
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from experiment import KEYS, ROW_KEYS, forecast_promotion_mask, zero_mask  # noqa: E402

DAYS = 28
BASE = ROOT / "submissions" / "submission_trial17_cv_family721_store80_family20_log_blend.csv"
OUT = ROOT / "submissions" / "submission_trial17_cv_family721_store80_family20_log_blend_zero28_promo_safe.csv"


def main():
    train = pd.read_csv(ROOT / "data" / "train.csv", usecols=ROW_KEYS + ["sales"], parse_dates=["date"])
    test = pd.read_csv(ROOT / "data" / "test.csv", parse_dates=["date"])
    sub = pd.read_csv(BASE)

    assert len(sub) == len(test) == 28512
    assert sub.id.equals(test.id), "submission id order differs from test.csv"
    assert sub.sales.notna().all() and sub.sales.ge(0).all()

    start = test.date.min()
    zero, n_series = zero_mask(train, test, start, DAYS)
    promo, _ = forecast_promotion_mask(test)
    mask = zero & ~promo

    out = sub.copy()
    out.loc[mask, "sales"] = 0.0

    assert len(out) == 28512 and out.id.equals(test.id)
    assert out.sales.notna().all() and out.sales.ge(0).all() and np.isfinite(out.sales).all()
    out.to_csv(OUT, index=False)

    changed = test.loc[mask, KEYS].drop_duplicates()
    print(f"zero-history series: {n_series} | promo-protected rows: {int((zero & promo).sum())}")
    print(f"rows set to 0: {int(mask.sum())} ({len(changed)} series)")
    print(f"already 0 before: {int((sub.sales[mask] == 0).sum())} | sum of zeroed predictions: {sub.sales[mask].sum():.4f}")
    print(changed.family.value_counts().to_string())
    print(OUT)


if __name__ == "__main__":
    main()
