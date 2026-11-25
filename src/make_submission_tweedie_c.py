"""Plan C submission: 80:20 log blend + Zero28, with Tweedie mixed 10% into the rows that are 100% Trial17.

Rows changed: the 40 stores that are not CV-routed on the store side x the 18 families routed to Trial17
on the family side (PLAYERS AND ELECTRONICS excluded). On those rows the 80:20 log prediction equals the
Trial17 log prediction, so the new value is 0.9 * log1p(80:20) + 0.1 * log1p(Tweedie). All other rows keep
the 80:20 value. Zero28 (28 zero days + promo-safe, src/experiment.py) is applied afterwards, as in
src/make_submission_zero28.py.

Tweedie: objective tweedie (variance power 1.5) on raw sales, 25 Trial17 features, otherwise TRIAL17 params.
Iterations: early stopping (RMSLE) on an inner holdout = the 16 days before the test period
(2017-07-31~2017-08-15, trained on data up to 2017-07-30), then refit on all data up to 2017-08-15
with that iteration count. Same rule as the 2016 backtests.

Run from the project root:
    py "src/make_submission_tweedie_c.py"
"""
import gc
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from experiment import forecast_promotion_mask, zero_mask  # noqa: E402

spec = importlib.util.spec_from_file_location("final_submission", ROOT / "src" / "final submission.py")
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)

BASE = ROOT / "submissions" / "submission_trial17_cv_family721_store80_family20_log_blend.csv"
BEST = ROOT / "submissions" / "submission_trial17_cv_family721_store80_family20_log_blend_zero28_promo_safe.csv"
CFG = ROOT / "results" / "trial17_cv_family721_oilraw_results" / "20261119_164707_823423" / "config.json"
EXCLUDED = ["PLAYERS AND ELECTRONICS", "HOME AND KITCHEN I"]
WEIGHT = 0.10
INNER_START = pd.Timestamp("2017-07-31")
PARAMS = {**fs.PARAMS, "objective": "tweedie", "tweedie_variance_power": 1.5, "metric": "None",
          "n_estimators": 8000}
N_ITER_ARG = int(sys.argv[sys.argv.index("--n-iter") + 1]) if "--n-iter" in sys.argv else None
OUT = ROOT / "submissions" / (
    "submission_store80_family20_zero28_tweedie10_trial17rows_40stores_18fam_"
    + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")


def rmsle_raw(y_true, y_pred):
    err = np.log1p(np.clip(y_true, 0, None)) - np.log1p(np.clip(y_pred, 0, None))
    return "rmsle", float(np.sqrt(np.mean(err ** 2))), False


def fit(xt, yt, n_iter, xv=None, yv=None):
    params = dict(PARAMS)
    callbacks = [lgb.log_evaluation(500)]
    if n_iter is None:
        callbacks.insert(0, lgb.early_stopping(150, first_metric_only=True, verbose=False))
    else:
        params["n_estimators"] = n_iter
        callbacks = [fs.progress]
        fs.N_ITERATIONS = n_iter
    model = lgb.LGBMRegressor(**params)
    kw = dict(eval_set=[(xv, yv)], eval_names=["inner"], eval_metric=rmsle_raw) if xv is not None else {}
    model.fit(xt, yt, categorical_feature=fs.CATEGORICAL, callbacks=callbacks, **kw)
    return model


def inner_early_stopping(train, feats):
    inner_tr = train[train.date < INNER_START]
    inner_va = train[train.date >= INNER_START]
    assert inner_va.date.nunique() == 16 and inner_va.date.max() == fs.TRAIN_END
    xt, xv = fs.make_matrices(inner_tr, inner_va, feats)
    print(f"[INNER] train {len(xt):,} (<= 2017-07-30) | holdout {len(xv):,} (2017-07-31~08-15)", flush=True)
    t1 = time.time()
    model = fit(xt, inner_tr["sales"], None, xv, inner_va["sales"])
    n_iter = int(model.best_iteration_ or 8000)
    print(f"[INNER] tweedie best_iter={n_iter} | holdout RMSLE {model.best_score_['inner']['rmsle']:.6f} | "
          f"{(time.time() - t1) / 60:.1f} min", flush=True)
    del xt, xv, model, inner_tr, inner_va
    gc.collect()
    return n_iter



def main():
    os.chdir(ROOT)
    from main import main as original_pipeline
    from feature_engineering import add_lag_features

    t0 = time.time()
    raw, original_test, _ = original_pipeline()
    train = fs.prepare_train(raw, add_lag_features)
    test = fs.prepare_test(original_test, train)
    del raw, original_test
    gc.collect()
    feats = fs.BASE_FEATURES
    assert len(feats) == 25
    print(f"[DATA] train {len(train):,} test {len(test):,} | {time.time() - t0:.0f}s", flush=True)

    # 1) inner holdout early stopping (skipped with --n-iter: the inner holdout is exactly Fold3,
    #    train <= 2017-07-30 / valid 2017-07-31~08-15, so the saved Tweedie Fold3 best_iteration is reused)
    if N_ITER_ARG is not None:
        n_iter = N_ITER_ARG
        print(f"[INNER] skipped: reuse Tweedie Fold3 early-stopping best_iter={n_iter}", flush=True)
    else:
        n_iter = inner_early_stopping(train, feats)

    # 2) full refit
    xt, xp = fs.make_matrices(train, test, feats)
    print(f"[FULL] refit {n_iter} iterations on {len(xt):,} rows", flush=True)
    t1 = time.time()
    model = fit(xt, train["sales"], n_iter)
    tw = np.log1p(np.clip(model.predict(xp, num_iteration=n_iter), 0, None))
    assert np.isfinite(tw).all()
    print(f"[FULL] done | {(time.time() - t1) / 60:.1f} min", flush=True)
    tw = pd.Series(tw, index=test["id"].to_numpy())

    # 3) plan C blend on the 80:20 submission, then Zero28
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    test_raw = pd.read_csv(ROOT / "data" / "test.csv", parse_dates=["date"])
    base = pd.read_csv(BASE)
    best = pd.read_csv(BEST)
    assert base.id.equals(test_raw.id) and best.id.equals(test_raw.id)
    rows = (~test_raw.store_nbr.isin(cfg["store_cv"]) & ~test_raw.family.isin(cfg["family_cv"])
            & ~test_raw.family.isin(cfg["family_uplift"]) & ~test_raw.family.isin(EXCLUDED)).to_numpy()
    blog = np.log1p(base.sales.to_numpy())
    new = np.where(rows, (1 - WEIGHT) * blog + WEIGHT * tw.reindex(test_raw.id).to_numpy(), blog)
    out = base.copy()
    out["sales"] = np.expm1(new)

    hist = pd.read_csv(ROOT / "data" / "train.csv", usecols=["date", "store_nbr", "family", "sales"],
                       parse_dates=["date"])
    zero, _ = zero_mask(hist, test_raw, test_raw.date.min(), 28)
    promo, _ = forecast_promotion_mask(test_raw)
    zm = zero & ~promo
    out.loc[zm, "sales"] = 0.0
    assert len(out) == 28512 and out.id.equals(test_raw.id)
    assert out.sales.notna().all() and out.sales.ge(0).all() and np.isfinite(out.sales).all()
    out.to_csv(OUT, index=False)

    # sanity vs current best (80:20 + Zero28)
    changed = ~np.isclose(out.sales, best.sales, rtol=0, atol=1e-9)
    dlog = np.log1p(out.sales) - np.log1p(best.sales)
    print(f"[CHECK] rows {len(out)} | target rows {int(rows.sum())} | zero28 rows {int(zm.sum())} | "
          f"changed rows {int(changed.sum())} (changed outside target: {int((changed & ~rows).sum())})")
    print(f"[CHECK] mean sales best {best.sales.mean():.4f} new {out.sales.mean():.4f} | "
          f"mean log diff (all) {dlog.mean():+.5f} | on changed {dlog[changed].mean():+.5f} | "
          f"max |log diff| {np.abs(dlog).max():.4f}")
    meta = {"tweedie_iterations": n_iter, "iterations_source": "Fold3 early stopping (reused)" if N_ITER_ARG else "inner holdout", "weight": WEIGHT, "target_rows": int(rows.sum()),
            "excluded_families": EXCLUDED, "base": str(BASE.name), "zero28_rows": int(zm.sum())}
    OUT.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
