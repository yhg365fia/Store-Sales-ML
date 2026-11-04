"""Trial17 plus targeted EMA16/EMA28 or lingerie calendar predictors.
Three separate experiments, each with Fold1-3 and August 2016 backtest.
Requires existing ROOT src/data/results. No other experiment script required.
No early stopping, tuning or submission. Not an independent holdout (2017 CV settings).
--check-only validates features and cached references without training.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import inspect
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

ROOT = Path(r'C:\projects\Store Sales-ML')
CACHE = ROOT / 'results/base23_lag364_lag28_optuna/20261028_182041_683679/cv_cache/07590e4e8d7e52dbe5a3'
sys.path.insert(0, str(ROOT / 'src'))
from main import main as original_pipeline
from feature_engineering import add_lag_features

BASE23 = [
    "store_nbr", "family", "city", "state", "type", "cluster",
    "year", "month", "day", "dayofweek", "is_weekend",
    "onpromotion", "is_promotion", "holiday_count",
    "event_christmas", "event_new_year", "event_carnival",
    "event_mothers_day", "event_independence_day", "event_city_foundation",
    "event_good_friday", "event_labor_day", "event_other",
]

LAGS = (364, 28)

EXPERIMENT = "base23_lag364_lag28"

FEATURES = BASE23 + ["sales_lag_364", "sales_lag_28"]

CATEGORICAL = ["store_nbr", "family", "city", "state", "type", "cluster"]

KEYS = ["store_nbr", "family"]

ROW_KEYS = ["date"] + KEYS

TARGET = "sales_log"

HORIZON = 16

EXPECTED_TRAIN_ROWS = 3_008_016

EXPECTED_VALID_ROWS = 28_512

N_SERIES = 54 * 33

N_ESTIMATORS = 8_000

EARLY_STOPPING_ROUNDS = 150

FOLDS = [
    dict(name="Fold 1", train_end="2017-06-28",
         valid_start="2017-06-29", valid_end="2017-07-14"),
    dict(name="Fold 2", train_end="2017-07-14",
         valid_start="2017-07-15", valid_end="2017-07-30"),
    dict(name="Fold 3", train_end="2017-07-30",
         valid_start="2017-07-31", valid_end="2017-08-15"),
]

AUGUST_BACKTEST = dict(
    name="August 2016 Backtest",
    train_end="2016-08-15",
    valid_start="2016-08-16",
    valid_end="2016-08-31",
)

SPLITS = FOLDS + [AUGUST_BACKTEST]

def check_configuration() -> None:
    if len(BASE23) != 23 or len(set(BASE23)) != 23:
        raise ValueError("BASE23 must have 23 unique columns")
    if len(FEATURES) != 25 or len(set(FEATURES)) != 25:
        raise ValueError("Expected exactly Base23 + lag364 + lag28")
    if FEATURES != BASE23 + ["sales_lag_364", "sales_lag_28"]:
        raise ValueError("Unexpected feature configuration")

    for split in SPLITS:
        cutoff = pd.Timestamp(split["train_end"])
        start = pd.Timestamp(split["valid_start"])
        stop = pd.Timestamp(split["valid_end"])
        if start != cutoff + pd.Timedelta(days=1):
            raise ValueError(f"Nonadjacent cutoff: {split['name']}")
        if (stop - start).days + 1 != HORIZON:
            raise ValueError(f"Validation must be 16 days: {split['name']}")
        for lag in LAGS:
            if stop - pd.Timedelta(days=lag) > cutoff:
                raise ValueError(f"Future target would enter lag{lag}: {split['name']}")

def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    check_configuration()
    required = ["date", "sales"] + BASE23
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"main() missing required Train columns: {missing}")

    df = raw.loc[:, required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["sales"] = pd.to_numeric(df["sales"], errors="raise")
    if (len(df) != EXPECTED_TRAIN_ROWS
            or df["date"].min() != pd.Timestamp("2013-01-01")
            or df["date"].max() != pd.Timestamp("2017-08-15")):
        raise ValueError("Expected complete restored Train 2013-01-01..2017-08-15")
    if df[KEYS + ["date"]].isna().any().any():
        raise ValueError("Null panel key or date")
    if df.duplicated(KEYS + ["date"]).any():
        raise ValueError("Duplicate store_nbr/family/date")
    if df[BASE23].isna().any().any():
        raise ValueError("Null in Base23 predictors")
    if (not np.isfinite(df["sales"].to_numpy(dtype=float)).all()
            or (df["sales"] < 0).any()):
        raise ValueError("Invalid sales targets")

    weekday = df["date"].dt.day_of_week
    if not np.array_equal(df["dayofweek"], weekday):
        raise ValueError("dayofweek inconsistent with date")
    if not np.array_equal(df["is_weekend"], (weekday >= 5).astype(int)):
        raise ValueError("is_weekend inconsistent with date")

    df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
    group_counts = df.groupby(KEYS, observed=True).size()
    if len(group_counts) != N_SERIES or group_counts.nunique() != 1:
        raise ValueError("Expected complete 1,782 store/family series")
    gaps = df.groupby(KEYS, observed=True)["date"].diff()
    if (gaps.notna() & gaps.ne(pd.Timedelta(days=1))).any():
        raise ValueError("Missing calendar dates within a store/family series")

    # Reuse the existing feature engineering helper, without changing any lags.
    df = add_lag_features(df, lags=LAGS)
    positions = df.groupby(KEYS, observed=True).cumcount()
    for lag in LAGS:
        col = f"sales_lag_{lag}"
        if col not in df.columns:
            raise ValueError(f"Existing lag helper did not create {col}")
        if not df[col].isna().equals(positions.lt(lag)):
            raise ValueError(f"Unexpected missing positions for {col}")
        if not np.isfinite(df[col].dropna().to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite values in {col}")
    df[TARGET] = np.log1p(df["sales"])
    print(f"[DATA] Restored train={len(df):,}; computed lags={LAGS}; "
          "initial lag NaNs retained.", flush=True)
    print(f"[FEATURES] {EXPERIMENT}: {len(FEATURES)} predictors", flush=True)
    print("[POLICY] Fixed-round diagnostic backtest; no early stopping.", flush=True)
    return df

def make_split(df: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(split["train_end"])
    start = pd.Timestamp(split["valid_start"])
    stop = pd.Timestamp(split["valid_end"])
    train = df.loc[df["date"].le(cutoff)]
    valid = df.loc[df["date"].between(start, stop)]
    counts = valid.groupby(KEYS, observed=True).size()
    if train.empty or train["date"].max() != cutoff:
        raise ValueError(f"Train cutoff mismatch: {split['name']}")
    if (len(valid) != EXPECTED_VALID_ROWS
            or len(counts) != N_SERIES
            or not counts.eq(HORIZON).all()
            or set(valid["date"]) != set(pd.date_range(start, stop))):
        raise ValueError(f"Incomplete 16-day validation panel: {split['name']}")
    for lag in LAGS:
        if valid[f"sales_lag_{lag}"].isna().any():
            raise ValueError(f"Missing valid lag{lag}: {split['name']}")
        if stop - pd.Timedelta(days=lag) > cutoff:
            raise ValueError(f"Leakage in lag{lag}: {split['name']}")
    return train, valid

def make_matrices(
    train: pd.DataFrame, valid: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = train.loc[:, features].copy()
    x_valid = valid.loc[:, features].copy()
    for col in CATEGORICAL:
        # Category vocabulary is determined by training rows only.
        categories = pd.Index(train[col].dropna().unique())
        x_train[col] = pd.Categorical(x_train[col], categories=categories)
        x_valid[col] = pd.Categorical(x_valid[col], categories=categories)
        if x_valid[col].isna().any():
            raise ValueError(f"Unseen validation category: {col}")
    if list(x_train.columns) != features or list(x_valid.columns) != features:
        raise ValueError("Train/Valid feature ordering changed")
    return x_train, x_valid

def rmsle(y_true, predicted_log) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(predicted_log, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite model prediction")
    return float(np.sqrt(np.mean((np.log1p(y) - np.maximum(p, 0)) ** 2)))

PARAMS = {'boosting_type': 'gbdt', 'objective': 'regression', 'learning_rate': 0.023127655409786754, 'n_estimators': 8000, 'num_leaves': 127, 'max_depth': -1, 'min_child_samples': 50, 'min_split_gain': 0.0, 'subsample': 0.7742745551930794, 'subsample_freq': 1, 'colsample_bytree': 0.9621753805232081, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'metric': 'rmse', 'importance_type': 'split', 'random_state': 42, 'n_jobs': -1, 'verbosity': -1}


BASELINE_DIR=ROOT/'results/base23_lag364_lag28_optuna/20261028_182041_683679/best_trial_0017'

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def load_reference(split,spec):
    if split=='backtest':
        directory=BASELINE_DIR
        stem='august_2016_backtest'
    else:
        directory=CACHE
        stem=f'fold_{split[-1]}'
    meta=read_json(directory/f'{stem}.json')
    expected_params=dict(PARAMS,n_estimators=6770) if split=='backtest' else PARAMS
    if meta['features']!=FEATURES or meta['parameters']!=expected_params or meta['split']!=spec:
        raise ValueError(f'{split}: baseline configuration mismatch')
    rounds=6770 if split=='backtest' else int(meta['best_iteration'])
    if rounds<=0: raise ValueError('Invalid training rounds')
    p=pd.read_csv(directory/f'{stem}_predictions.csv',parse_dates=['date'])
    if len(p)!=28512 or p.duplicated(ROW_KEYS).any():
        raise ValueError('Invalid baseline keys')
    if not np.isfinite(p[['y_true','pred_sales']]).all().all() or p[['y_true','pred_sales']].lt(0).any().any():
        raise ValueError('Invalid baseline values')
    expected=[.3789663835830142,.3993002542779057,.3939883127869828]
    expected_score=.7477345821606495 if split=='backtest' else expected[int(split[-1])-1]
    if not np.isclose(rmsle(p.y_true,np.log1p(p.pred_sales)),expected_score,atol=1e-9,rtol=0):
        raise ValueError('Baseline score mismatch')
    return p[ROW_KEYS+['y_true','pred_sales']].rename(columns={'y_true':'sales_ref','pred_sales':'trial17_pred'}),rounds

TARGET_FAMILIES=['LINGERIE','GROCERY II']
EXPERIMENTS={
 'ema16':dict(families=TARGET_FAMILIES,extra=['target_ema16_shift16']),
 'ema28':dict(families=TARGET_FAMILIES,extra=['target_ema28_shift16']),
 'lingerie_calendar':dict(families=['LINGERIE'],extra=['lingerie_month_start5','lingerie_days_to_month_end'])}

def add_experiment_features(df):
    selected=df.family.isin(TARGET_FAMILIES)
    for span in [16,28]:
        col=f'target_ema{span}_shift16'
        df[col]=df.groupby(KEYS,observed=True).sales.transform(
            lambda s:s.shift(16).ewm(span=span,adjust=False,min_periods=span).mean()).where(selected)
    mask=df.family.eq('LINGERIE')
    df['lingerie_month_start5']=df.date.dt.day.le(5).astype(float).where(mask)
    df['lingerie_days_to_month_end']=(df.date.dt.days_in_month-df.date.dt.day).astype(float).where(mask)
    return df

def verify_features(df,spec):
    cutoff=pd.Timestamp(spec['train_end'])
    start,stop=pd.Timestamp(spec['valid_start']),pd.Timestamp(spec['valid_end'])
    if stop-pd.Timedelta(days=16)>cutoff: raise ValueError('Future sales access')
    for _,series in df[df.family.isin(TARGET_FAMILIES)].groupby(KEYS,observed=True):
        hist=series[series.date<=cutoff].set_index('date').sales
        valid=series[series.date.between(start,stop)]
        for span in [16,28]:
            expected=hist.ewm(span=span,adjust=False,min_periods=span).mean().reindex(valid.date-pd.Timedelta(days=16))
            if not np.allclose(expected,valid[f'target_ema{span}_shift16']):
                raise ValueError('EMA history/date mismatch')

def run():
    parser=argparse.ArgumentParser(description='Independent EMA16, EMA28 and lingerie calendar ablations against Trial17')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--cv-only',action='store_true')
    parser.add_argument('--experiment',choices=['all']+list(EXPERIMENTS),default='all')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'group4_feature_experiments')
    args=parser.parse_args()
    output=args.output_dir.resolve()
    experiments=EXPERIMENTS if args.experiment=='all' else {args.experiment:EXPERIMENTS[args.experiment]}
    specs=[(f'fold{i}',s) for i,s in enumerate(FOLDS,1)]
    if not args.cv_only: specs.append(('backtest',AUGUST_BACKTEST))
    refs={split:load_reference(split,spec) for split,spec in specs}
    os.chdir(ROOT)
    raw,test,holidays=original_pipeline()
    df=add_experiment_features(prepare_features(raw))
    del raw,test,holidays
    gc.collect()
    for split,spec in specs:
        verify_features(df,spec)
        _,valid=make_split(df,spec)
        p=valid[ROW_KEYS+['sales']].merge(refs[split][0],on=ROW_KEYS,how='left',validate='one_to_one')
        if len(p)!=28512 or not np.allclose(p.sales,p.sales_ref) or p.trial17_pred.isna().any():
            raise ValueError('Cached baseline alignment failed')
    print('[CHECK OK] Trial17 references and leakage checks passed.',flush=True)
    if args.check_only:return
    out=output/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    cfg=dict(experiments=experiments,base_features=FEATURES,params=PARAMS,
        rounds={s:r[1] for s,r in refs.items()},training='All families; added features masked to target families; only target predictions replace Trial17',
        policy='Three independent experiments. No EMA/calendar combination. Same original Trial17 rounds, no early stopping or tuning.',
        shift=16,oil=False,lightgbm_version=lightgbm.__version__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    reports=[]
    for split,spec in specs:
        train,valid=make_split(df,spec)
        reference,rounds=refs[split]
        for name,setting in experiments.items():
            folder=out/name
            folder.mkdir(exist_ok=True)
            features=FEATURES+setting['extra']
            x_train,x_valid=make_matrices(train,valid,features)
            model=LGBMRegressor(**dict(PARAMS,n_estimators=rounds))
            print(f'[FIT] {name}/{split}, {rounds} rounds, {len(features)} features',flush=True)
            model.fit(x_train,train[TARGET],categorical_feature=CATEGORICAL)
            pred=np.asarray(model.predict(x_valid,num_iteration=rounds))
            if not np.isfinite(pred).all():raise ValueError('Nonfinite predictions')
            p=valid[ROW_KEYS+['sales','onpromotion']+setting['extra']].copy()
            p['model_pred']=np.expm1(np.maximum(pred,0))
            p=p.merge(reference,on=ROW_KEYS,how='left',validate='one_to_one')
            mask=p.family.isin(setting['families'])
            assert mask.sum()==864*len(setting['families'])
            p['candidate_pred']=np.where(mask,p.model_pred,p.trial17_pred)
            assert np.array_equal(p.loc[~mask,'candidate_pred'],p.loc[~mask,'trial17_pred'])
            p['split']=split
            p['experiment']=name
            p['horizon_day']=(p.date-pd.Timestamp(spec['valid_start'])).dt.days+1
            for label,col in [('baseline','trial17_pred'),('candidate','candidate_pred')]:
                p[label+'_log_error']=np.log1p(p[col])-np.log1p(p.sales)
                p[label+'_squared_log_error']=p[label+'_log_error']**2
            p['delta_squared_log_error']=p.candidate_squared_log_error-p.baseline_squared_log_error
            p.to_csv(folder/f'{split}_predictions.csv',index=False)
            model.booster_.save_model(str(folder/f'{split}_model.txt'))
            pd.DataFrame({'feature':features,'gain':model.booster_.feature_importance(importance_type='gain')}).to_csv(folder/f'{split}_importance.csv',index=False)
            records=[]
            partitions=[('overall','ALL',p),('target','TARGET',p[mask])]
            for scope in ['family','store_nbr','date','horizon_day']:
                partitions.extend((scope,str(key),b) for key,b in p[mask].groupby(scope,observed=True))
            for scope,key,b in partitions:
                base=np.sqrt(b.baseline_squared_log_error.mean())
                new=np.sqrt(b.candidate_squared_log_error.mean())
                records.append(dict(experiment=name,split=split,scope=scope,name=key,rows=len(b),trial17_rmsle=base,new_rmsle=new,delta=new-base,
                    baseline_log_bias=b.baseline_log_error.mean(),candidate_log_bias=b.candidate_log_error.mean(),actual_sum=b.sales.sum(),pred_sum=b.candidate_pred.sum()))
            scores=pd.DataFrame(records)
            scores.to_csv(folder/f'{split}_scores.csv',index=False)
            reports.append(scores)
            pd.concat(reports,ignore_index=True).to_csv(out/'all_scores.csv',index=False)
            print(scores[scores.scope.isin(['overall','family'])].round(6).to_string(index=False),flush=True)
            del model,x_train,x_valid,p
            gc.collect()
    result=pd.concat(reports,ignore_index=True)
    mean=result[(result.split!='backtest')&result.scope.isin(['overall','target','family'])].groupby(['experiment','scope','name'])[['trial17_rmsle','new_rmsle','delta']].mean().reset_index()
    mean.to_csv(out/'cv_mean_scores.csv',index=False)
    print('\n[CV MEAN]\n'+mean.round(6).to_string(index=False))
    print(f'\nSaved to: {out}')

if __name__=='__main__':run()

