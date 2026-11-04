"""Saved predictions only: no-oil EMA7, BEAUTY and MAGAZINES routed to Trial17.
Recalculate Fold1/2/3 and August 2016 backtest. No retraining or submission.
Report RMSLE (primary) and raw-sales RMSE. All comparisons use NEW group labels.
"""
from pathlib import Path
from datetime import datetime
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(r'C:\projects\Store Sales-ML')
CV = ROOT / 'results/group1_lag16_ema7_mix/20261102_214811_779601'
BT = ROOT / 'src/backtest_outputs/20261103_155940_882756/predictions.csv'
GROUPS = {
    1: ['HARDWARE', 'HOME APPLIANCES', 'LAWN AND GARDEN',
        'LIQUOR,WINE,BEER', 'SCHOOL AND OFFICE SUPPLIES'],
    2: ['AUTOMOTIVE', 'CELEBRATION', 'HOME AND KITCHEN II', 'PET SUPPLIES',
        'PLAYERS AND ELECTRONICS', 'SEAFOOD'],
    4: ['LINGERIE', 'LADIESWEAR', 'HOME AND KITCHEN I', 'GROCERY II',
        'MAGAZINES', 'BEAUTY'],
}
SPECS = {
    'fold1': ('2017-06-29','2017-07-14',.3789663835830142, .37941303870439164),
    'fold2': ('2017-07-15','2017-07-30',.3993002542779057, .3979574032708647),
    'fold3': ('2017-07-31','2017-08-15',.39398831278698276, .3927465405495937),
    'backtest': ('2016-08-16','2016-08-31',.7477345821606495, None),
}


def scores(y, pred):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    return dict(rmsle=float(np.sqrt(np.mean((np.log1p(pred)-np.log1p(y))**2))),
                rmse=float(np.sqrt(np.mean((pred-y)**2))))


def load(split):
    is_bt = split == 'backtest'
    path = BT if is_bt else CV / f'fold_{split[-1]}_predictions.csv'
    raw = pd.read_csv(path, parse_dates=['date'])
    col = 'ema7_mix_pred_sales' if is_bt else 'new_pred_sales'
    p = raw[['date','store_nbr','family','sales','baseline_pred_sales',col]].copy()
    p = p.rename(columns={col:'ema_original'})
    start, end, expected, expected_ema = SPECS[split]
    keys = ['date','store_nbr','family']
    if (len(p) != 28512 or p[keys].isna().any().any() or p.duplicated(keys).any()
            or set(p.date) != set(pd.date_range(start,end))
            or len(p.groupby(['store_nbr','family'])) != 1782
            or not p.groupby(['store_nbr','family']).size().eq(16).all()):
        raise ValueError(f'{split}: invalid panel')
    for c in ['sales','baseline_pred_sales','ema_original']:
        if not np.isfinite(p[c]).all() or p[c].lt(0).any():
            raise ValueError(f'{split}: invalid {c}')
    if 'y_true' in raw and not np.allclose(p.sales,raw.y_true):
        raise ValueError('Truth mismatch')
    if not np.isclose(scores(p.sales,p.baseline_pred_sales)['rmsle'], expected, atol=1e-9, rtol=0):
        raise ValueError('Baseline score mismatch')
    if expected_ema is not None and not np.isclose(scores(p.sales,p.ema_original)['rmsle'],expected_ema,atol=1e-9,rtol=0):
        raise ValueError('EMA score mismatch')
    if is_bt and not np.isclose(scores(p.sales,p.ema_original)['rmsle'],.747199,atol=1e-6,rtol=0):
        raise ValueError('Backtest EMA score mismatch')
    mapping = {f:g for g,names in GROUPS.items() for f in names}
    p['analysis_group'] = p.family.map(mapping).fillna(3).astype(int)
    p['previous_magazine_only'] = np.where(p.family.eq('MAGAZINES'),p.baseline_pred_sales,p.ema_original)
    p['new_beauty_magazine'] = np.where(p.family.isin(['BEAUTY','MAGAZINES']),p.baseline_pred_sales,p.ema_original)
    if not p.groupby('analysis_group').size().to_dict() == {1:4320,2:5184,3:13824,4:5184}:
        raise ValueError('Unexpected new group sizes')
    outside=p.analysis_group.ne(1)
    if not np.array_equal(p.loc[outside,'new_beauty_magazine'],p.loc[outside,'baseline_pred_sales']):
        raise ValueError('Baseline routing mismatch')
    unchanged=~p.family.eq('BEAUTY')
    if not np.array_equal(p.loc[unchanged,'new_beauty_magazine'],p.loc[unchanged,'previous_magazine_only']):
        raise ValueError('Unexpected change outside BEAUTY')
    p['split']=split
    return p, path


def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'ema7_beauty_magazines_results')
    args=parser.parse_args()
    # Validate every input before writing any output.
    loaded={split:load(split) for split in SPECS}
    out=args.output_dir.resolve()/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    records=[]
    for split,(p,path) in loaded.items():
        partitions=[('overall','ALL',p)]
        partitions += [('group',str(g),b) for g,b in p.groupby('analysis_group')]
        partitions += [('family',str(f),b) for f,b in p.groupby('family')]
        for scope,name,b in partitions:
            for model,col in [('trial17','baseline_pred_sales'),('ema_original_7','ema_original'),
                              ('magazine_only','previous_magazine_only'),('beauty_magazine','new_beauty_magazine')]:
                records.append(dict(split=split,scope=scope,name=name,model=model,rows=len(b),**scores(b.sales,b[col])))
        p.to_csv(out/f'{split}_predictions.csv',index=False,encoding='utf-8-sig')
    result=pd.DataFrame(records)
    result.to_csv(out/'all_scores.csv',index=False,encoding='utf-8-sig')
    refs=['trial17','ema_original_7','magazine_only']
    keys=['split','scope','name','rows']
    new=result.loc[result.model.eq('beauty_magazine')].drop(columns='model')
    comparisons=[]
    for ref in refs:
        old=result.loc[result.model.eq(ref)].drop(columns='model')
        c=old.merge(new,on=keys,validate='one_to_one',suffixes=('_reference','_new'))
        c['comparison']='vs_'+ref
        for metric in ['rmsle','rmse']:
            c['delta_'+metric]=c[metric+'_new']-c[metric+'_reference']
        comparisons.append(c)
    comparisons=pd.concat(comparisons,ignore_index=True)
    comparisons.to_csv(out/'all_comparisons.csv',index=False,encoding='utf-8-sig')
    overall=result.loc[result.scope.eq('overall')].pivot(index='split',columns='model',values='rmsle')
    overall=overall.reindex(['fold1','fold2','fold3','backtest'])
    overall.loc['cv_mean']=overall.loc[['fold1','fold2','fold3']].mean()
    overall=overall[['trial17','ema_original_7','magazine_only','beauty_magazine']]
    overall.to_csv(out/'overall_rmsle_summary.csv',encoding='utf-8-sig')
    cv=result.loc[result.split.ne('backtest')].groupby(['scope','name','model'])[['rmsle','rmse']].mean().reset_index()
    cv.to_csv(out/'cv_mean_scores.csv',index=False,encoding='utf-8-sig')
    metadata=dict(groups=GROUPS,sources={s:str(path) for s,(_,path) in loaded.items()},oil=False,
                  training_performed=False,metric='RMSLE primary; raw sales RMSE also reported',
                  rule='Only BEAUTY additionally switched from EMA to Trial17; MAGAZINES remains Trial17',
                  training_mask='Saved model still trained with original seven-family mask; no masking change',
                  note='Exploratory reused validation. Every comparison uses identical new group membership; CV means exclude backtest.')
    (out/'config.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n[OVERALL RMSLE]\n'+overall.round(9).to_string())
    print('\n[BEAUTY SWITCH EFFECT: negative=improvement]\n'+comparisons.loc[
        comparisons.comparison.eq('vs_magazine_only') & comparisons.scope.isin(['overall','group']),
        ['split','scope','name','rmsle_reference','rmsle_new','delta_rmsle']].round(9).to_string(index=False))
    print('\nSAVED:',out)


if __name__=='__main__':
    run()
