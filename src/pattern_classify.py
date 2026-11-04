"""분류 보류 4개 품목 상세 확인. 학습/튜닝 없이 원본과 저장 예측만 읽음.

실행: python inspect_unclassified_families.py
ROOT/CACHE만 자신의 경로에 맞게 변경. pandas, numpy, matplotlib 필요.
사분위 기준은 반드시 전체 33개 품목으로 계산한 뒤 4개만 출력.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r'C:\projects\Store Sales-ML')
CACHE = ROOT / 'results/base23_lag364_lag28_optuna/20261028_182041_683679/cv_cache/07590e4e8d7e52dbe5a3'
OUT = Path(__file__).resolve().parent / 'residual_outputs/unclassified_detail'
TARGETS = ['GROCERY II', 'LINGERIE', 'HOME AND KITCHEN I', 'LADIESWEAR']
MIN_BASE_MEAN = 1.0


def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / 'data/train.csv',
                        usecols=['date', 'store_nbr', 'family', 'sales'], parse_dates=['date'])
    if train.duplicated(['date', 'store_nbr', 'family']).any():
        raise ValueError('train 키 중복')
    if not np.isfinite(train.sales).all() or train.sales.lt(0).any():
        raise ValueError('train 판매량 오류')
    daily = train.groupby(['date', 'family']).sales.sum().unstack('family')
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()))
    changes = np.log1p(daily).diff(7).abs()
    parts = []
    for fold in (1, 2, 3):
        p = pd.read_csv(CACHE / f'fold_{fold}_predictions.csv', parse_dates=['date'])
        p = p[['date', 'store_nbr', 'family', 'y_true', 'pred_sales']].copy()
        if p.duplicated(['date', 'store_nbr', 'family']).any():
            raise ValueError(f'Fold{fold} 키 중복')
        vals = p[['y_true', 'pred_sales']].to_numpy()
        if not np.isfinite(vals).all() or (vals < 0).any():
            raise ValueError(f'Fold{fold} 예측/정답 오류')
        p['fold'] = fold
        p['sqlog'] = (np.log1p(p.y_true) - np.log1p(p.pred_sales)) ** 2
        parts.append(p)
    pred = pd.concat(parts, ignore_index=True)
    truth = train.set_index(['date', 'store_nbr', 'family']).sales.rename('source_sales')
    checked = pred.join(truth, on=['date', 'store_nbr', 'family'], validate='many_to_one')
    if not np.allclose(checked.y_true, checked.source_sales):
        raise ValueError('검증 정답과 원본이 일치하지 않음')
    if not set(TARGETS).issubset(daily.columns):
        raise ValueError('대상 품목 누락')

    records, sizes = [], []
    for fold, p in pred.groupby('fold'):
        start, end = p.date.min(), p.date.max()
        cutoff = start - pd.Timedelta(days=1)
        year_start = start - pd.Timedelta(days=365)
        vol = changes.loc[year_start:cutoff].mean()
        q50, q75 = vol.quantile([.5, .75])
        recent = daily.loc[start - pd.Timedelta(days=7):cutoff]
        base = daily.loc[start - pd.Timedelta(days=35):start - pd.Timedelta(days=8)]
        overall = np.sqrt(p.sqlog.mean())
        for family in TARGETS:
            s = daily[family].loc[year_start:cutoff]
            raw = train.loc[train.family.eq(family) & train.date.between(year_start, cutoff)]
            v = p.loc[p.family.eq(family)]
            b, r = base[family].mean(), recent[family].mean()
            valid = bool(base[family].count() == 28 and recent[family].count() == 7
                         and b >= MIN_BASE_MEAN)
            ratio = r / b if valid else np.nan
            rmsle = np.sqrt(v.sqlog.mean())
            records.append(dict(
                family=family, fold=fold, cutoff=cutoff.date(),
                validation_start=start.date(), validation_end=end.date(),
                volatility=vol[family], q50=q50, q75=q75,
                high_volatility=bool(vol[family] >= q75),
                low_volatility=bool(vol[family] <= q50),
                recent_7d_mean=r, preceding_28d_mean=b,
                recent_ratio=ratio, ratio_valid=valid,
                recent_abrupt=bool(valid and (ratio >= 1.5 or ratio <= 2/3)),
                family_rmsle=rmsle, fold_rmsle=overall, well_predicted=bool(rmsle <= overall),
                validation_actual=v.y_true.sum(), validation_predicted=v.pred_sales.sum(),
                validation_shortfall=v.y_true.sum()-v.pred_sales.sum(),
            ))
            sizes.append(dict(
                family=family, fold=fold, history_start=year_start.date(), cutoff=cutoff.date(),
                stores=raw.store_nbr.nunique(), observations=len(raw),
                observed_days=s.count(), missing_days=s.isna().sum(),
                zero_sales_row_pct=raw.sales.eq(0).mean()*100,
                total_sales=raw.sales.sum(), daily_mean=s.mean(), daily_median=s.median(),
                daily_std=s.std(), daily_min=s.min(), daily_max=s.max(),
                daily_p10=s.quantile(.1), daily_p90=s.quantile(.9),
                daily_p90_p10=s.quantile(.9)-s.quantile(.1),
                mean_abs_daily_change=s.diff().abs().mean(),
                max_abs_daily_change=s.diff().abs().max(),
            ))
    metrics, stats = pd.DataFrame(records), pd.DataFrame(sizes)
    save(metrics, 'conditions_by_fold.csv')
    save(stats, 'sales_stats_by_fold.csv')

    decisions = []
    report = [
        '보류 4개 품목 상세 진단',
        '상품 SKU 수는 데이터로 확인할 수 없음. 매장 수/관측 수/판매 수량을 별도로 출력.',
        '판매 변동은 전체 매장 일별 합계, RMSLE는 매장별 관측 행으로 계산.',
        '통계는 각 Fold 시작 전 1년. 최근 비율은 직전 7일 / 그 이전 28일.',
        '사분위는 전체 품목 기준. 누락 날짜는 0으로 채우지 않음.',
        '판정 순서: 3번 → 1번 → 2번. 수동 확인 이전의 수치 후보만 판정.',
        '전체 판매 추이 그래프는 사후 진단용이며 검증 이후 정보도 포함됨.\n',
    ]
    for family in TARGETS:
        fm = metrics.loc[metrics.family.eq(family)].sort_values('fold')
        g3 = bool(fm.well_predicted.all())
        g1 = bool((fm.high_volatility | fm.recent_abrupt).any())
        g2 = bool(fm.low_volatility.all() and not fm.recent_abrupt.any() and fm.ratio_valid.all())
        candidate = '3 후보' if g3 else '1 후보' if g1 else '2 후보' if g2 else '판단 보류'
        failures = []
        if not g3:
            failures.append('3번: 전체 RMSLE를 초과한 Fold=' + ','.join(map(str, fm.loc[~fm.well_predicted, 'fold'])))
        if not g1:
            failures.append('1번: 모든 Fold에서 상위25% 변동/최근 급변 신호 없음')
        if not g2:
            failures.append('2번: 하위50% 미충족 Fold=' + ','.join(map(str, fm.loc[~fm.low_volatility, 'fold']))
                            + '; 급변 Fold=' + ','.join(map(str, fm.loc[fm.recent_abrupt, 'fold']))
                            + '; 비율 미확인 Fold=' + ','.join(map(str, fm.loc[~fm.ratio_valid, 'fold'])))
        decisions.append(dict(family=family, numeric_candidate=candidate,
                              group3_condition=g3, group1_condition=g1, group2_condition=g2,
                              reason=' / '.join(failures)))
        report += [f'\n[{family}] {candidate}', *failures,
                   fm.round(4).to_string(index=False),
                   stats.loc[stats.family.eq(family)].round(3).to_string(index=False)]

        v = pred.loc[pred.family.eq(family)]
        vd = v.groupby(['fold', 'date']).agg(actual=('y_true', 'sum'),
                                            predicted=('pred_sales', 'sum'),
                                            rmsle=('sqlog', lambda x: np.sqrt(x.mean()))).reset_index()
        vd['shortfall'] = vd.actual - vd.predicted
        safe = re.sub(r'[^A-Za-z0-9_-]+', '_', family)
        save(vd, f'{safe}_validation_daily.csv')
        hist = daily[family].rename('actual').to_frame()
        hist['mean_28d'] = hist.actual.rolling(28, min_periods=28).mean()
        hist.index.name = 'date'
        save(hist.reset_index(), f'{safe}_history_daily.csv')

        fig, axes = plt.subplots(3, 2, figsize=(17, 12), constrained_layout=True)
        ax = axes[0, 0]
        ax.plot(hist.index, hist.actual, alpha=.3, color='gray', label='Daily total')
        ax.plot(hist.index, hist.mean_28d, color='navy', label='28-day trailing mean')
        ax.axvspan(pred.date.min(), pred.date.max(), alpha=.15, color='orange', label='Validation')
        ax.set_title('Full history: descriptive only'); ax.legend()
        ax = axes[0, 1]
        for fold, block in vd.groupby('fold'):
            ax.plot(block.date, block.actual, color='black', label='Actual' if fold == 1 else None)
            ax.plot(block.date, block.predicted, color='tomato', label='Predicted' if fold == 1 else None)
            ax.axvline(block.date.min(), color='gray', linestyle=':')
        ax.set_title('Validation daily totals'); ax.legend()
        ax = axes[1, 0]
        ax.plot(fm.fold, fm.volatility, 'o-', label='Family volatility')
        ax.plot(fm.fold, fm.q50, '--', label='All-family median')
        ax.plot(fm.fold, fm.q75, '--', label='All-family 75th percentile')
        ax.set_title('Short-term variability: mean absolute 7-day log difference'); ax.legend()
        ax = axes[1, 1]
        ax.bar(fm.fold, fm.recent_ratio, color='steelblue')
        ax.axhline(1.5, color='red', linestyle='--', label='Abrupt-change thresholds')
        ax.axhline(2/3, color='red', linestyle='--')
        ax.axhline(1, color='gray', linestyle=':')
        ax.set_title('Before each Fold: recent 7d / preceding 28d'); ax.legend()
        ax = axes[2, 0]
        ax.bar(fm.fold-.18, fm.family_rmsle, width=.36, label='Family RMSLE')
        ax.bar(fm.fold+.18, fm.fold_rmsle, width=.36, label='All-family RMSLE')
        ax.set_title('Validation error: store-level observations'); ax.legend()
        ax = axes[2, 1]
        for fold, block in vd.groupby('fold'):
            ax.plot(block.date, block.shortfall, label=f'Fold {fold}')
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_title('Daily actual - prediction: positive = underprediction'); ax.legend()
        for ax in [axes[1, 0], axes[1, 1], axes[2, 0]]:
            ax.set_xticks([1, 2, 3], ['Fold 1', 'Fold 2', 'Fold 3'])
        for ax in axes.flat:
            ax.grid(alpha=.2)
        fig.suptitle(family, fontsize=17)
        fig.savefig(OUT / f'{safe}.png', dpi=150)
        plt.close(fig)

    decisions = pd.DataFrame(decisions)
    save(decisions, 'candidate_reasons.csv')
    text = '\n'.join(report)
    (OUT / 'report.txt').write_text(text, encoding='utf-8')
    print(text)
    print(f'\n결과 저장: {OUT}')


if __name__ == '__main__':
    main()
