"""Rescore saved rolling Elastic signals and trade all three on identical rules."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import numpy as np
import pandas as pd
from run_expanded_backtest import DATA, save, sha
from strict_execution_market import prepare
from holding_buffer import buffer_panel, annotate_orders
from lot_execution_simulation import simulate_released_cash
from run_continuous_backtest import audit_ledger

OUT = DATA / "v2/elastic_versions_500k"
MODELS = ["linear_prediction","full_elastic","compact_elastic"]
KEY = ["ts_code","signal_date"]


def main():
    OUT.mkdir(exist_ok=True)
    parts, daily, hashes = [], [], {}
    for year in range(2017,2024):
        old = DATA / f"validation_signals/validation_{year}.parquet"
        new = DATA / f"v2/elastic_500k/validation_{year}/predictions.parquet"
        for path in [old,new]:
            hashes[str(path)] = sha(path)
        a = pd.read_parquet(old,columns=KEY+[MODELS[0]])
        b = pd.read_parquet(new,columns=KEY+MODELS[1:]+["y","volatility_20d"])
        a.signal_date = pd.to_datetime(a.signal_date)
        b.signal_date = pd.to_datetime(b.signal_date)
        frame = a.merge(b,on=KEY,validate="one_to_one")
        assert len(frame)==len(a)==len(b) and np.isfinite(frame[MODELS]).all().all()
        scored = frame.loc[np.isfinite(frame.y)]
        ranks = scored.groupby("signal_date")[MODELS+["y"]].rank()
        for day, group in ranks.groupby(scored.signal_date):
            if len(group)>=20:
                daily.append(dict(year=year,date=day,rows=len(group),**group[MODELS].corrwith(group.y).to_dict()))
        parts.append(frame.drop(columns="y"))
    ic = pd.DataFrame(daily)
    assert np.isfinite(ic[MODELS]).all().all()
    ic.to_parquet(OUT/"daily_ic.parquet",index=False)
    for name in MODELS[1:]:
        from run_expanded_backtest import read
        for year, group in ic.groupby("year"):
            ref=read(DATA/f"v2/elastic_500k/validation_{year}/results.json")
            np.testing.assert_allclose(group[name].mean(),next(r["mean_rank_ic"] for r in ref if r["name"]==name),atol=1e-12)
    print(ic[MODELS].mean(),flush=True)
    frame=pd.concat(parts,ignore_index=True)
    del parts
    template, _, panel, missing, evidence = prepare(frame[KEY+["volatility_20d",MODELS[0]]].rename(columns={MODELS[0]:"prediction"}))
    del evidence
    template=template.drop(columns="prediction")
    template["entry_tradable"]=template.entry_tradable.fillna(False)&template.strict_buy
    results=[]
    for name in MODELS:
        orders=template.merge(frame[KEY+[name]],on=KEY,validate="one_to_one").rename(columns={name:"prediction"})
        assert len(orders)==len(template)
        selected,decision=buffer_panel(orders,panel,100)
        for impact in [.5,1.]:
            m,nav,trades=simulate_released_cash(orders,selected,aum=1e8,cost_bps=20,impact_coefficient=impact,round_lots=True)
            trades=annotate_orders(trades,decision,panel,orders)
            audit=audit_ledger(trades,nav,orders,panel,missing)
            assert audit["filled_entry_missing_limits"]==audit["sold_exit_missing_limits"]==0
            np.testing.assert_allclose(nav.nav,np.cumprod(1+nav.daily_return))
            stem=f"{name}_{impact:g}"
            nav.to_parquet(OUT/f"{stem}_nav.parquet",index=False)
            trades.to_parquet(OUT/f"{stem}_orders.parquet",index=False)
            results.append(dict(model=name,impact=impact,audit=audit,**m))
            print(stem,m["net_return"],flush=True)
    lines=["# Elastic Net 两版逐年预测与交易对照","","2017–2023 历史开发期。V1 使用逐年 linear_prediction（selected_at_least_6），两版按股票和信号日期一一匹配。Rank IC 使用相同的 V2 五日原始收益标签，每日截面至少20只。交易使用全部信号，含期末缺失标签的行。","",f"评分 {int(ic.rows.sum()):,} 行，共 {len(ic)} 天。","","| 年份 | V1 | V2完整 | V2精简 |","|---|---:|---:|---:|"]
    for year,group in [*ic.groupby("year"),("日均",ic)]:
        lines.append(f"| {year} | "+" | ".join(f"{group[n].mean():.5f}" for n in MODELS)+" |")
    lines+=["","初始1亿元，严格限价及买入数量约束，前50买入、每五日复核并保留前100，20bp往返基础费。累计收益覆盖七年。","","| 模型 | 冲击系数 | 累计净收益 | 最大回撤 |","|---|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['impact']} | {r['net_return']:.2%} | {r['max_drawdown']:.2%} |")
    lines+=["","因子集合、训练标签门控、样本权重和正则化设置随版本变化。差异反映整套模型配置。两版候选和持有规则均经过历史开发筛选，冲击参数为研究假设。"]
    save(OUT/"summary.json",dict(inputs=hashes,rank_ic=ic[MODELS].mean().to_dict(),results=results))
    (OUT/"COMPARISON.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
