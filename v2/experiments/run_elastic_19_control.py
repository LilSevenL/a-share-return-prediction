"""Refit the V1 feature list under the V2 Elastic Net protocol."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import numpy as np
import pandas as pd
from run_elastic_500k import (SAMPLES, FEATURES, OUT as ELASTIC, ROOT, read, save,
                              sha, transform_features, fit_elastic_net_from_gram,
                              L1_RATIO, ALPHA_FRACTION, threadpool_limits)
from compare_elastic_versions import KEY, OUT as PRIOR
from strict_execution_market import prepare
from holding_buffer import buffer_panel, annotate_orders
from lot_execution_simulation import simulate_released_cash
from run_continuous_backtest import audit_ledger

OUT = ELASTIC.parent / "elastic_19_control"


def main():
    OUT.mkdir(exist_ok=True)
    prior = read(PRIOR/"summary.json")
    for path, digest in prior["inputs"].items():
        from pathlib import Path
        assert sha(Path(path)) == digest
    names = read(ROOT/"quant_data/processed/final_models_2011_2024/final_training_manifest.json")["configs"]["linear_baseline"]["factors"]
    assert len(set(names)) == len(names) == 19
    parts, daily = [], []
    for year in range(2017,2024):
        fold = f"validation_{year}"
        folder = SAMPLES/fold
        sample = read(folder/"sample.json")
        for letter in ["x","y"]:
            assert sha(folder/f"sample_{letter}.npy")==sample[f"{letter}_sha"]
        stats = read(ELASTIC/fold/"full_elastic_model.json")["preprocessing"]
        cols = [stats["active_names"].index(n) for n in names]
        raw = [stats["names"].index(n) for n in names]
        small = {k:[stats[k][i] for i in raw] for k in ["median","lower","upper","mean","std","active"]}
        small["names"] = names
        x = np.load(folder/"sample_x.npy",mmap_mode="r")[:,cols].astype(float)
        y = np.load(folder/"sample_y.npy")
        assert x.shape == (500000,19) and len(y)==len(x)
        mx = x.mean(axis=0)
        x -= mx
        gram = x.T@x/len(x)
        rhs = x.T@(y-y.mean())/len(x)
        alpha = ALPHA_FRACTION*np.max(abs(rhs))/L1_RATIO
        model = fit_elastic_net_from_gram(dict(gram=gram,rhs=rhs,mean_x=mx,mean_y=y.mean()),alpha,L1_RATIO,np.zeros(19),30000,1e-10)
        coef = model["coefficient"]
        grad = gram@coef-rhs+alpha*(1-L1_RATIO)*coef
        error = np.where(coef!=0,abs(grad+alpha*L1_RATIO*np.sign(coef)),np.maximum(abs(grad)-alpha*L1_RATIO,0))
        assert model["converged"] and max(error)<1e-8
        save(OUT/f"{year}_model.json",dict(features=names,preprocessing=small,alpha=alpha,l1_ratio=L1_RATIO,alpha_fraction=ALPHA_FRACTION,coefficient=coef.tolist(),intercept=model["intercept"],sample=sample,max_kkt_error=max(error)))
        chunks=[]
        for path in sorted((FEATURES/f"feature_year={year}").glob("*.parquet")):
            frame=pd.read_parquet(path,columns=KEY+names)
            pred=frame[KEY].copy()
            pred["prediction"]=transform_features(frame,small).astype(float)@coef+model["intercept"]
            chunks.append(pred)
        controls=pd.read_parquet(ELASTIC/fold/"predictions.parquet",columns=KEY+["y","volatility_20d","full_elastic","compact_elastic"])
        frame=pd.concat(chunks).merge(controls,on=KEY,validate="one_to_one")
        assert len(frame)==len(controls) and np.isfinite(frame.prediction).all()
        scored=frame.loc[np.isfinite(frame.y)]
        ranks=scored.groupby("signal_date")[["prediction","compact_elastic","full_elastic","y"]].rank()
        for day,g in ranks.groupby(scored.signal_date):
            if len(g)>=20:
                daily.append(dict(year=year,date=day,rows=len(g),**g.iloc[:,:3].corrwith(g.y).to_dict()))
        part=frame[KEY+["prediction","volatility_20d"]]
        part.to_parquet(OUT/f"{year}_predictions.parquet",index=False)
        parts.append(part)
        print(year,"trained and scored",flush=True)
    ic=pd.DataFrame(daily)
    ic.to_parquet(OUT/"daily_ic.parquet",index=False)
    np.testing.assert_allclose(ic[["compact_elastic","full_elastic"]].mean(),[prior["rank_ic"][n] for n in ["compact_elastic","full_elastic"]])
    orders,_,panel,missing,evidence=prepare(pd.concat(parts,ignore_index=True))
    del evidence,parts
    orders["entry_tradable"]=orders.entry_tradable.fillna(False)&orders.strict_buy
    selected,decision=buffer_panel(orders,panel,100)
    results={}
    for impact in [.5,1.]:
        metrics,nav,trades=simulate_released_cash(orders,selected,aum=1e8,cost_bps=20,impact_coefficient=impact,round_lots=True)
        trades=annotate_orders(trades,decision,panel,orders)
        audit=audit_ledger(trades,nav,orders,panel,missing)
        assert audit["filled_entry_missing_limits"]==audit["sold_exit_missing_limits"]==0
        np.testing.assert_allclose(nav.nav,np.cumprod(1+nav.daily_return))
        nav.to_parquet(OUT/f"{impact:g}_nav.parquet",index=False)
        trades.to_parquet(OUT/f"{impact:g}_orders.parquet",index=False)
        results[impact]=dict(**metrics,audit=audit)
        print(impact,metrics["net_return"],flush=True)
    lines=["# Elastic Net 因子集合对照","","2017–2023 历史开发期。每年相同50万训练行、等权、相同V2预处理。L1比例0.5，alpha取各集合训练样本全零临界值的10%，绝对惩罚随集合变化。19项均通过七折可用性检查，其中 turnover_zscore_20d 使用修复后的市场日历窗口。333及448项复用已训练模型及同规则回测。买入前50，五日复核保留前100，1亿元，严格成交，20bp往返基础费。","","| 因子集合 | 日均Rank IC | 原冲击累计收益 | 双倍冲击累计收益 | 双倍冲击最大回撤 |","|---|---:|---:|---:|---:|"]
    for n,label in [("prediction","19"),("compact_elastic","333"),("full_elastic","448")]:
        data=results if n=="prediction" else {r["impact"]:r for r in prior["results"] if r["model"]==n}
        lines.append(f"| {label} | {ic[n].mean():.5f} | {data[.5]['net_return']:.2%} | {data[1.]['net_return']:.2%} | {data[1.]['max_drawdown']:.2%} |")
    lines+=["","| 年份 | 19项 | 333项 | 448项 |","|---|---:|---:|---:|"]
    for year,g in ic.groupby("year"):
        lines.append(f"| {year} | "+" | ".join(f"{g[n].mean():.5f}" for n in ["prediction","compact_elastic","full_elastic"])+" |")
    lines+=["","19项名单来自历史开发中筛选的V1配置。训练使用V2标签和预处理，本表比较固定训练规则下的因子集合效果。"]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    save(OUT/"summary.json",dict(features=names,results=results,days=len(ic),rows=int(ic.rows.sum())))

if __name__=="__main__":
    with threadpool_limits(limits=4):
        main()
