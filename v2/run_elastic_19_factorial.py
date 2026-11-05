"""Matched date-weight and penalty ablation for the inherited 19 factor names."""
from pathlib import Path
import numpy as np
import pandas as pd
from run_elastic_19_control import (OUT as CONTROL, ELASTIC, SAMPLES, ROOT, FEATURES, KEY,
    read, save, sha, fit_elastic_net_from_gram, transform_features, threadpool_limits,
    prepare, buffer_panel, annotate_orders, simulate_released_cash, audit_ledger)

OUT = CONTROL.parent / "elastic_19_factorial"


def main():
    OUT.mkdir(exist_ok=True)
    cfg = read(ROOT/"quant_data/processed/final_models_2011_2024/final_training_manifest.json")["configs"]["linear_baseline"]
    rules = {"v2":(.5,.1), "v1":(cfg["l1_ratio"],cfg["alpha_ratio"])}
    groups = [f"{weight}_{rule}" for weight in ["uniform","date"] for rule in rules]
    parts, daily = [], []
    for year in range(2017,2024):
        fold=f"validation_{year}"
        folder=SAMPLES/fold
        control=read(CONTROL/f"{year}_model.json")
        names=control["features"]
        assert names==cfg["factors"]
        full=read(ELASTIC/fold/"full_elastic_model.json")["active_feature_names"]
        for letter in ["x","y"]:
            assert sha(folder/f"sample_{letter}.npy")==control["sample"][f"{letter}_sha"]
        x=np.load(folder/"sample_x.npy",mmap_mode="r")[:,[full.index(n) for n in names]].astype(float)
        y=np.load(folder/"sample_y.npy")
        keys=pd.read_parquet(folder/"sample_keys.parquet")
        assert len(x)==len(y)==len(keys)==500000 and not keys.duplicated(KEY).any()
        assert keys.equals(keys.sort_values(["signal_date","ts_code"]).reset_index(drop=True))
        dates=keys.signal_date
        counts=dates.map(dates.value_counts()).to_numpy(float)
        models={}
        for weight,gamma in [("uniform",0),("date",cfg["gamma"])]:
            w=counts**(-gamma)
            w/=w.sum()
            mx=w@x
            my=w@y
            centered=x-mx
            gram=centered.T@(centered*w[:,None])
            rhs=centered.T@(w*(y-my))
            for rule,(ratio,fraction) in rules.items():
                alpha=fraction*max(abs(rhs))/ratio
                fit=fit_elastic_net_from_gram(dict(gram=gram,rhs=rhs,mean_x=mx,mean_y=my),alpha,ratio,np.zeros(19),30000,1e-10)
                assert fit["converged"]
                beta=fit["coefficient"]
                gradient=gram@beta-rhs+alpha*(1-ratio)*beta
                assert np.max(np.where(beta!=0,abs(gradient+alpha*ratio*np.sign(beta)),np.maximum(abs(gradient)-alpha*ratio,0)))<1e-8
                models[f"{weight}_{rule}"]=dict(coefficient=beta.tolist(),intercept=fit["intercept"],alpha=alpha,l1_ratio=ratio,alpha_fraction=fraction,gamma=gamma)
        np.testing.assert_allclose(models["uniform_v2"]["coefficient"],control["coefficient"],atol=1e-10)
        save(OUT/f"{year}_models.json",dict(models=models,preprocessing=control["preprocessing"],sample=control["sample"],keys_sha=sha(folder/"sample_keys.parquet")))
        chunks=[]
        for path in sorted((FEATURES/f"feature_year={year}").glob("*.parquet")):
            data=pd.read_parquet(path,columns=KEY+names)
            matrix=transform_features(data,control["preprocessing"]).astype(float)
            block=data[KEY].copy()
            for group,model in models.items():
                block[group]=matrix@model["coefficient"]+model["intercept"]
            chunks.append(block)
        labels=pd.read_parquet(ELASTIC/fold/"predictions.parquet",columns=KEY+["y","volatility_20d"])
        frame=pd.concat(chunks).merge(labels,on=KEY,validate="one_to_one")
        assert len(frame)==len(labels) and np.isfinite(frame[groups]).all().all()
        scored=frame.loc[np.isfinite(frame.y)]
        ranks=scored.groupby("signal_date")[groups+["y"]].rank()
        for day,g in ranks.groupby(scored.signal_date):
            if len(g)>=20:
                daily.append(dict(year=year,date=day,**g[groups].corrwith(g.y).to_dict()))
        parts.append(frame.drop(columns="y"))
        parts[-1].to_parquet(OUT/f"{year}_predictions.parquet",index=False)
    ic=pd.DataFrame(daily)
    np.testing.assert_allclose(ic.uniform_v2,pd.read_parquet(CONTROL/"daily_ic.parquet").prediction,atol=1e-8)
    ic.to_parquet(OUT/"daily_ic.parquet",index=False)
    print(ic[groups].mean(),flush=True)
    frame=pd.concat(parts,ignore_index=True)
    del parts
    template,_,panel,missing,evidence=prepare(frame[KEY+["volatility_20d","uniform_v2"]].rename(columns={"uniform_v2":"prediction"}))
    del evidence
    template=template.drop(columns="prediction")
    template["entry_tradable"]=template.entry_tradable.fillna(False)&template.strict_buy
    results={"uniform_v2":read(CONTROL/"summary.json")["results"]}
    for group in groups[1:]:
        orders=template.merge(frame[KEY+[group]],on=KEY,validate="one_to_one").rename(columns={group:"prediction"})
        assert len(orders)==len(template)
        selected,decision=buffer_panel(orders,panel,100)
        results[group]={}
        for impact in [.5,1.]:
            metrics,nav,trades=simulate_released_cash(orders,selected,aum=1e8,cost_bps=20,impact_coefficient=impact,round_lots=True)
            trades=annotate_orders(trades,decision,panel,orders)
            audit=audit_ledger(trades,nav,orders,panel,missing)
            assert audit["filled_entry_missing_limits"]==audit["sold_exit_missing_limits"]==0
            np.testing.assert_allclose(nav.nav,np.cumprod(1+nav.daily_return))
            nav.to_parquet(OUT/f"{group}_{impact:g}_nav.parquet",index=False)
            trades.to_parquet(OUT/f"{group}_{impact:g}_orders.parquet",index=False)
            results[group][str(impact)]=dict(**metrics,audit=audit)
            print(group,impact,metrics["net_return"],flush=True)
    lines=["# 19因子日期权重与正则化交叉实验","","2017–2023年，每折同50万行，V2标签和19项预处理。日期权重使用本折抽样后的每日股票数的−0.75次方。两套规则的（L1比例，alpha占全零临界值比例）分别为V2（0.5，0.1）、V1（0.0000001，0.2）。alpha根据当折加权训练统计重新计算。严格成交，1亿元，20bp，前50买入，前100保留。等权V2对照复核系数和每日IC后复用原回测。","","| 权重及规则 | 日均Rank IC | 原冲击累计净收益 | 双倍冲击累计净收益 |","|---|---:|---:|---:|"]
    for group in groups:
        r=results[group]
        lines.append(f"| {group} | {ic[group].mean():.5f} | {r['0.5']['net_return']:.2%} | {r['1.0']['net_return']:.2%} |")
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    save(OUT/"summary.json",dict(results=results,ic=ic[groups].mean().to_dict()))

if __name__=="__main__":
    with threadpool_limits(limits=4):
        main()
