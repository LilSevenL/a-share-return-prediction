"""Repeat the compact Elastic Net fit on the saved paired training samples."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from run_elastic_19_control import (ELASTIC, ROOT, FEATURES, KEY,
    read, save, sha, fit_elastic_net_from_gram, transform_features, threadpool_limits,
    prepare, buffer_panel, annotate_orders, simulate_released_cash, audit_ledger)
from run_elastic_500k import L1_RATIO, ALPHA_FRACTION
from feature_model import bundle_model, predict_frame

OUT = ELASTIC.parent / "elastic_compact_gate_seeds"
SEEDS = (20261025, 20261026, 20261027)
BRANCHES = ("baseline", "features_only")
GROUPS = [f"s{seed}_{branch}" for seed in SEEDS for branch in BRANCHES]
YEARS = range(2017, 2024)


def train_year(year, directory):
    source = ELASTIC/f"validation_{year}/compact_elastic_model.json"
    stats = read(source)["preprocessing"]
    names = stats["active_names"]
    cols = np.flatnonzero(stats["active"])
    assert len(cols) == 333 and [stats["names"][i] for i in cols] == names
    small = {k:[stats[k][i] for i in cols] for k in ("names","median","lower","upper","mean","std","active")}
    con = duckdb.connect()
    key_paths = {}
    samples = []
    for seed in SEEDS:
        for branch in BRANCHES:
            group = f"s{seed}_{branch}"
            if seed == SEEDS[0]:
                path = OUT.parent/"elastic_gate_ablation"/f"{year}_{branch}_keys.parquet"
            else:
                path = OUT.parent/"elastic_gate_seeds"/f"{year}_{group}_keys.parquet"
            key_paths[group] = sha(path)
            keys = pd.read_parquet(path, columns=KEY)
            assert len(keys) == 500000 and not keys.duplicated(KEY).any()
            samples.append(keys.set_index(KEY).assign(**{group: True}))
    members = pd.concat(samples, axis=1).notna().reset_index()
    con.register("members",members)
    target_paths = [str(path) for y in range(2011,year) for path in sorted(
        (ROOT/"quant_data/processed/targets_5d"/f"signal_year={y}").glob("*.parquet"))]
    con.read_parquet(target_paths,hive_partitioning=False).create_view("targets")
    pool = con.execute(f"""SELECT m.*, t.target_logret_5d_raw y FROM members m
        JOIN targets t USING(ts_code,signal_date) WHERE isfinite(t.target_logret_5d_raw)
        AND m.signal_date<DATE '{year}-01-01' AND t.exit_date<DATE '{year}-01-01'""").df()
    assert len(pool)==len(members) and not pool.duplicated(KEY).any()
    con.close()
    del samples, keys, members
    moments = {g: dict(xx=np.zeros((333,333)),xy=np.zeros(333),sx=np.zeros(333),sy=0.,n=0) for g in GROUPS}
    for (y,m), rows in pool.groupby([pool.signal_date.dt.year,pool.signal_date.dt.month]):
        path = FEATURES/f"feature_year={y}"/f"month_{m:02d}.parquet"
        raw = pd.read_parquet(path,columns=KEY+names)
        index = pd.MultiIndex.from_frame(raw[KEY])
        loc = index.get_indexer(pd.MultiIndex.from_frame(rows[KEY]))
        assert (loc>=0).all()
        matrix = transform_features(raw.iloc[loc],small)
        for group, totals in moments.items():
            mask = rows[group].to_numpy(bool)
            x = matrix[mask].astype(float)
            target = rows.y.to_numpy()[mask]
            totals["xx"] += x.T @ x
            totals["xy"] += target @ x
            totals["sx"] += x.sum(axis=0)
            totals["sy"] += target.sum()
            totals["n"] += len(x)
    models = {}
    for group,t in moments.items():
        n = t["n"]
        assert n==500000
        mx,my = t["sx"]/n,t["sy"]/n
        gram = t["xx"]/n-np.outer(mx,mx)
        rhs = t["xy"]/n-mx*my
        alpha = ALPHA_FRACTION*np.max(abs(rhs))/L1_RATIO
        fit = fit_elastic_net_from_gram(dict(gram=gram,rhs=rhs,mean_x=mx,mean_y=my),
            alpha,L1_RATIO,np.zeros(len(mx)),30000,1e-10)
        assert fit["converged"]
        beta = fit["coefficient"]
        grad = gram@beta-rhs+alpha*(1-L1_RATIO)*beta
        kkt = np.max(np.where(beta!=0,abs(grad+alpha*L1_RATIO*np.sign(beta)),np.maximum(abs(grad)-alpha*L1_RATIO,0)))
        assert kkt<1e-8
        linear = dict(coefficient=beta.tolist(),intercept=float(my-mx@beta), method="elastic",
            alpha=float(alpha), l1_ratio=L1_RATIO, alpha_fraction=ALPHA_FRACTION, weight="uniform",
            training_rows=n, keys_sha=key_paths[group], max_kkt_error=float(kkt))
        models[group] = bundle_model(linear,stats)
        save(directory/f"{group}_model.json",models[group])

    chunks = []
    for path in sorted((FEATURES/f"feature_year={year}").glob("*.parquet")):
        raw = pd.read_parquet(path)
        matrix = transform_features(raw,small).astype(float)
        chunk = raw[KEY].copy()
        for group, model in models.items():
            linear = model["linear_model"]
            chunk[group] = matrix@linear["coefficient"]+linear["intercept"]
            np.testing.assert_allclose(chunk[group].iloc[:8],predict_frame(model,raw.iloc[:8]),atol=1e-10)
        chunks.append(chunk)
    labels = pd.read_parquet(source.parent/"predictions.parquet",columns=KEY+["volatility_20d","y"])
    frame = pd.concat(chunks).merge(labels,on=KEY,validate="one_to_one")
    assert len(frame)==len(labels) and np.isfinite(frame[GROUPS]).all().all()
    scored = frame.loc[np.isfinite(frame.y)]
    ranks = scored.groupby("signal_date")[GROUPS+["y"]].rank()
    daily = [dict(year=year,date=day,**v[GROUPS].corrwith(v.y).to_dict())
             for day,v in ranks.groupby(scored.signal_date) if len(v)>=20]
    assert pd.DataFrame(daily)[GROUPS].notna().all().all()
    frame.drop(columns="y").to_parquet(directory/"predictions.parquet",index=False)
    pd.DataFrame(daily).to_parquet(directory/"daily_ic.parquet",index=False)
    save(directory/"completed.json",dict(year=year,models=len(models),rows=len(scored),days=len(daily),
        mae={g:float((scored[g]-scored.y).abs().mean()) for g in GROUPS}))


def backtest(frame):
    template, _, panel, missing, evidence = prepare(frame[KEY+["volatility_20d",GROUPS[0]]].rename(columns={GROUPS[0]:"prediction"}))
    del evidence
    template = template.drop(columns="prediction")
    template["entry_tradable"] = template.entry_tradable.fillna(False) & template.strict_buy
    results = read(OUT/"backtest_progress.json") if (OUT/"backtest_progress.json").exists() else {}
    for group in GROUPS:
        if len(results.get(group, {})) == 2:
            continue
        orders = template.merge(frame[KEY+[group]], on=KEY, validate="one_to_one").rename(columns={group:"prediction"})
        assert len(orders) == len(template)
        selected, decision = buffer_panel(orders, panel, 100)
        results.setdefault(group, {})
        for impact in [.5, 1.]:
            if str(impact) in results[group]:
                continue
            metrics, nav, trades = simulate_released_cash(orders, selected, aum=1e8, cost_bps=20, impact_coefficient=impact, round_lots=True)
            trades = annotate_orders(trades, decision, panel, orders)
            audit = audit_ledger(trades, nav, orders, panel, missing)
            assert audit["filled_entry_missing_limits"] == audit["sold_exit_missing_limits"] == 0
            np.testing.assert_allclose(nav.nav, np.cumprod(1+nav.daily_return))
            nav.to_parquet(OUT/f"{group}_{impact:g}_nav.parquet", index=False)
            trades.to_parquet(OUT/f"{group}_{impact:g}_orders.parquet", index=False)
            results[group][str(impact)] = dict(**metrics,audit=audit)
            save(OUT/"backtest_progress.json", results)
            print(group, impact, metrics["net_return"], flush=True)

    return results


def main():
    OUT.mkdir(exist_ok=True)
    for year in YEARS:
        directory = OUT/str(year)
        directory.mkdir(exist_ok=True)
        if not (directory/"completed.json").exists():
            train_year(year,directory)
        print("fold complete",year,flush=True)
    ic = pd.concat([pd.read_parquet(OUT/str(y)/"daily_ic.parquet") for y in YEARS])
    print("mean IC",ic[GROUPS].mean().to_dict(),flush=True)
    frame = pd.concat([pd.read_parquet(OUT/str(y)/"predictions.parquet") for y in YEARS],ignore_index=True)
    results = backtest(frame)
    rows, diffs = [], []
    for seed in SEEDS:
        pair = []
        for branch in BRANCHES:
            group = f"s{seed}_{branch}"
            r = results[group]
            assert all(v["audit"]["status"]=="passed" for v in r.values())
            row = dict(seed=seed,branch=branch,ic=float(ic[group].mean()),normal=r["0.5"]["net_return"],
                stress=r["1.0"]["net_return"],drawdown=r["1.0"]["max_drawdown"])
            rows.append(row)
            pair.append(row)
        diffs.append(dict(seed=seed,**{k:pair[1][k]-pair[0][k] for k in ("ic","normal","stress","drawdown")}))
    save(OUT/"summary.json",dict(results=results, rows=rows, differences=diffs,script_sha=sha(Path(__file__))))
    lines = ["# 333 因子 Elastic Net 的训练样本完整度复验","",
        "2017–2023 年滚动验证。逐行复用阶段 32、33 的三个种子及每折 50 万条样本。两组都使用精简 V2 的 333 个活跃因子、训练折预处理、等权训练和相同正则化规则（L1 比例 0.5，alpha 为当折临界值的 10%）。筛选组仅用旧 50 项中至少 30 项有效的训练行。验证及交易股票池相同。","",
        "回测采用 1 亿元初始资金、20bp 成本、前 50 买入和前 100 保留，连续跨年记账。收益为七年累计净收益。","",
        "| 种子 | 训练组 | 日均 Rank IC | 原冲击收益 | 双倍冲击收益 | 双倍冲击最大回撤 |","|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['branch']} | {r['ic']:.6f} | {r['normal']:.2%} | {r['stress']:.2%} | {r['drawdown']:.2%} |")
    lines += ["","| 种子 | 筛选带来的 IC 差 | 原冲击收益差（百分点） | 双倍冲击收益差（百分点） |","|---|---:|---:|---:|"]
    for r in diffs:
        lines.append(f"| {r['seed']} | {r['ic']:+.6f} | {100*r['normal']:+.2f} | {100*r['stress']:+.2f} |")
    lines += ["", "三组种子共享相同开发期。这次比较检验筛选在 333 因子既有配置上的表现。与 19 因子结果的差异还包括训练权重和正则化规则。"]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__":
    with threadpool_limits(limits=4):
        main()
