"""Paired date-count reference experiment on a fixed common-pool sample."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from run_elastic_19_control import (OUT as CONTROL, ELASTIC, ROOT, FEATURES, KEY,
    read, save, sha, fit_elastic_net_from_gram, transform_features, threadpool_limits,
    prepare, buffer_panel, annotate_orders, simulate_released_cash, audit_ledger)

OUT = CONTROL.parent / "elastic_common_weights"
GROUPS = ["sample_counts", "pool_counts"]
SEED = 20261025


def main():
    OUT.mkdir(exist_ok=True)
    data_root = ROOT / "quant_data/processed"
    cfg = read(data_root / "final_models_2011_2024/final_training_manifest.json")["configs"]["linear_baseline"]
    expected = {r["year"]: r["common"] for r in read(CONTROL.parent / "elastic_sample_audit/audit.json")["rows"]}
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='3GB'")
    inputs = []
    for name, root, partition in [
        ("old", data_root / "model_dataset/final_split=training", "signal_year"),
        ("features", FEATURES, "feature_year")]:
        files = [p for year in range(2011, 2023) for p in sorted((root / f"{partition}={year}").glob("*.parquet"))]
        inputs.extend(dict(path=str(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in files)
        con.read_parquet([str(p) for p in files], hive_partitioning=False).create_view(name)
    parts, daily, diagnostics = [], [], []
    for year in range(2017, 2024):
        control = read(CONTROL / f"{year}_model.json")
        names = control["features"]
        assert names == cfg["factors"] and len(names) == 19
        bound = f"{year}-01-01"
        con.execute(f"""CREATE OR REPLACE TEMP TABLE pool AS
            SELECT o.ts_code,o.signal_date,o.exit_date,o.target_logret_5d y
            FROM old o JOIN features f USING(ts_code,signal_date)
            WHERE o.signal_date>=DATE '2011-01-01'
            AND o.signal_date<DATE '{bound}' AND o.exit_date<DATE '{bound}'""")
        size, unique = con.execute("SELECT count(*),count(DISTINCT (ts_code,signal_date)) FROM pool").fetchone()
        assert size == unique == expected[year]
        counts = con.execute("SELECT signal_date,count(*) pool_count FROM pool GROUP BY signal_date ORDER BY signal_date").df()
        con.execute(f"""CREATE OR REPLACE TEMP TABLE sampled AS SELECT * FROM pool
            ORDER BY hash(ts_code,signal_date,{SEED}),ts_code,signal_date LIMIT 500000""")
        columns = ",".join(f'f."{n}"' for n in names)
        sample = con.execute(f"SELECT s.*,{columns} FROM sampled s JOIN features f USING(ts_code,signal_date) ORDER BY s.signal_date,s.ts_code").df()
        assert len(sample) == 500000 and not sample.duplicated(KEY).any()
        assert sample.exit_date.max() < pd.Timestamp(bound)
        keys = sample[KEY].copy()
        keys.to_parquet(OUT / f"{year}_sample_keys.parquet", index=False)
        x = transform_features(sample, control["preprocessing"]).astype(float)
        y = sample.y.to_numpy(float)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        counts["sample_count"] = counts.signal_date.map(keys.signal_date.value_counts()).fillna(0).astype(int)
        assert counts.sample_count.sum() == len(sample)
        counts.to_parquet(OUT / f"{year}_daily_counts.parquet", index=False)
        models, weights = {}, {}
        for group, count_column in zip(GROUPS, ["sample_count", "pool_count"]):
            reference = keys.signal_date.map(counts.set_index("signal_date")[count_column]).to_numpy(float)
            assert np.isfinite(reference).all() and (reference > 0).all()
            w = reference ** (-cfg["gamma"])
            w /= w.sum()
            weights[group] = w
            mx, my = w @ x, w @ y
            centered = x - mx
            gram = centered.T @ (centered * w[:, None])
            rhs = centered.T @ (w * (y - my))
            ratio, fraction = cfg["l1_ratio"], cfg["alpha_ratio"]
            alpha = fraction * max(abs(rhs)) / ratio
            fit = fit_elastic_net_from_gram(dict(gram=gram,rhs=rhs,mean_x=mx,mean_y=my), alpha, ratio, np.zeros(19), 30000, 1e-10)
            assert fit["converged"]
            beta = fit["coefficient"]
            gradient = gram @ beta - rhs + alpha * (1-ratio) * beta
            error = float(np.max(np.where(beta != 0, abs(gradient + alpha*ratio*np.sign(beta)), np.maximum(abs(gradient)-alpha*ratio, 0))))
            assert error < 1e-8
            models[group] = dict(coefficient=beta.tolist(), intercept=fit["intercept"], alpha=alpha,
                l1_ratio=ratio, alpha_fraction=fraction, gamma=cfg["gamma"], max_kkt_error=error,
                effective_sample_size=float(1/(w@w)), weight_sum=float(w.sum()))
        diagnostics.append(dict(year=year, pool_rows=size, sample_rows=len(sample),
            weight_correlation=float(np.corrcoef(weights[GROUPS[0]], weights[GROUPS[1]])[0,1]),
            max_relative_weight_difference=float(np.max(abs(weights[GROUPS[1]]/weights[GROUPS[0]]-1)))))
        save(OUT / f"{year}_models.json", dict(models=models, features=names,
            preprocessing=control["preprocessing"], preprocessing_source_sha=sha(CONTROL/f"{year}_model.json"),
            sample_keys_sha=sha(OUT/f"{year}_sample_keys.parquet"), seed=SEED, pool_rows=size))
        chunks = []
        for path in sorted((FEATURES / f"feature_year={year}").glob("*.parquet")):
            frame = pd.read_parquet(path, columns=KEY+names)
            matrix = transform_features(frame, control["preprocessing"]).astype(float)
            block = frame[KEY].copy()
            for group, model in models.items():
                block[group] = matrix @ model["coefficient"] + model["intercept"]
            chunks.append(block)
        labels = pd.read_parquet(ELASTIC / f"validation_{year}/predictions.parquet", columns=KEY+["y","volatility_20d"])
        frame = pd.concat(chunks).merge(labels, on=KEY, validate="one_to_one")
        assert len(frame) == len(labels) and np.isfinite(frame[GROUPS]).all().all()
        scored = frame.loc[np.isfinite(frame.y)]
        ranks = scored.groupby("signal_date")[GROUPS+["y"]].rank()
        for day, values in ranks.groupby(scored.signal_date):
            if len(values) >= 20:
                daily.append(dict(year=year,date=day,**values[GROUPS].corrwith(values.y).to_dict()))
        parts.append(frame.drop(columns="y"))
        parts[-1].to_parquet(OUT / f"{year}_predictions.parquet", index=False)
        print("fold complete", diagnostics[-1], flush=True)
    con.close()
    ic = pd.DataFrame(daily)
    ic.to_parquet(OUT / "daily_ic.parquet", index=False)
    print("mean IC", ic[GROUPS].mean().to_dict(), flush=True)
    frame = pd.concat(parts, ignore_index=True)
    del parts
    template, _, panel, missing, evidence = prepare(frame[KEY+["volatility_20d",GROUPS[0]]].rename(columns={GROUPS[0]:"prediction"}))
    del evidence
    template = template.drop(columns="prediction")
    template["entry_tradable"] = template.entry_tradable.fillna(False) & template.strict_buy
    results = {}
    for group in GROUPS:
        orders = template.merge(frame[KEY+[group]], on=KEY, validate="one_to_one").rename(columns={group:"prediction"})
        assert len(orders) == len(template)
        selected, decision = buffer_panel(orders, panel, 100)
        results[group] = {}
        for impact in [.5, 1.]:
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
    save(OUT/"summary.json", dict(results=results, ic=ic[GROUPS].mean().to_dict(), diagnostics=diagnostics,
        inputs=inputs, script_sha256=sha(Path(__file__)), seed=SEED))
    lines = ["# Elastic共同训练池日期权重对照", "",
        "2017–2023七折，每折从共同训练池按固定hash顺序抽取50万行。两组使用完全相同的样本、19因子、V2训练折预处理和V1正则化规则（l1_ratio=1e-7，alpha_fraction=0.2）。日期权重为每日股票数的−0.75次方，唯一改变是计数来自抽样还是共同完整训练池；每组按其加权统计重新计算alpha。预处理复用既有训练折参数，未重新拟合。", "",
        "回测：1亿元、20bp、前50买入/前100保留、严格涨跌停与整手约束、连续跨年现金账本。下表为七年累计净收益。", "",
        "| 计数来源 | 日均Rank IC | 原冲击净收益 | 双倍冲击净收益 | 双倍冲击最大回撤 |", "|---|---:|---:|---:|---:|"]
    for group in GROUPS:
        r = results[group]
        lines.append(f"| {group} | {ic[group].mean():.6f} | {r['0.5']['net_return']:.2%} | {r['1.0']['net_return']:.2%} | {r['1.0']['max_drawdown']:.2%} |")
    lines += ["", "| 验证年 | 共同池行数 | 抽样行数 | 抽样计数IC | 完整池计数IC | 权重相关系数 |", "|---|---:|---:|---:|---:|---:|"]
    for row in diagnostics:
        annual = ic.loc[ic.year == row['year'], GROUPS].mean()
        lines.append(f"| {row['year']} | {row['pool_rows']} | {row['sample_rows']} | {annual.iloc[0]:.6f} | {annual.iloc[1]:.6f} | {row['weight_correlation']:.6f} |")
    lines += ["", "验证：共同池数量匹配前一步审计、样本主键唯一、目标退出日期早于验证年、两组输入相同、所有拟合收敛且KKT误差小于1e-8、成交价格上下限齐全、现金与持仓账本及累计收益复核通过。",
        "", "边界：共同池沿用V1的target_valid门控，包含未来成交状态筛选，因此本实验仅作历史诊断，不可直接当作生产样本规则。2017–2023为已反复研究的开发期，不是未见测试。训练源只读取2011–2022分区，V1与永久测试保持只读。既有存活偏差、冲击成本近似等限制仍在。",
        "", "本轮两组的差异可以归因于计数参考及其引发的alpha变化；与上一轮广义V2池结果比较还混合了样本成员变化，不能归因于计数方式。"]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
