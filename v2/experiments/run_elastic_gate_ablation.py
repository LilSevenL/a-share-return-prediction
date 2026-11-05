"""Separate feature availability and future tradability gates in Elastic training."""

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

OUT = CONTROL.parent / "elastic_gate_ablation"
SEED = 20261025
GATES = {"baseline": "TRUE", "features_only": "feature_ok",
         "target_only": "target_ok", "both": "feature_ok AND target_ok"}
GROUPS = list(GATES)


def main():
    OUT.mkdir(exist_ok=True)
    data_root = ROOT / "quant_data/processed"
    cfg = read(data_root / "final_models_2011_2024/final_training_manifest.json")["configs"]["linear_baseline"]
    audit_rows = {r["year"]: r for r in read(CONTROL.parent / "elastic_sample_audit/audit.json")["rows"]}
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='3GB'")
    for name, root, partition in [("features", FEATURES, "feature_year"),
            ("targets", data_root / "targets_5d", "signal_year"),
            ("legacy", data_root / "features_50", "feature_year")]:
        files = [str(p) for year in range(2011, 2023)
                 for p in sorted((root / f"{partition}={year}").glob("*.parquet"))]
        con.read_parquet(files, hive_partitioning=False).create_view(name)
    con.execute(f"""CREATE TABLE universe AS SELECT f.ts_code, f.signal_date, t.exit_date,
        t.target_logret_5d_raw y, coalesce(t.target_valid,false) target_ok,
        coalesce(l.feature_valid_count>=30,false) feature_ok,
        hash(f.ts_code,CAST(f.signal_date AS DATE),{SEED}) priority
        FROM features f JOIN targets t USING(ts_code,signal_date)
        LEFT JOIN legacy l USING(ts_code,signal_date)
        WHERE isfinite(t.target_logret_5d_raw) AND f.signal_date>=DATE '2011-01-01'""")
    parts, daily, diagnostics, residuals = [], [], [], []
    for year in range(2017, 2024):
        control = read(CONTROL / f"{year}_model.json")
        names = control["features"]
        assert names == cfg["factors"] and len(names) == 19
        bound = f"{year}-01-01"
        con.execute(f"""CREATE OR REPLACE TEMP TABLE pool AS SELECT *,
            count(*) OVER(PARTITION BY signal_date) reference_count FROM universe
            WHERE signal_date<DATE '{bound}' AND exit_date<DATE '{bound}'""")
        size, unique_keys, unique_ids = con.execute("SELECT count(*),count(DISTINCT (ts_code,signal_date)),count(DISTINCT priority) FROM pool").fetchone()
        assert size == unique_keys == unique_ids == audit_rows[year]["v2_pool"]
        models, ids, pool_sizes = {}, {}, {}
        for group, condition in GATES.items():
            pool_sizes[group] = con.execute(f"SELECT count(*) FROM pool WHERE {condition}").fetchone()[0]
            columns = ",".join(f'f."{name}"' for name in names)
            sample = con.execute(f"""WITH sampled AS (
                SELECT * FROM pool WHERE {condition}
                ORDER BY priority,ts_code,signal_date LIMIT 500000)
                SELECT s.*, {columns} FROM sampled s JOIN features f USING(ts_code,signal_date)
                ORDER BY s.signal_date,s.ts_code""").df()
            assert len(sample) == 500000 and not sample.duplicated(KEY).any()
            assert sample.exit_date.max() < pd.Timestamp(bound)
            ids[group] = sample.priority.to_numpy().copy()
            if group == "baseline":
                earlier = pd.read_parquet(CONTROL.parent / "expanded_losses_500k" / f"validation_{year}/sample_keys.parquet")
                earlier_overlap = len(sample[KEY].merge(earlier, on=KEY, validate="one_to_one"))/len(sample)
            if group == "both":
                assert pool_sizes[group] == audit_rows[year]["common"]
                prior = pd.read_parquet(CONTROL.parent / "elastic_common_weights" / f"{year}_sample_keys.parquet")
                pd.testing.assert_frame_equal(sample[KEY], prior, check_dtype=False)
            sample[KEY+["reference_count"]].to_parquet(OUT/f"{year}_{group}_keys.parquet",index=False)
            x = transform_features(sample, control["preprocessing"]).astype(float)
            y = sample.y.to_numpy(float)
            assert np.isfinite(x).all() and np.isfinite(y).all()
            reference = sample.reference_count.to_numpy(float)
            w = reference ** (-cfg["gamma"])
            w /= w.sum()
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
        overlap = {f"{a}/{b}": len(np.intersect1d(ids[a], ids[b], assume_unique=True))/500000
                   for i,a in enumerate(GROUPS) for b in GROUPS[i+1:]}
        diagnostics.append(dict(year=year, pool_sizes=pool_sizes, sample_overlap=overlap,
                                prior_broad_overlap=earlier_overlap))
        save(OUT/f"{year}_models.json", dict(models=models, preprocessing=control["preprocessing"],
            features=names, seed=SEED, date_reference="ungated full fold"))
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
        for group in GROUPS:
            errors = scored[group] - scored.y
            residuals.append(dict(year=year, group=group, rows=len(errors), mae=float(errors.abs().mean()),
                rmse=float(np.sqrt((errors**2).mean())), bias=float(errors.mean())))
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
        residuals=residuals, seed=SEED, script_sha256=sha(Path(__file__))))
    report = ["# Elastic 训练样本筛选对照", "",
        "2017–2023 年滚动验证。四组每折各选 50 万行，使用相同 hash 优先序，日期权重均来自当折未经两项筛选的完整训练池。固定 19 因子、既有预处理、V1 正则化规则，每组按训练统计重算 alpha。所有组使用相同验证行和交易机会。features_only 要求旧 50 因子至少 30 项有效，target_only 要求历史目标满足未来可交易条件，both 同时满足两项。", "",
        "模拟资金 1 亿元，20bp 成本，前 50 买入、前 100 保留。收益为连续七年累计。MAE 按验证行加权。", "",
        "| 组别 | 日均 Rank IC | MAE | 原冲击净收益 | 双倍冲击净收益 | 双倍冲击最大回撤 |",
        "|---|---:|---:|---:|---:|---:|"]
    errors = pd.DataFrame(residuals)
    for group in GROUPS:
        e = errors.loc[errors.group == group]
        r = results[group]
        report.append(f"| {group} | {ic[group].mean():.6f} | {np.average(e.mae,weights=e.rows):.7f} | {r['0.5']['net_return']:.2%} | {r['1.0']['net_return']:.2%} | {r['1.0']['max_drawdown']:.2%} |")
    report.extend(["", "各折的 Rank IC、MAE、RMSE 和平均预测偏差如下。", "",
                   "| 年 | 组别 | Rank IC | MAE | RMSE | 平均偏差 |", "|---|---|---:|---:|---:|---:|"])
    for e in residuals:
        annual_ic = ic.loc[ic.year == e["year"], e["group"]].mean()
        report.append(f"| {e['year']} | {e['group']} | {annual_ic:.6f} | {e['mae']:.7f} | {e['rmse']:.7f} | {e['bias']:.7f} |")
    report.extend(["", "样本重合率按交集行数除以 50 万计算。下表为七折范围。", "",
                   "| 配对 | 最低 | 最高 |", "|---|---:|---:|"])
    previous_rates = [row["prior_broad_overlap"] for row in diagnostics]
    for pair in diagnostics[0]["sample_overlap"]:
        rates = [d["sample_overlap"][pair] for d in diagnostics]
        report.append(f"| {pair} | {min(rates):.2%} | {max(rates):.2%} |")
    report.extend(["", "target_valid 依赖信号日之后的成交信息，含此条件的分支仅供历史归因。筛选后按相同顺序补足 50 万行，差异包含替换样本及重新拟合的影响。全部结果来自已经查看的开发期。"])
    report.extend(["", f"日期统一转成 DATE 再计算 hash，双筛选组逐折复现了上轮共同池的全部样本键。早期广义池直接对时间戳计算 hash，同一广义池重新抽样后与早期样本仅重合 {min(previous_rates):.2%} 至 {max(previous_rates):.2%}。早期跨实验收益差还包含抽样顺序变化。"])
    (OUT/"REPORT.md").write_text("\n".join(report)+"\n", encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
