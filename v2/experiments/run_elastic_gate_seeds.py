"""Repeat the paired training-sample experiment with two prespecified new seeds."""

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

PRIOR = CONTROL.parent / "elastic_gate_ablation"
OUT = CONTROL.parent / "elastic_gate_seeds"
SEEDS = (20261026, 20261027)
BRANCHES = {"baseline": "TRUE", "features_only": "feature_ok"}
GROUPS = [f"s{seed}_{branch}" for seed in SEEDS for branch in BRANCHES]


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
        t.target_logret_5d_raw y,
        coalesce(l.feature_valid_count>=30,false) feature_ok
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
        size, unique_keys = con.execute("SELECT count(*),count(DISTINCT (ts_code,signal_date)) FROM pool").fetchone()
        assert size == unique_keys == audit_rows[year]["v2_pool"]
        models, ids, pool_sizes = {}, {}, {}
        for group in GROUPS:
            seed_id, branch = group.split("_", 1)
            seed = int(seed_id[1:])
            condition = BRANCHES[branch]
            pool_sizes[group] = con.execute(f"SELECT count(*) FROM pool WHERE {condition}").fetchone()[0]
            columns = ",".join(f'f."{name}"' for name in names)
            sample = con.execute(f"""WITH sampled AS (
                SELECT *, hash(ts_code,CAST(signal_date AS DATE),{seed}) priority FROM pool WHERE {condition}
                ORDER BY priority,ts_code,signal_date LIMIT 500000)
                SELECT s.*, {columns} FROM sampled s JOIN features f USING(ts_code,signal_date)
                ORDER BY s.signal_date,s.ts_code""").df()
            assert len(sample) == 500000 and not sample.duplicated(KEY).any()
            assert sample.exit_date.max() < pd.Timestamp(bound)
            ids[group] = pd.util.hash_pandas_object(sample[KEY], index=False).to_numpy()
            assert len(np.unique(ids[group])) == len(sample)
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
        prior_keys = pd.read_parquet(PRIOR/f"{year}_baseline_keys.parquet", columns=KEY)
        prior_keys["signal_date"] = prior_keys.signal_date.astype(sample.signal_date.dtype)
        original = pd.util.hash_pandas_object(prior_keys, index=False).to_numpy()
        prior_overlap = {g: len(np.intersect1d(original, ids[g]))/len(original) for g in GROUPS}
        diagnostics.append(dict(year=year, pool_sizes=pool_sizes, sample_overlap=overlap, prior_overlap=prior_overlap))
        save(OUT/f"{year}_models.json", dict(models=models, preprocessing=control["preprocessing"],
            features=names, seeds=SEEDS, date_reference="ungated full fold"))
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
        residuals=residuals, seeds=SEEDS, script_sha256=sha(Path(__file__))))
    original = read(PRIOR/"summary.json")
    assert original["seed"] == 20261025 and original["script_sha256"] == sha(_source_path('v2/run_elastic_gate_ablation.py'))
    rows = []
    for seed in (original["seed"], *SEEDS):
        for branch in BRANCHES:
            if seed == original["seed"]:
                group, scores, metrics = branch, original["ic"], original["results"]
            else:
                group, scores, metrics = f"s{seed}_{branch}", ic[GROUPS].mean(), results
            by_cost = metrics[group]
            assert all(r["audit"]["status"] == "passed" for r in by_cost.values())
            rows.append(dict(seed=seed, branch=branch, ic=float(scores[group]),
                normal=by_cost["0.5"]["net_return"], stress=by_cost["1.0"]["net_return"],
                stress_drawdown=by_cost["1.0"]["max_drawdown"]))
    frame = pd.DataFrame(rows).set_index(["seed","branch"])
    diffs = frame.xs("features_only",level="branch") - frame.xs("baseline",level="branch")
    save(OUT/"paired_comparison.json",dict(rows=rows, differences=diffs.reset_index().to_dict("records"),
        previous_summary_sha256=sha(PRIOR/"summary.json")))
    lines = ["# Elastic 因子完整度筛选的多种子复验", "",
        "20261025 复用阶段 32 的原始结果。运行前固定两个新增种子 20261026、20261027。每个种子按股票代码与信号日期生成排序，在同一顺序下分别抽取无筛选和旧 50 项至少 30 项有效的训练样本，每折各 50 万行。筛选仅用于训练。", "",
        "固定 19 因子、V1 正则化规则、各折预处理和完整未筛选训练池的日期计数。2017–2023 年滚动验证，连续回测资金 1 亿元、20bp 成本、前 50 买入、前 100 保留，记录原冲击与双倍冲击的七年累计净收益。", "",
        "| 种子 | 训练组 | 日均 Rank IC | 原冲击净收益 | 双倍冲击净收益 | 双倍冲击最大回撤 |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['branch']} | {row['ic']:.6f} | {row['normal']:.2%} | {row['stress']:.2%} | {row['stress_drawdown']:.2%} |")
    lines += ["", "筛选组减无筛选组的配对差异，收益差以百分点计。", "",
              "| 种子 | Rank IC 差 | 原冲击收益差 | 双倍冲击收益差 |", "|---|---:|---:|---:|"]
    for seed, row in diffs.iterrows():
        lines.append(f"| {seed} | {row.ic:+.6f} | {100*row.normal:+.2f} | {100*row.stress:+.2f} |")
    wins = (diffs.loc[list(SEEDS)] > 0).sum()
    lines.extend(["", f"两个新增种子中，筛选提高 Rank IC 的次数为 {wins.ic}/2，原冲击收益为 {wins.normal}/2，双倍冲击收益为 {wins.stress}/2。三个种子共享同一段历史行情，结果衡量训练抽样敏感性。此复验仍局限于 19 因子训练分支。"])
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
