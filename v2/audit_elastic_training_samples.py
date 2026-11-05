"""Audit membership and target gates for the seven Elastic development folds."""
from pathlib import Path
import duckdb
import pandas as pd
from run_expanded_losses import ROOT, DATA, OUT as SAMPLES, read, save, sha

OUT = DATA / "v2/elastic_sample_audit"


def main():
    OUT.mkdir(exist_ok=True)
    con=duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='3GB'")
    inputs=[]
    sources=[
        ("old", DATA/"model_dataset/final_split=training","signal_year"),
        ("features", DATA/"v2/local_features_2011_2023_stable","feature_year"),
        ("targets", DATA/"targets_5d","signal_year"),
        ("legacy", DATA/"features_50","feature_year")]
    for name,folder,partition in sources:
        files=[p for year in range(2011,2023) for p in sorted((folder/f"{partition}={year}").glob("*.parquet"))]
        assert files
        inputs.extend(dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in files)
        con.read_parquet([str(p) for p in files],hive_partitioning=False).create_view(name)
    con.execute("""CREATE TABLE v2 AS SELECT f.ts_code,f.signal_date,t.exit_date,
        t.target_logret_5d_raw y,t.target_valid,l.feature_valid_count
        FROM features f JOIN targets t USING(ts_code,signal_date)
        LEFT JOIN legacy l USING(ts_code,signal_date)
        WHERE isfinite(t.target_logret_5d_raw) AND f.signal_date>=DATE '2011-01-01'""")
    assert con.execute("SELECT count(*)-count(DISTINCT (ts_code,signal_date)) FROM v2").fetchone()[0]==0
    folds=read(DATA/"model_dataset/rolling_folds.json")["folds"]
    rows=[]
    for year in range(2017,2024):
        bound=f"{year}-01-01"
        con.execute(f"CREATE OR REPLACE TEMP TABLE a AS SELECT ts_code,signal_date,target_logret_5d y FROM old WHERE signal_date<DATE '{bound}' AND exit_date<DATE '{bound}'")
        con.execute(f"CREATE OR REPLACE TEMP TABLE b AS SELECT * FROM v2 WHERE signal_date<DATE '{bound}' AND exit_date<DATE '{bound}'")
        a=con.execute("SELECT count(*) FROM a").fetchone()[0]
        b=con.execute("SELECT count(*) FROM b").fetchone()[0]
        expected=next(f["training_rows"] for f in folds if f["name"]==f"validation_{year}")
        assert a==expected
        matched=con.execute("""SELECT count(*), max(abs(a.y-b.y)) FROM a JOIN b USING(ts_code,signal_date)""").fetchone()
        reasons=con.execute("""SELECT count(*) FILTER(WHERE NOT coalesce(target_valid,false)),
            count(*) FILTER(WHERE coalesce(target_valid,false) AND feature_valid_count<30),
            count(*) FILTER(WHERE coalesce(target_valid,false) AND feature_valid_count IS NULL),
            count(*) FILTER(WHERE coalesce(target_valid,false) AND feature_valid_count>=30)
            FROM b ANTI JOIN a USING(ts_code,signal_date)""").fetchone()
        assert sum(reasons)==b-matched[0]
        keys=pd.read_parquet(SAMPLES/f"validation_{year}/sample_keys.parquet")
        con.register("sample_keys",keys)
        assert len(keys)==500000 and con.execute("SELECT count(*) FROM sample_keys JOIN b USING(ts_code,signal_date)").fetchone()[0]==len(keys)
        sample_common=con.execute("SELECT count(*) FROM sample_keys JOIN a USING(ts_code,signal_date)").fetchone()[0]
        boundary=con.execute(f"SELECT count(*) FROM v2 WHERE signal_date<DATE '{bound}' AND exit_date>=DATE '{bound}'").fetchone()[0]
        row=dict(year=year,v1_pool=a,v2_pool=b,common=matched[0],v1_only=a-matched[0],v2_only=b-matched[0],
                 target_gate=reasons[0],feature_gate=reasons[1],missing_legacy=reasons[2],other=reasons[3],
                 max_common_label_difference=matched[1],sample_rows=len(keys),sample_common_v1=sample_common,
                 sample_v2_only=len(keys)-sample_common,boundary_purged=boundary)
        rows.append(row)
        print(row,flush=True)
    con.close()
    lines=["# Elastic训练样本与标签门控审计","","比较V1保存的数据集与V2稳定特征连接原始目标得到的训练池。每折均要求信号日期和目标退出日期早于验证年起点。只读取2011–2022分区，覆盖2017–2023训练折。","",
           "| 验证年 | V1训练池 | V2训练池 | 共同样本 | V1独有 | V2独有 | 50万样本中V1也纳入的行 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| "+" | ".join(str(r[k]) for k in ["year","v1_pool","v2_pool","common","v1_only","v2_only","sample_common_v1"])+" |")
    lines+=["","| 验证年 | V2独有且target_valid不成立 | 其余中旧因子有效数不足30 | 缺少旧因子行 | 其他差异 | 时间边界剔除 |","|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| "+" | ".join(str(r[k]) for k in ["year","target_gate","feature_gate","missing_legacy","other","boundary_purged"])+" |")
    maxdiff=max(r["max_common_label_difference"] or 0 for r in rows)
    lines+=["",f"共同样本目标最大绝对差为 {maxdiff:.12g}。V1数据集要求target_valid和旧50项有效因子数至少30。target_valid依赖未来入场及退出的可交易状态。V2使用有限原始目标，并在交易模拟中执行可交易约束。",
            "", "此前四组实验的日期股票数由每折50万抽样重新计数。V1原训练池有数百万行，因此同一日期权重公式的实际样本权重仍会变化。",
            "", "下一轮可先在共同训练池构造固定50万样本，对比按全训练池计数和按抽样计数的日期权重。旧target_valid规则可作为历史诊断分支，生产候选应明确记录未来成交门控的样本选择影响。"]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    save(OUT/"audit.json",dict(rows=rows,inputs=inputs,script_sha256=sha(Path(__file__)),v1_counts_match_saved_folds=True,sample_membership_verified=True))

if __name__=="__main__":
    main()
