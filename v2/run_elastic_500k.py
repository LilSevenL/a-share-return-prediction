"""First matched V2 Elastic Net experiment on the verified 500k samples."""
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_expanded_losses import OUT as SAMPLES, FEATURES, ROOT, read, save, sha
from feature_model import bundle_model, predict_frame
from research_controls import transform_features, evaluate_common, sample_weights, WeightConfig
from tune_elastic_net import fit_elastic_net_from_gram

OUT = SAMPLES.parent / "elastic_500k"
L1_RATIO = 0.5
ALPHA_FRACTION = 0.1


def train(folder, dest):
    sample = read(folder / "sample.json")
    for name in ["x", "y"]:
        assert sha(folder / f"sample_{name}.npy") == sample[f"{name}_sha"]
    x = np.load(folder / "sample_x.npy", mmap_mode="r")
    y = np.load(folder / "sample_y.npy")
    assert x.shape == (500000,448) and len(y) == len(x)
    mean = x.mean(axis=0, dtype=np.float64)
    gram = np.zeros((448,448))
    rhs = np.zeros(448)
    for start in range(0,len(y),20000):
        block = x[start:start+20000].astype(float) - mean
        gram += block.T @ block / len(y)
        rhs += block.T @ (y[start:start+20000]-y.mean()) / len(y)
    full = read(folder / "full_huber/model.json")["active_feature_names"]
    models = {}
    for kind in ["full","compact"]:
        stats = read(folder / f"{kind}_huber/model.json")["preprocessing"]
        cols = [full.index(n) for n in stats["active_names"]]
        g = gram[np.ix_(cols,cols)]
        b = rhs[cols]
        alpha = ALPHA_FRACTION * np.max(abs(b)) / L1_RATIO
        fit = fit_elastic_net_from_gram(dict(gram=g,rhs=b,mean_y=y.mean(),mean_x=mean[cols]),
                                        alpha,L1_RATIO,np.zeros(len(cols)),30000,1e-10)
        assert fit["converged"]
        beta = fit["coefficient"]
        grad = g@beta-b+alpha*(1-L1_RATIO)*beta
        residual = np.where(beta!=0,abs(grad+alpha*L1_RATIO*np.sign(beta)),
                            np.maximum(abs(grad)-alpha*L1_RATIO,0))
        assert residual.max()<1e-8
        ols = np.linalg.lstsq(g,b,rcond=1e-12)[0]
        assert max(abs(g@ols-b))<1e-8
        for method,coef in [("elastic",beta),("ols",ols)]:
            name = f"{kind}_{method}"
            model = dict(coefficient=coef.tolist(),intercept=float(y.mean()-mean[cols]@coef),
                         method=method,alpha=float(alpha) if method=="elastic" else 0.,
                         training_rows=len(y),l1_ratio=L1_RATIO if method=="elastic" else 0.)
            if method=="elastic":
                model.update(iterations=fit["iterations"],max_kkt_error=float(residual.max()))
            models[name] = bundle_model(model,stats)
            save(dest/f"{name}_model.json",models[name])
    return models,full


def main():
    OUT.mkdir(exist_ok=True)
    assert read(SAMPLES/"verification.json")["models"]==28
    parent = read(SAMPLES/"manifest.json")
    for item in parent["inputs"]:
        stat = Path(item["path"]).stat()
        assert (stat.st_size,stat.st_mtime_ns)==(item["bytes"],item["mtime_ns"])
    manifest = dict(l1_ratio=L1_RATIO,alpha_fraction=ALPHA_FRACTION,weight="uniform",
                    years=list(range(2017,2024)),parent_sha=sha(SAMPLES/"manifest.json"),
                    runner_sha=sha(Path(__file__)),solver_sha=sha(ROOT/"tune_elastic_net.py"))
    if (OUT/"manifest.json").exists(): assert read(OUT/"manifest.json")==manifest
    save(OUT/"manifest.json",manifest)
    rows=[]
    for year in manifest["years"]:
        folder=SAMPLES/f"validation_{year}"
        dest=OUT/folder.name
        dest.mkdir(exist_ok=True)
        if (dest/"results.json").exists():
            rows.extend(read(dest/"results.json"))
            continue
        print(f"Training {year}",flush=True)
        models,full=train(folder,dest)
        chunks=[]
        for path in sorted((FEATURES/f"feature_year={year}").glob("*.parquet")):
            frame=pd.read_parquet(path)
            matrix=transform_features(frame,models["full_elastic"]["preprocessing"])
            pred=frame[["ts_code","signal_date","volatility_20d"]].copy()
            for name,model in models.items():
                linear=model["linear_model"]
                cols=[full.index(n) for n in model["active_feature_names"]]
                pred[name]=matrix[:,cols].astype(float)@linear["coefficient"]+linear["intercept"]
                np.testing.assert_allclose(pred[name].iloc[:16],predict_frame(model,frame.iloc[:16]),atol=1e-10)
            chunks.append(pred)
        labels=pd.read_parquet(folder/"full_huber/predictions.parquet",columns=["ts_code","signal_date","y"])
        pred=pd.concat(chunks).merge(labels,on=["ts_code","signal_date"],validate="one_to_one")
        assert len(pred)==len(labels) and np.isfinite(pred[list(models)]).all().all()
        pred.to_parquet(dest/"predictions.parquet",index=False)
        mask=np.isfinite(pred.y)
        _,weights=sample_weights(pred.signal_date.to_numpy(),pred.volatility_20d.to_numpy(),WeightConfig(date_gamma=0,volatility_power=1))
        results=[]
        for name,model in models.items():
            m,_=evaluate_common(pred[name][mask].to_numpy(),pred.y[mask].to_numpy(),pred.signal_date[mask].to_numpy(),weights[mask])
            results.append(dict(year=year,name=name,nonzero=int(np.count_nonzero(model["linear_model"]["coefficient"])),**m))
        save(dest/"results.json",results)
        rows.extend(results)
        print(year,[(r["name"],round(r["mean_rank_ic"],5),r["nonzero"]) for r in results],flush=True)
    table=pd.DataFrame(rows)
    lines=["# V2 Elastic Net 首轮训练","", "2017–2023 历史开发验证，每折50万行，等权。各折按训练数据设置 alpha 为全零解临界值的10%，L1比例固定0.5。OLS使用相同样本、截距、输入列和评分股票。","", "| 模型 | 日均Rank IC | 非零系数范围 |","|---|---:|---:|"]
    for name,g in table.groupby("name"):
        assert len(g)==7
        ic=np.average(g.mean_rank_ic,weights=g.rank_ic_valid_days)
        lines.append(f"| {name} | {ic:.5f} | {g.nonzero.min()}–{g.nonzero.max()} |")
    lines+=["","| 年份 | 完整OLS | 完整Elastic | 精简OLS | 精简Elastic |","|---|---:|---:|---:|---:|"]
    for year,g in table.groupby("year"):
        values=g.set_index("name").mean_rank_ic
        lines.append(f"| {year} | "+" | ".join(f"{values[name]:.5f}" for name in ["full_ols","full_elastic","compact_ols","compact_elastic"])+" |")
    save(OUT/"summary.json",dict(status="completed",results=rows))
    (OUT/"ELASTIC_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines[3:10]),flush=True)

if __name__=="__main__":
    with threadpool_limits(limits=4):
        main()
