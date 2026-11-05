"""Compare all precommitted scenarios across the two training sample sizes."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'quant_data/processed/v2'
OLD=BASE/'compact_continuous';NEW=BASE/'expanded_continuous_500k'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def key(r):return tuple(r[n] for n in ['kind','loss','policy','scenario'])
def main():
    assert read(NEW/'artifact_verification.json')['cases']==36
    old=read(OLD/'summary.json');new=read(NEW/'summary.json')
    assert old['status']==new['status']=='completed'
    a={key(r):r for r in old['results']};b={key(r):r for r in new['results']}
    assert len(a)==len(b)==36 and a.keys()==b.keys()
    for folder in [OLD,NEW]:
        manifest=read(folder/'manifest.json')
        for path,digest in manifest['sources'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
        for item in manifest['inputs']:
            p=Path(item['path']);assert p.stat().st_size==item['bytes'] and p.stat().st_mtime_ns==item['mtime_ns']
    rows=[]
    for k in a:
        stem='_'.join(k);on=pd.read_parquet(OLD/f'{stem}_nav.parquet');nn=pd.read_parquet(NEW/f'{stem}_nav.parquet')
        pd.testing.assert_series_equal(on.iloc[:,0],nn.iloc[:,0])
        drawdown=float(np.min(nn.nav/np.maximum.accumulate(np.r_[1.,nn.nav])[1:]-1))
        np.testing.assert_allclose(drawdown,b[k]['max_drawdown'],atol=1e-12)
        np.testing.assert_allclose(nn.nav.iloc[-1]**(252/len(nn))-1,b[k]['annualized_return'],atol=1e-12)
        rows.append(dict(zip(['kind','loss','policy','scenario'],k),old_return=a[k]['net_return'],new_return=b[k]['net_return'],delta=b[k]['net_return']-a[k]['net_return'],old_drawdown=a[k]['max_drawdown'],new_drawdown=b[k]['max_drawdown']))
    lines=['# 20万与50万训练样本的连续回测对照','','2017–2023全期累计收益；并非年化收益。四候选、三退出规则、三成本情景全部报告。成本20/50bp为往返基础成本，另加既有冲击代理。','','| 模型 | 损失 | 退出规则 | 成本 | 20万累计收益 | 50万累计收益 | 差值(百分点) | 50万最大回撤 |','|---|---|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["kind"]} | {r["loss"]} | {r["policy"]} | {r["scenario"]} | {r["old_return"]:.2%} | {r["new_return"]:.2%} | {100*r["delta"]:+.2f} | {r["new_drawdown"]:.2%} |')
    improved=sum(r['delta']>0 for r in rows)
    compact_wins=sum(b['compact',l,p,s]['net_return']>b['full',l,p,s]['net_return'] for l in ['huber','mae'] for p in ['fixed5','keep50','keep100'] for s in ['gross','cost20','cost50'])
    lines+=['',f'扩样本后收益提高的匹配场景：{improved}/36；50万样本下精简胜过完整：{compact_wins}/18。','', '仍为历史开发模拟，未据此自动选最终模型。Huber目标尺度也随扩样本重估。沿用当前上市样本带来的幸存者偏差、日线撮合及缺失涨跌停处理、缺乏历史ST和整手约束；没有使用2024年及之后价格，末期未退出持仓按末日估值。下一阶段应检查年度收益、换手及成本来源，再决定是否开展更大样本或冻结候选。']
    (NEW/'SAMPLE_SIZE_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (NEW/'sample_size_comparison.json').write_text(json.dumps(dict(rows=rows,improved_cases=improved,compact_wins=compact_wins,matched_nav_dates=True,source_and_input_checks=True),indent=2),encoding='utf-8')
    print(f'Expanded sample wins {improved}/36; compact wins {compact_wins}/18')
    for r in rows:
        if r['scenario']=='cost20':print(r)
if __name__=='__main__':main()
