"""Build a sourced comparison from saved V1/V2 artifacts without refitting."""
from pathlib import Path
import json
import hashlib
import numpy as np
from elastic_report_section import build_section

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'quant_data/processed'
OUT = ROOT / 'v2/V1_V2_MODEL_COMPARISON.md'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    sources = []
    pairs = []
    for year in range(2017, 2024):
        old_path = DATA / f'model_training_ols/validation_{year}/ols_metrics.json'
        new_path = DATA / f'v2/ols_baseline_stable/validation_{year}/result.json'
        old = read(old_path)
        new = read(new_path)
        pairs.append((year, old, new['predictive_metrics']))
        sources.extend([old_path, new_path])
    v1_manifest = DATA / 'final_models_2011_2024/final_training_manifest.json'
    final = read(v1_manifest)
    expanded_path = DATA / 'v2/expanded_losses_500k/summary.json'
    expanded = read(expanded_path)['aggregate']
    sources.extend([v1_manifest, expanded_path])
    old_ic = np.mean([a['daily_rank_ic_mean'] for _, a, _ in pairs])
    new_ic = np.mean([b['mean_rank_ic'] for _, _, b in pairs])
    old_days = sum(a['trading_days'] for _, a, _ in pairs)
    new_days = sum(b['days'] for _, _, b in pairs)
    old_daily = sum(a['daily_rank_ic_mean']*a['trading_days'] for _, a, _ in pairs)/old_days
    new_daily = sum(b['mean_rank_ic']*b['days'] for _, _, b in pairs)/new_days
    lines = ['# V1与V2相同模型对比报告', '',
        '本报告比较两版 OLS 与 Elastic Net 的已保存结果，统一取 2017–2023 七个验证年度。第 9 节更新至 2026-11-04，包含 Elastic 样本筛选、三种子复验及板块账户诊断。Huber 与 MAE 作为后续线性候选单列。数值来自实际保存产物。', '',
        '## 1. 对应模型与范围', '',
        '| 模型 | V1状态 | V2状态 | 本次比较方式 |',
        '|---|---|---|---|',
        '| OLS线性基准 | 保存2017–2024逐年模型 | 保存2017–2023逐年模型 | 以共同七年结果为主表 |',
        '| Elastic Net | 滚动线性参照及最终19因子配置 | 已训练448/333因子及19因子诊断分支 | 统一标签和交易规则对照，见第9节 |',
        '| LightGBM | V1已有训练及最终候选 | 当前阶段尚无对应正式比较结果 | 待对应训练后补表 |',
        '| MLP | V1已有训练及最终主模型 | 当前阶段尚无对应正式比较结果 | 待对应训练后补表 |', '',
        f'V1最终模型角色来自冻结清单，主模型为 {final["roles"]["primary"]}，线性参照为 {final["roles"]["baseline"]}，挑战者为 {final["roles"]["strong_challenger"]}。最终模型使用2011–2024训练数据，不能用其训练期内2017–2023预测充当逐年验证。这里采用V1保存的滚动OLS作为同模型参照。', '',
        '## 2. 两版OLS的实现和数据口径', '',
        '| 项目 | V1 OLS | V2 OLS |',
        '|---|---|---|',
        '| 因子 | 原50因子库，按各折预处理决定有效列 | 500候选库，各折448个有效输入 |',
        '| 目标字段 | target_logret_5d | target_logret_5d_raw |',
        '| 样本门控 | 数据集构建要求target_valid | 使用有限原始目标，交易可成交性在模拟中处理 |',
        '| 训练方式 | 等权、MSE、线性回归 | 等权、MSE、线性回归 |',
        '| 截距处理 | 将训练y均值固定为截距，再估计系数 | 截距与系数共同拟合 |',
        '| 求解器 | 累积X转置X，用伪逆求解 | 流式增广QR，再用SVD求解 |',
        '| 时间验证 | 已保存逐年扩展窗口 | 逐年扩展窗口，并要求目标退出日在验证起点之前 |',
        '| 预处理 | 训练期缺失值、截尾及标准化 | 训练期预处理及高相关列筛选 |', '',
        '两版共同覆盖相同年份，但评分样本、标签可用性、因子、预处理和求解器均有变化。下表反映各自原实验的历史结果，分数差不能单独归因于增加因子。', '',
        '## 3. 逐年OLS结果', '',
        '| 年份 | V1 Rank IC | V2 Rank IC | 差值 | V1评分行数 | V2评分行数 | V1 RMSE | V2 RMSE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for year, old, new in pairs:
        lines.append(f'| {year} | {old["daily_rank_ic_mean"]:.5f} | {new["mean_rank_ic"]:.5f} | {new["mean_rank_ic"]-old["daily_rank_ic_mean"]:+.5f} | {old["rows"]:,} | {new["rows"]:,} | {old["rmse"]:.5f} | {new["rmse"]:.5f} |')
    wins = sum(b['mean_rank_ic'] > a['daily_rank_ic_mean'] for _, a, b in pairs)
    lines += ['', '## 4. 共同七年汇总', '',
        '| 汇总口径 | V1 OLS | V2 OLS |', '|---|---:|---:|',
        f'| 年度Rank IC均值，再七年等权 | {old_ic:.5f} | {new_ic:.5f} |',
        f'| 按各自有效交易日数加权的Rank IC | {old_daily:.5f} | {new_daily:.5f} |',
        f'| 评分行数合计 | {sum(a["rows"] for _, a, _ in pairs):,} | {sum(b["rows"] for _, _, b in pairs):,} |',
        f'| 有效交易日合计 | {old_days:,} | {new_days:,} |', '',
        f'V2 OLS年度Rank IC高于V1的年份数为{wins}/7。RMSE受标签分布和评分样本变化影响，应结合上表样本数阅读。V1原汇总含2024年，本报告重新取共同七年汇总。', '',
        '## 5. V2同样本改进结果', '',
        '以下为V2内部实验，验证期2017–2023。完整OLS/MSE的20万样本Rank IC为0.05098。它与下列50万训练样本候选的差异同时包含样本量和训练方法变化。', '',
        '| V2候选 | 每折训练样本 | 实际输入 | 日均Rank IC |', '|---|---:|---:|---:|']
    for name, title, count in [('full_huber','完整Huber',448),('full_mae','完整MAE',448),('compact_huber','精简Huber',333),('compact_mae','精简MAE',333)]:
        lines.append(f'| {title} | 500,000 | {count} | {expanded[name]["mean_rank_ic"]:.5f} |')
    lines += ['', '## 6. 交易结果的可比范围', '',
        'Elastic Net 已完成共同股票日期、标签及交易规则下的两版对照，见第 9 节。OLS 的两版同口径交易增量仍待对应实验。下表保留 V2 稳健损失候选的联合压力结果。', '',
        '| V2前100名保留候选 | 严格成交与双倍冲击累计收益 | 年化收益 | 最大回撤 |',
        '|---|---:|---:|---:|',
        '| 精简Huber | 20.38% | 2.79% | −32.26% |',
        '| 精简MAE | 12.19% | 1.72% | −34.30% |', '',
        '以上采用初始1亿元、20bp往返基础费及双倍冲击，期间2017–2023。股票池历史覆盖和成交假设仍影响结果。', '',
        '## 7. 当前结论与待补比较', '',
        'OLS 与 Elastic Net 均已有两版结果。因子输入、样本门控、预处理、权重和正则化等同时变化，增加因子的独立效果仍需匹配实验。Elastic 完整度筛选已在 333 因子、三个种子和共同初始板块预算下复验。LightGBM、MLP 的 V2 配对比较仍待实施。', '',
        '2017–2023属于已多次查看的历史开发期。V1已查看的永久Test和V2未来独立验证需分别记录。', '',
        '## 8. 结果和代码依据', '',
        '- [V1 OLS训练源码](../train_ols.py)',
        '- [V1模型数据集门控源码](../build_model_dataset.py)',
        '- [V1最终模型清单](../quant_data/processed/final_models_2011_2024/final_training_manifest.json)',
        '- [V2 OLS报告](../quant_data/processed/v2/ols_baseline_stable/OLS_REPORT.md)',
        '- [V2扩样本结果](../quant_data/processed/v2/expanded_losses_500k/EXPANDED_SAMPLE_REPORT.md)',
        '- [V2联合压力结果](../quant_data/processed/v2/joint_execution_stress_500k/JOINT_STRESS_REPORT.md)',
        '- [V2总步骤报告](V2_PROGRESS_SUMMARY.md)', '',
        '逐年指标来源及SHA256保存在 V1_V2_MODEL_COMPARISON_SOURCES.json，可运行 build_v1_v2_comparison.py 重新生成。']
    elastic_lines, elastic_sources = build_section()
    lines += ['', '## 9. Elastic Net 及后续实验更新', ''] + elastic_lines
    sources.extend(elastic_sources)
    sources.extend([Path(__file__), Path(__file__).with_name('elastic_report_section.py')])
    OUT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    provenance = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (OUT.parent/'V1_V2_MODEL_COMPARISON_SOURCES.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding='utf-8')
    print(dict(report=str(OUT),v1_rank_ic=old_daily,v2_rank_ic=new_daily,v2_better_years=wins))


if __name__ == '__main__':
    main()
