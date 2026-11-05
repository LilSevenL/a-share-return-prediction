"""Build the Elastic status tables from saved experiment outputs."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"quant_data/processed/v2"


def build_section():
    paths=[BASE/"elastic_versions_500k/summary.json",
           BASE/"elastic_compact_gate_seeds/summary.json",
           BASE/"elastic_board_accounts/summary.json"]
    versions,compact,boards=[json.loads(p.read_text(encoding="utf-8")) for p in paths]
    lines=["更新日期：2026-11-04。研究目录已到阶段 38。","",
        "### Elastic Net 两版统一评估","",
        "阶段 27 使用 V1 滚动验证预测和 V2 每折 50 万样本预测，在 2017–2023 年共同的 6,176,670 行标签上评分。交易统一采用 1 亿元初始资金、20bp、严格成交、前 50 买入与前 100 保留。收益为七年累计净收益。","",
        "|模型|日均 Rank IC|原冲击收益|双倍冲击收益|","|---|---:|---:|---:|"]
    for model,label in [("linear_prediction","V1 滚动 Elastic"),("full_elastic","V2 完整 448 因子 Elastic"),("compact_elastic","V2 精简 333 因子 Elastic，首轮")]:
        scenarios={r["impact"]:r for r in versions["results"] if r["model"]==model}
        lines.append(f"|{label}|{versions['rank_ic'][model]:.5f}|{scenarios[.5]['net_return']:.2%}|{scenarios[1.]['net_return']:.2%}|")
    lines+=["","训练样本、因子、预处理、权重和正则化规则随版本变化，上表比较的是已保存配置的整体表现。[两版逐年对照](../quant_data/processed/v2/elastic_versions_500k/COMPARISON.md)","",
        "### 训练样本筛选与三种子复验","",
        "阶段 28–33 先用继承的 19 个因子名称，拆分正则化、日期权重、旧因子有效数及未来成交门控。随后阶段 34 在 333 因子候选上逐行复用三个种子的训练键，固定等权训练、L1 比例 0.5 和 alpha 临界比例 0.1，每折 50 万行。筛选规则为旧 50 项因子至少 30 项有效，仅影响训练样本。","",
        "|种子|训练组|日均 Rank IC|原冲击收益|双倍冲击收益|双倍冲击最大回撤|","|---|---|---:|---:|---:|---:|"]
    for r in compact["rows"]:
        label="无完整度筛选" if r["branch"]=="baseline" else "有完整度筛选"
        lines.append(f"|{r['seed']}|{label}|{r['ic']:.5f}|{r['normal']:.2%}|{r['stress']:.2%}|{r['drawdown']:.2%}|")
    lines+=["","阶段 34 完成 42 次拟合和 12 组回测。三个种子均提高排序分数及两档成本下的收益。Huber/MAE 尚未做这套匹配筛选复验，因此当前数值差同时包含样本选择与训练方法差异。[333 因子复验](../quant_data/processed/v2/elastic_compact_gate_seeds/REPORT.md)","",
        "早期广义池使用时间戳计算抽样 hash，共同池使用日期，固定种子也会抽到不同的行。已确认两种抽法每折仅重合 6.23%–17.13%。后续配对实验统一为日期，跨早期实验的收益差需要同时考虑抽样顺序变化。target_valid 依赖未来成交状态，含该门控的分支仅供历史诊断。[样本筛选审计与对照](../quant_data/processed/v2/elastic_gate_ablation/REPORT.md)","",
        "### 年度、个股、板块与风格归因","",
        "- 阶段 35 核对 12 组策略的年度账本。双倍冲击下，筛选组在七个年度、三个种子的年度净收益率均高于对应基准。2022 年贡献七年人民币净收益差的 37%–46%，前五只股票占净差约 15%–16%。[收益与成本归因](../quant_data/processed/v2/elastic_compact_gate_attribution/REPORT.md)",
        "- 阶段 36–37 完成七年逐日板块和风格拆解，84 本年度账通过核验。降低科创板配置在 2020、2021 年带来负贡献，在 2022 年带来正贡献。主板内部贡献在 2018–2023 年、三个种子下均为正。内部项包含选股、仓位与交易时点差异，各风格维度不能相加。[七年归因](../quant_data/processed/v2/elastic_board_style_multiyear/REPORT.md)","",
        "### 固定初始板块预算对照","",
        "阶段 38 预先指定主板 60%、创业板 20%、科创板 18%、其余 2% 的初始资金比例，以及 30、10、9、1 个买入名额。两组使用同一套约束、独立账户与双倍冲击。","",
        "|种子|无筛选累计收益|筛选累计收益|筛选最大回撤|","|---|---:|---:|---:|"]
    by={(r["group"],r["impact"]):r for r in boards["combined"]}
    for seed in (20261025,20261026,20261027):
        b=by[(f"s{seed}_baseline",1.)]
        f=by[(f"s{seed}_features_only",1.)]
        lines.append(f"|{seed}|{b['net_return']:.2%}|{f['net_return']:.2%}|{f['max_drawdown']:.2%}|")
    lines+=["","48 个板块账户及 12 条合并净值已核验。主板、创业板和科创板三个种子的筛选组收益均更高，其余股票结果方向不一致。账户之间不调拨资金，市值权重随盈亏漂移，板块内名额也与原始自由选股策略不同。该配置用于诊断，尚未做配置优化。[板块账户报告](../quant_data/processed/v2/elastic_board_accounts/REPORT.md)","",
        "### 当前候选与尚待完成事项","",
        "当前研究候选包含精简 Huber、MAE，以及加入训练样本完整度筛选的 333 因子 Elastic Net。已完成的 Elastic 筛选效果在三个抽样种子和共同板块预算下重复出现。候选最终选择仍需要统一训练样本比较各损失、预先记录选择标准、冻结配置，再用之后取得的新数据验证。2017–2023 年已用于反复开发，历史退市覆盖、实际公司行动处理及成交成本校准仍是限制。LightGBM 与 MLP 的 V2 同模型配对尚待实施。"]
    return lines,paths
