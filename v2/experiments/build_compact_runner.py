"""Derive separate frozen combination experiment from audited ablation workflow."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

from pathlib import Path
def main():
    root=Path(__file__).resolve().parent;s=(_source_path('v2/run_robust_ablation.py')).read_text(encoding='utf-8')
    s=s.replace("OUT=DATA/'v2/robust_ablation_200k'","OUT=DATA/'v2/compact_cm_200k'")
    s=s.replace("GROUPS=['C','M','F']","GROUPS=['CM']")
    s=s.replace('if family[n]!=group','if family[n] not in set(group)')
    s=s.replace('OLS-informed historical development follow-up; no automatic selection','joint C and M removal; development candidate; no automatic selection')
    s=s.replace('C=条件筛选收益，M=动量反转，F=资金流。所有42个模型收敛','CM=同时移除条件筛选和动量反转。所有14个模型收敛')
    s=s.replace('models=42','models=14')
    s=s.replace('本轮尚未测试同时删除多组，也未评价扣费交易表现。单组增量不能相加；要精简需另行训练组合候选并统一回测。','本轮已实际联合删除CM并重新训练。连续交易结果另见compact_continuous目录。')
    (_source_path('v2/run_compact_candidate.py')).write_text(s,encoding='utf-8')
if __name__=='__main__':main()
