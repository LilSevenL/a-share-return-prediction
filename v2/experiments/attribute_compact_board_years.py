"""Apply the reconciled board/style attribution to all seven development years."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import numpy as np
import pandas as pd
from attribute_compact_board_style import (
    SOURCE, DATA, AUM, prices, style_labels, daily, pair, SEEDS)
from run_expanded_losses import save, sha
from pathlib import Path

OUT = DATA/"v2/elastic_board_style_multiyear"
YEARS = range(2017,2024)


def main():
    OUT.mkdir(exist_ok=True)
    books,navs,inputs = {},{},{}
    for seed in SEEDS:
        for impact in [.5,1.]:
            for branch in ["baseline","features_only"]:
                key=(seed,impact,branch)
                stem=SOURCE/f"s{seed}_{branch}_{impact:g}"
                order_path=Path(str(stem)+"_orders.parquet")
                nav_path=Path(str(stem)+"_nav.parquet")
                inputs[str(order_path)]=sha(order_path)
                inputs[str(nav_path)]=sha(nav_path)
                t=pd.read_parquet(order_path)
                t=t.loc[t.status=="filled"].copy()
                for col in ["entry_date","actual_exit_date","signal_date"]:
                    t[col]=pd.to_datetime(t[col])
                books[key]=t
                nav=pd.read_parquet(nav_path).set_index("date")
                nav["prior"]=nav.nav.shift(fill_value=1)*AUM
                navs[key]=nav
    allkeys=pd.concat([t[["ts_code","signal_date"]] for t in books.values()]).drop_duplicates()
    con,_=prices()
    con.register("keys",allkeys)
    info=con.execute("SELECT k.*,t.entry_adj_open FROM keys k LEFT JOIN targets t USING(ts_code,signal_date)").df()
    info=info.merge(style_labels(info.signal_date.min(),info.signal_date.max()),
                    on=["ts_code","signal_date"],validate="one_to_one")
    marks=con.execute("""SELECT ts_code,trade_date,CAST(close*adj_factor AS FLOAT) price
        FROM daily JOIN adj_factor USING(ts_code,trade_date)""").df()
    con.close()
    close=marks.pivot(index="trade_date",columns="ts_code",values="price").sort_index().ffill().astype(float)
    initial=pd.DataFrame(np.nan,index=[pd.Timestamp("2016-12-31")],columns=close.columns)
    close=pd.concat([initial,close])
    summaries=[]
    for year in YEARS:
        daily_rows=[]
        for seed in SEEDS:
            for impact in [.5,1.]:
                keys=[(seed,impact,b) for b in ["baseline","features_only"]]
                frames=[]
                annual_navs=[]
                for key in keys:
                    nav=navs[key].loc[str(year)]
                    dates=nav.index
                    start=close.index.get_loc(dates[0])
                    end=close.index.get_loc(dates[-1])
                    annual_close=close.iloc[start-1:end+1]
                    assert np.array_equal(annual_close.index[1:].values,dates.values)
                    trades=books[key]
                    selected=trades.loc[(trades.entry_date<=dates[-1])&
                        (trades.actual_exit_date.isna()|(trades.actual_exit_date>=dates[0]))]
                    frame=daily(selected,nav,annual_close,info)
                    daily_rows.append(frame.assign(seed=seed,impact=impact,branch=key[-1],year=year))
                    frames.append(frame)
                    annual_navs.append(nav)
                summaries.append(pair(*frames,*annual_navs).assign(seed=seed,impact=impact,year=year))
        pd.concat(daily_rows).to_parquet(OUT/f"{year}_daily.parquet",index=False)
        print("reconciled",year,flush=True)
    summary=pd.concat(summaries,ignore_index=True)
    previous=pd.read_parquet(DATA/"v2/elastic_board_style_2022/attribution.parquet")
    key=["seed","impact","axis","bucket"]
    pd.testing.assert_frame_equal(summary.loc[summary.year==2022].drop(columns="year").sort_values(key).reset_index(drop=True),
                                  previous.sort_values(key).reset_index(drop=True),atol=1e-8,rtol=1e-8)
    summary.to_parquet(OUT/"attribution.parquet",index=False)
    save(OUT/"verification.json",dict(years=list(YEARS),annual_ledgers=84,paired_comparisons=42,
        dimensions=4,previous_2022_matched=True,inputs=inputs,script_sha=sha(Path(__file__))))
    report=["# 2017–2023 年板块与风格归因","","三种抽样种子、两档冲击成本、两组组合，每年逐日重建持仓与净损益。84 本年度账核对通过，四种分组分别重构每个年度的净收益率差。2022 年逐项复现上一阶段结果。","",
        "以下区间为三个种子在双倍冲击下的最小至最大值。科创板仓位差为筛选组减无筛选组的日均收盘仓位。贡献单位为年度百分点，各年贡献不能直接相加。","",
        "|年|科创板仓位差 pp|科创板配置贡献 pp|主板内部贡献 pp|所有板块配置 pp|所有板块内部 pp|净收益率差 pp|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    board=summary.loc[(summary.impact==1)&(summary["axis"]=="board")].copy()
    def interval(values):
        return f"{100*values.min():+.2f} 至 {100*values.max():+.2f}"
    for year,g in board.groupby("year"):
        star=g.loc[g.bucket=="STAR"]
        main=g.loc[g.bucket=="Main"]
        totals=g.groupby("seed")[["allocation","within","net"]].sum()
        weight=star.filtered_close_weight-star.base_close_weight
        star_text=[interval(weight),interval(star.allocation)] if len(star) else ["无持仓","无持仓"]
        report.append(f"|{year}|{star_text[0]}|{star_text[1]}|{interval(main.within)}|{interval(totals.allocation)}|{interval(totals.within)}|{interval(totals.net)}|")
    report.extend(["","降低科创板配置的贡献随年份变化。2020、2021 年三个种子均为负，2022 年均为正，2023 年均为小幅负值。主板内部贡献在 2018–2023 年三个种子均为正，2017 年均为负。风格变化从 2019 年起持续体现为降低高波动组占比、提高低波动组占比。因此 2022 年的配置收益具有时期依赖性，主板内部改善的重复范围更广。","",
        "风格变化仍按信号日三分位分组，每笔持仓的类别在买入前确定。以下为同样三个种子的区间。","",
        "|年|高波动组仓位差 pp|低波动组仓位差 pp|大市值组仓位差 pp|高动量组仓位差 pp|","|---|---:|---:|---:|---:|"])
    for year in YEARS:
        values=[]
        for axis,bucket in [("volatility","high"),("volatility","low"),("size","high"),("momentum","high")]:
            g=summary.loc[(summary.year==year)&(summary.impact==1)&(summary["axis"]==axis)&(summary.bucket==bucket)]
            values.append(interval(g.filtered_close_weight-g.base_close_weight))
        report.append(f"|{year}|"+"|".join(values)+"|")
    report.extend(["","归因沿用上一阶段的对称分配方式，将投入权重与桶内回报的交互项各分一半，再逐日链接到各年累计收益差。单边持有桶的差异计入配置。成本按实际成交日单列。板块内部项包含个股选择、交易时点与内部仓位差。不同风格维度为对同一收益差的独立拆分。","",
        "上述结果使用已经查看的历史开发期，具体分类及链接方式见阶段 36。完整两档成本和全部风格明细保存于 attribution.parquet，各年逐日结果保存于对应 daily.parquet。"])
    (OUT/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
