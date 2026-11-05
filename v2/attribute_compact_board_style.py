"""Daily 2022 allocation and within-bucket attribution on the executed lots."""
from pathlib import Path
import numpy as np
import pandas as pd
from attribute_compact_gate import SOURCE, DATA, AUM, prices, read
from run_expanded_losses import FEATURES

OUT = DATA/"v2/elastic_board_style_2022"
STYLES = {"size":"log_circ_mv","volatility":"volatility_20d","momentum":"logret_60d"}
SEEDS = (20261025,20261026,20261027)


def style_labels(first, last):
    frames = []
    for month in pd.period_range(first,last,freq="M"):
        path = FEATURES/f"feature_year={month.year}"/f"month_{month.month:02d}.parquet"
        data = pd.read_parquet(path,columns=["ts_code","signal_date",*STYLES.values()])
        for axis,col in STYLES.items():
            values = data[col].where(np.isfinite(data[col]))
            rank = values.groupby(data.signal_date).rank(pct=True)
            data[axis] = np.select([rank<=1/3,rank<=2/3,rank.notna()],["low","mid","high"],"missing")
        frames.append(data[["ts_code","signal_date",*STYLES]])
    return pd.concat(frames)


def load_books():
    books, navs = {}, {}
    for seed in SEEDS:
        for impact in [.5,1.]:
            for branch in ["baseline","features_only"]:
                stem = SOURCE/f"s{seed}_{branch}_{impact:g}"
                nav = pd.read_parquet(str(stem)+"_nav.parquet").set_index("date")
                nav["prior"] = nav.nav.shift()*AUM
                key = (seed,impact,branch)
                navs[key] = nav.loc["2022"]
                t = pd.read_parquet(str(stem)+"_orders.parquet")
                t = t.loc[t.status=="filled"].copy()
                for col in ["entry_date","actual_exit_date","signal_date"]:
                    t[col] = pd.to_datetime(t[col])
                books[key] = t.loc[(t.entry_date<"2023-01-01") &
                    (t.actual_exit_date.isna() | (t.actual_exit_date>="2022-01-01"))].copy()
    return books,navs


def daily(book,nav,close,info):
    t = book.merge(info,on=["ts_code","signal_date"],how="left",validate="many_to_one")
    assert (t.entry_adj_open>0).all() and t[list(STYLES)].notna().all().all()
    assert (t.signal_date<t.entry_date).all()
    codes = t.ts_code
    t["board"] = np.select([codes.str.match(r"^68[89]"),codes.str.match(r"^30[01]"),
        codes.str.endswith(".BJ"),codes.str.match(r"^(60|00)")],["STAR","ChiNext","Beijing","Main"],"Other")
    days = nav.index
    starts = days.searchsorted(t.entry_date)
    ends = days.searchsorted(t.actual_exit_date.fillna(days[-1]+pd.Timedelta(days=1)),side="right")
    lengths = ends-starts
    lot = np.repeat(np.arange(len(t)),lengths)
    day = np.arange(len(lot)) + np.repeat(starts-(lengths.cumsum()-lengths),lengths)
    calendar = days.to_numpy(dtype="datetime64[D]")[day]
    buy = calendar==t.entry_date.to_numpy(dtype="datetime64[D]")[lot]
    sell = calendar==t.actual_exit_date.to_numpy(dtype="datetime64[D]")[lot]
    shares = (t.buy_notional/t.entry_adj_open).to_numpy()[lot]
    stock = close.columns.get_indexer(t.ts_code)[lot]
    assert (stock>=0).all()
    mark = close.to_numpy()
    exposure = np.where(buy,t.buy_notional.to_numpy()[lot],shares*mark[day,stock])
    end = np.where(sell,0.,shares*mark[day+1,stock])
    pnl = end + np.where(sell,t.sell_notional.to_numpy()[lot],0.) - exposure
    fee = buy*t.buy_cost_amount.to_numpy()[lot]+sell*t.sell_cost_amount.to_numpy()[lot]
    daily = pd.DataFrame(dict(date=days.take(day),exposure=exposure,end=end,pnl=pnl,fee=fee))
    assert np.isfinite(daily[["exposure","end","pnl","fee"]]).all().all()
    sums = daily.groupby("date").sum().reindex(days,fill_value=0)
    np.testing.assert_allclose(sums.end,nav.holdings,rtol=1e-9,atol=.05)
    np.testing.assert_allclose(sums.pnl-sums.fee,nav.daily_return*nav.prior,rtol=0,atol=.05)
    groups=[]
    for axis in ["board",*STYLES]:
        bucket = t[axis].to_numpy()[lot]
        g = daily.groupby(["date",bucket]).sum()
        g.index.names = ["date","bucket"]
        g=g.reset_index()
        prior=g.date.map(nav.prior)
        g["w"]=g.exposure/prior
        g["r"]=g.pnl/g.exposure
        g["cost_rate"]=g.fee/prior
        g["close_w"]=g.end/(g.date.map(nav.nav)*AUM)
        groups.append(g.assign(axis=axis))
    return pd.concat(groups)


def pair(base,filtered,bnav,fnav):
    cols=["axis","date","bucket"]
    p=base.merge(filtered,on=cols,how="outer",suffixes=("_b","_f"),validate="one_to_one")
    # A bucket held by only one side is attributed to allocation.
    rb=p.r_b.fillna(p.r_f)
    rf=p.r_f.fillna(p.r_b)
    p=p.fillna(0)
    wb,wf=p.w_b,p.w_f
    prev_f=fnav.prior/fnav.prior.iloc[0]
    future_b=(1+bnav.daily_return).prod()/(1+bnav.daily_return).cumprod()
    link=p.date.map(prev_f*future_b)
    p["allocation"]=(wf-wb)*(rf+rb)/2*link
    p["within"]=(wf+wb)/2*(rf-rb)*link
    p["cost_effect"]=(p.cost_rate_b-p.cost_rate_f)*link
    p["net"]=p.allocation+p.within+p.cost_effect
    expected=fnav.nav.iloc[-1]*AUM/fnav.prior.iloc[0]-bnav.nav.iloc[-1]*AUM/bnav.prior.iloc[0]
    np.testing.assert_allclose(p.groupby("axis").net.sum(),expected,rtol=0,atol=1e-8)
    return p.groupby(["axis","bucket"]).agg(base_close_weight=("close_w_b","sum"),
        filtered_close_weight=("close_w_f","sum"),allocation=("allocation","sum"),
        within=("within","sum"),cost_effect=("cost_effect","sum"),net=("net","sum")).assign(
        base_close_weight=lambda z:z.base_close_weight/len(bnav),
        filtered_close_weight=lambda z:z.filtered_close_weight/len(bnav)).reset_index()


def main():
    OUT.mkdir(exist_ok=True)
    books,navs=load_books()
    con,_=prices()
    allkeys=pd.concat([t[["ts_code","signal_date"]] for t in books.values()]).drop_duplicates()
    con.register("keys",allkeys)
    info=con.execute("SELECT k.*, t.entry_adj_open FROM keys k LEFT JOIN targets t USING(ts_code,signal_date)").df()
    info=info.merge(style_labels(info.signal_date.min(),info.signal_date.max()),on=["ts_code","signal_date"],validate="one_to_one")
    marks=con.execute("""WITH m AS (SELECT ts_code,trade_date,CAST(close*adj_factor AS FLOAT) price
        FROM daily JOIN adj_factor USING(ts_code,trade_date) WHERE trade_date<DATE '2023-01-01')
        SELECT * FROM m WHERE trade_date>=DATE '2022-01-01' UNION ALL
        SELECT ts_code,DATE '2021-12-31',arg_max(price,trade_date) FROM m WHERE trade_date<DATE '2022-01-01' GROUP BY ts_code""").df()
    con.close()
    close=marks.pivot(index="trade_date",columns="ts_code",values="price").sort_index().ffill().astype(float)
    daily_rows, summaries=[],[]
    for seed in SEEDS:
        for impact in [.5,1.]:
            keys=[(seed,impact,branch) for branch in ["baseline","features_only"]]
            frames=[]
            for k in keys:
                nav=navs[k]
                assert np.array_equal(close.index[1:].values,nav.index.values)
                g=daily(books[k],nav,close,info)
                daily_rows.append(g.assign(seed=seed,impact=impact,branch=k[-1]))
                frames.append(g)
            summaries.append(pair(*frames,navs[keys[0]],navs[keys[1]]).assign(seed=seed,impact=impact))
    pd.concat(daily_rows).to_parquet(OUT/"daily_buckets.parquet",index=False)
    summary=pd.concat(summaries)
    summary.to_parquet(OUT/"attribution.parquet",index=False)
    lines=["# 2022 年板块与风格归因","",
        "基准为同种子无筛选组合。平均持仓占比按每日收盘持仓市值除以组合净值计算。每日毛收益归因按昨收持仓加当日买入金额衡量投入，卖出按当日收盘执行，剩余现金收益为零。每日配置项为投入权重差乘两组桶内毛回报的均值，桶内项为平均权重乘桶内回报差，两者平分交互项。单边持有的桶全部计入配置。成本按成交日单列。按筛选组此前累计净值与基准此后增长倍数链接到全年，逐日核对持仓及净损益，四种分组均复核全年净收益率差。","",
        "风格按各笔买单信号日全截面有效值的三分位分组，持有期间固定。同日缺失值独立成组。规模使用流通市值对数，波动使用过去 20 日波动率，动量使用过去 60 日收益。各维度都是同一收益差的独立拆解，不能跨维度相加。桶内项包含个股选择、交易时点与桶内权重变化。","",
        "|种子|冲击|维度|类别|无筛选收盘占比|筛选收盘占比|配置 pp|桶内 pp|成本影响 pp|净贡献 pp|",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    stress=summary.loc[(summary.impact==1)&(summary["axis"]=="board")]
    overview=["**双倍冲击下的年度结果**","","|种子|科创板日均收盘占比（无筛选 → 筛选）|全板块配置 pp|全板块内部 pp|成本影响 pp|年度净收益差 pp|",
              "|---|---:|---:|---:|---:|---:|"]
    for seed,g in stress.groupby("seed"):
        star=g.loc[g.bucket=="STAR"].iloc[0]
        overview.append(f"|{seed}|{star.base_close_weight:.2%} → {star.filtered_close_weight:.2%}|{100*g.allocation.sum():+.2f}|{100*g.within.sum():+.2f}|{100*g.cost_effect.sum():+.2f}|{100*g.net.sum():+.2f}|")
    overview.extend(["","2022 年筛选组降低了科创板配置，三个种子的科创板配置效应均为正。主板内部收益差也是三个种子的正贡献来源。风格上，三个种子均提高低波动组占比、降低高波动组占比。以下分类结果为收益差的会计拆分，仅覆盖 2022 年。",""])
    lines[2:2]=overview
    for r in summary.itertuples():
        lines.append(f"|{r.seed}|{r.impact}|{r.axis}|{r.bucket}|{r.base_close_weight:.2%}|{r.filtered_close_weight:.2%}|{100*r.allocation:+.2f}|{100*r.within:+.2f}|{100*r.cost_effect:+.2f}|{100*r.net:+.2f}|")
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
