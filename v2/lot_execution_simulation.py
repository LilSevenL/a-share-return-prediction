"""Separate diagnostic: release sold lots' cash, keep unsold lots marked and locked."""
import numpy as np
import pandas as pd


def simulate_released_cash(orders,panel,top_n=50,aum=1e8,cost_bps=20.,participation_cap=.05,impact_coefficient=.5,round_lots=False):
    dates,close,_,next_sellable,lookup=panel;n=len(dates)
    cash=np.full((5,n),aum/5.);held=np.zeros((5,n));records=[];empty_days=0
    if orders.duplicated(['ts_code','signal_date']).any() or not np.isfinite(orders.prediction).all():raise ValueError('Bad orders')
    for day,rows in orders.groupby('signal_date',sort=True):
        si=lookup.get(np.datetime64(day,'D'));entry=si+1 if si is not None else n
        if entry>=n:continue
        slot=si%5;available=cash[slot,entry-1]
        if available<=1e-8:empty_days+=1;continue
        budget=available/top_n
        for row in rows.sort_values(['prediction','ts_code'],ascending=[False,True]).head(top_n).itertuples():
            rec=dict(signal_date=str(day),ts_code=row.ts_code,budget=float(budget),status='unfilled');records.append(rec)
            if pd.isna(row.entry_date) or np.datetime64(row.entry_date,'D')!=dates[entry]:continue
            planned=lookup.get(np.datetime64(row.scheduled_exit_date,'D')) if pd.notna(row.scheduled_exit_date) else None
            if planned is None:rec['status']='outside_period';continue
            if pd.isna(row.entry_tradable) or not bool(row.entry_tradable) or not np.isfinite(row.entry_adj_open) or row.entry_adj_open<=0:continue
            if not np.isfinite(row.amount_yuan) or row.amount_yuan<=0:rec['status']='missing_liquidity';continue
            participation=budget/row.amount_yuan
            if participation>participation_cap:rec['status']='capacity_rejected';continue
            if not np.isfinite(row.volatility_20d) or row.volatility_20d<=0:rec['status']='missing_risk';continue
            impact=impact_coefficient*row.volatility_20d*np.sqrt(participation)
            buy=cost_bps/10000*.25+impact;sell=cost_bps/10000*.75+impact
            if sell>=1:rec['status']='invalid_cost';continue
            code=int(row.code_idx);actual=int(next_sellable[code,planned]);shares=budget/(row.entry_adj_open*(1+buy))
            spent=budget
            if round_lots:
                raw_price=row.entry_raw_open
                if not np.isfinite(raw_price) or raw_price<=0:rec['status']='missing_raw_open';continue
                quantity=buy_quantity(row.ts_code,budget/(raw_price*(1+buy)),row.entry_date)
                if quantity<=0:rec['status']='below_minimum_quantity';continue
                notional=quantity*raw_price;spent=notional*(1+buy)
                assert spent<=budget+1e-7
                shares=notional/row.entry_adj_open
                rec.update(allocated_budget=budget,budget=spent,buy_raw_quantity=quantity,entry_raw_open=raw_price)
            stop=actual if actual>=0 else n;values=shares*close[code,entry:stop].astype(float)
            if not np.isfinite(values).all():raise ValueError('Missing holding price')
            cash[slot,entry:]-=spent;held[slot,entry:stop]+=values
            sell_notional=float(shares*close[code,actual]) if actual>=0 else 0.
            if actual>=0:
                if not np.isfinite(sell_notional):raise ValueError('Missing sale price')
                cash[slot,actual:]+=sell_notional*(1-sell)
            rec.update(status='filled',entry_date=str(dates[entry]),scheduled_exit_date=str(dates[planned]),
                actual_exit_date=str(dates[actual]) if actual>=0 else None,participation=float(participation),
                buy_notional=float(shares*row.entry_adj_open),sell_notional=sell_notional,
                buy_cost_amount=float(shares*row.entry_adj_open*buy),sell_cost_amount=sell_notional*sell,
                buy_cost_rate=float(buy),sell_cost_rate=float(sell),unresolved_at_end=actual<0,delayed_exit=actual<0 or actual>planned)
    if cash.min()<-1e-5:raise ValueError('Negative cash')
    nav=(cash+held).sum(0)/aum;ret=nav/np.r_[1.,nav[:-1]]-1
    filled=[r for r in records if r['status']=='filled'];traded=sum(r['buy_notional']+r['sell_notional'] for r in filled)
    metrics=dict(net_return=float(nav[-1]-1),max_drawdown=float(np.min(nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1)),
        annualized_two_way_turnover=float(traded/(aum*nav.mean())*252/n),orders=len(records),filled_orders=len(filled),
        total_cost_amount=sum(r['buy_cost_amount']+r['sell_cost_amount'] for r in filled),
        unresolved_positions=sum(r['unresolved_at_end'] for r in filled),delayed_exit_orders=sum(r['delayed_exit'] for r in filled),
        empty_cash_days=empty_days,mean_cash_fraction=float(np.mean(cash.sum(0)/(aum*nav))),cost_bps=cost_bps,
        execution_model='five sleeves; only available prior-close cash reused; unsold lots retained; fractional shares')
    return metrics,pd.DataFrame(dict(date=dates,nav=nav,daily_return=ret,cash=cash.sum(0),holdings=held.sum(0))),pd.DataFrame(records)


def buy_quantity(code,affordable,entry_date):
    if code.endswith('.BJ'):
        if pd.Timestamp(entry_date)<pd.Timestamp('2021-11-15'):return 0
        minimum,step=100,1
    elif code.startswith('688') and code.endswith('.SH'):minimum,step=200,1
    elif code.endswith(('.SH','.SZ')):minimum,step=100,100
    else:return 0
    quantity=int(np.floor(affordable/step))*step
    return quantity if quantity>=minimum else 0
