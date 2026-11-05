"""Read-only, label-free audit of an existing balanced-500 feature build.

Writes diagnostics separately; never uses audit-period statistics to fit a model.
Raw comparisons use independent formulas, not the production feature calculator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import duckdb
from threadpoolctl import threadpool_limits
from balanced_features import COLUMNS,RECORDS,FINANCIAL_NAMES
from build_balanced_features import save_json,save_frame

ROOT=Path(__file__).resolve().parents[1]


def factor_bounds(name,family):
    if name.startswith('return_kurtosis_'):
        w=int(name.removeprefix('return_kurtosis_').removesuffix('d'))
        return -2*(w-1)/(w-3),float(w)
    if family=='I':return -.25,.25
    if name.startswith(('trend_r2_','path_efficiency_','up_day_share_','longest_up_streak_',
        'longest_down_streak_','overnight_reversal_share_','intraday_abs_share_','direction_switch_share_',
        'return_ts_rank_','maximum_drawdown_','ulcer_index_','large_move_share_','zero_return_share_',
        'volume_hhi_','up_volume_share_','large_net_positive_share_','aggregate_large_buy_share_',
        'residual_positive_share_','residual_ts_rank_')) or name in ('close_location','upper_shadow','lower_shadow',
        'large_buy_share','extra_large_buy_share','order_size_hhi','negative_earnings_flag') or '_ts_rank_' in name:
        return 0.,1.
    if '_corr_' in name or name.startswith('return_autocorr_') or name.startswith('close_pressure_') or name=='signed_body':return -1.,1.
    if name.startswith(('volatility_','upside_rms_','downside_rms_','return_mad_','parkinson_vol_',
        'garman_klass_vol_','amount_cv_','log_volume_std_','amihud_proxy_','turnover_std_',
        'large_net_std_','gross_order_imbalance_std_','residual_volatility_')) or name=='downside_volatility_20d':return 0.,None
    return None,None


def compare(actual,expected,keys,name,check):
    a=np.asarray(actual,float);b=np.asarray(expected,float);af=np.isfinite(a);bf=np.isfinite(b)
    mismatch=(af!=bf)|((af&bf)&~np.isclose(a,b,rtol=2e-5,atol=2e-7))
    examples=[]
    for i in np.flatnonzero(mismatch)[:5]:
        examples.append(dict(ts_code=str(keys.iloc[i].ts_code),signal_date=str(keys.iloc[i].signal_date),
            actual=float(a[i]) if af[i] else None,expected=float(b[i]) if bf[i] else None))
    return dict(name=name,check=check,rows=len(a),both_finite=int((af&bf).sum()),
        null_mismatches=int((af!=bf).sum()),value_mismatches=int((mismatch&(af&bf)).sum()),
        mismatches=int(mismatch.sum()),max_abs_error=float(np.max(np.abs(a[af&bf]-b[af&bf]))) if (af&bf).any() else None,
        examples=examples)


def independent_checks(con,frame,config,output):
    selected=pd.DataFrame({'ts_code':sorted(frame.ts_code.unique())});con.register('selected_codes',selected)
    raw_quality={}
    for name,items in config['inputs'].items():
        if name=='base':continue
        paths=[v['path'] for v in items]
        con.read_parquet(paths,hive_partitioning=False).create_view('source_'+name)
        con.execute(f'CREATE TEMP TABLE audit_{name} AS SELECT s.* FROM source_{name} s JOIN selected_codes USING(ts_code)')
        raw_quality[name]=con.execute(f'''SELECT count(*) row_count,count(*)-count(DISTINCT(ts_code,trade_date)) duplicate_keys
            FROM audit_{name}''').fetch_df().iloc[0].to_dict()
    calendar=pd.DatetimeIndex(con.execute('SELECT DISTINCT trade_date FROM source_daily ORDER BY trade_date').fetch_df().iloc[:,0])
    basic=pd.read_csv(ROOT/'quant_data/combined/stock_basic.csv',dtype=str)
    delisted=pd.read_csv(ROOT/'quant_data/extra_cache/stock_basic/status_D.csv',dtype=str)
    listed=pd.to_datetime(delisted.list_date,format='%Y%m%d',errors='coerce')
    ended=pd.to_datetime(delisted.delist_date,format='%Y%m%d',errors='coerce')
    eligible=delisted[(listed<=calendar.max())&(ended>=calendar.min())]
    con.register('delisted_overlap',eligible[['ts_code']])
    present=con.execute('SELECT DISTINCT d.ts_code FROM source_daily d JOIN delisted_overlap USING(ts_code)').fetch_df()
    universe=dict(delisted_metadata_count=len(delisted),delisted_overlapping_raw_period=len(eligible),
        overlap_codes_present_in_daily=len(present),raw_period=[str(calendar.min()),str(calendar.max())],
        examples_missing=eligible[~eligible.ts_code.isin(present.ts_code)].ts_code.head(10).tolist(),
        main_stock_basic_status_counts=basic.list_status.value_counts().to_dict(),
        caveat='Cached status snapshot is not a complete historical-universe certification',
        source='preprocess_data.py load_universe keeps only status L; extra_cache/stock_basic/status_D.csv')
    save_json(output/'universe_audit.json',universe)
    flowcols=','.join(f'm.{side}_{size}_amount' for side in ('buy','sell') for size in ('sm','md','lg','elg'))
    raw=con.execute(f'''SELECT d.ts_code,d.trade_date signal_date,d.open,d.high,d.low,d.close,d.pre_close,
        d.vol,d.amount,a.adj_factor,b.turnover_rate,b.pe_ttm,b.pb,b.ps_ttm,{flowcols},m.net_mf_amount
        FROM audit_daily d LEFT JOIN audit_adj_factor a USING(ts_code,trade_date)
        LEFT JOIN audit_daily_basic b USING(ts_code,trade_date) LEFT JOIN audit_moneyflow m USING(ts_code,trade_date)
        ORDER BY ts_code,signal_date''').fetch_df()
    raw_quality['joined_rows']=len(raw)
    raw_quality['invalid_ohlc']=int(((raw[['open','high','low','close']]<=0).any(axis=1)|(raw.high<raw[['open','close','low']].max(axis=1)-1e-8)|(raw.low>raw[['open','close','high']].min(axis=1)+1e-8)).sum())
    raw_quality['missing_or_nonpositive_adj']=int((raw.adj_factor.isna()|(raw.adj_factor<=0)).sum())
    raw_quality['negative_amount_or_volume']=int(((raw.amount<0)|(raw.vol<0)).sum())
    checks=[];formula_frames=[]
    for code,part in raw.groupby('ts_code',sort=True):
        d=part.set_index('signal_date').reindex(calendar);c=d.close*d.adj_factor;o=d.open*d.adj_factor
        r=np.log(c/c.shift());a=d.amount*1000;v=d.vol;t=d.turnover_rate
        div=lambda x,y:x/y.replace(0,np.nan)
        expected={}
        for w in (1,2,3,5,10,15,20,40,60,120,250):expected[f'logret_{w}d']=r.rolling(w,min_periods=w).sum()
        for w in (5,10,20,40,60,120):expected[f'volatility_{w}d']=r.rolling(w,min_periods=w).std(ddof=1)
        expected['intraday_logret']=np.log(c/o);expected['overnight_logret']=np.log(o/c.shift())
        expected['large_order_net_ratio']=div((d.buy_lg_amount+d.buy_elg_amount-d.sell_lg_amount-d.sell_elg_amount)*10000,a)
        expected['net_mf_ratio']=div(d.net_mf_amount*10000,a)
        for w in (5,20):expected[f'amount_ratio_{w}d']=div(a,a.rolling(w,min_periods=w).mean())-1
        expected['volume_ratio_20d']=div(v,v.rolling(20,min_periods=20).mean())-1
        expected['turnover_zscore_20d']=div(t-t.rolling(20,min_periods=20).mean(),t.rolling(20,min_periods=20).std())
        expected['net_mf_ratio_5d']=expected['net_mf_ratio'].rolling(5,min_periods=5).mean()
        expected['log_amount']=np.log(d.amount.where(d.amount>0))
        expected['log_mean_amount_20d']=np.log(a.rolling(20,min_periods=20).mean())
        for name,den in [('earnings_yield',d.pe_ttm),('book_to_price',d.pb),('sales_yield',d.ps_ttm)]:expected[name]=(1/den).where(den.abs()>1e-12)
        for w in (5,10,20,40,60):expected[f'aggregate_net_flow_ratio_{w}d']=div((d.net_mf_amount*10000).rolling(w,min_periods=w).sum(),a.rolling(w,min_periods=w).sum())
        # Independent explicit top-half selection for every non-correlation C factor.
        selectors={'volume':v,'amount':a,'turnover':t,'large_buy':d.buy_lg_amount+d.buy_elg_amount,
            'large_net':d.buy_lg_amount+d.buy_elg_amount-d.sell_lg_amount-d.sell_elg_amount}
        for rec in RECORDS:
            if rec['family']!='C' or '_corr_' in rec['name']:continue
            name=rec['name'];match=re.match(r'cond_(.+)_(high|low)_(close|intraday|overnight)_(sum|std|downside_rms)_(\d+)d',name)
            if not match:raise ValueError(f'Unrecognized conditional factor {name}')
            selector,side,ret,stat,w=match.groups();w=int(w);s=selectors[selector].to_numpy()
            rv={'close':r,'intraday':expected['intraday_logret'],'overnight':expected['overnight_logret']}[ret].to_numpy()
            values=np.full(len(d),np.nan)
            for end in range(w-1,len(d)):
                window=s[end-w+1:end+1]
                if not np.isfinite(window).all() or np.ptp(window)==0:continue
                order=sorted(range(w),key=lambda j:(window[j],j));ix=order[-w//2:] if side=='high' else order[:w//2]
                chosen=rv[end-w+1:end+1][ix]
                if np.isfinite(chosen).all():values[end]=chosen.sum() if stat=='sum' else chosen.std(ddof=1) if stat=='std' else np.sqrt(np.mean(np.minimum(chosen,0)**2))
            expected[name]=pd.Series(values,index=d.index)
        e=pd.DataFrame(expected,index=d.index);e['ts_code']=code;e.index.name='signal_date'
        formula_frames.append(e.reset_index())
    all_expected=pd.concat(formula_frames,ignore_index=True)
    merged=frame[['ts_code','signal_date']].merge(all_expected,on=['ts_code','signal_date'],validate='one_to_one')
    for name in all_expected.columns:
        if name in COLUMNS:checks.append(compare(frame[name],merged[name],frame[['ts_code','signal_date']],name,'independent_raw_formula_market_calendar'))
    # Accounting identity is tested wherever both components exist.
    mask=frame[['logret_1d','intraday_logret','overnight_logret']].notna().all(axis=1)
    checks.append(compare(frame.loc[mask,'logret_1d'],frame.loc[mask,'intraday_logret']+frame.loc[mask,'overnight_logret'],frame.loc[mask,['ts_code','signal_date']].reset_index(drop=True),'logret_identity','close = overnight + intraday'))
    save_json(output/'raw_checks.json',dict(quality=raw_quality,comparisons=checks,
        scope='80-stock build universe; raw lookback from build manifest; independent formulas for selected families'))
    return checks,raw_quality


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features-dir',default=str(ROOT/'quant_data/processed/v2/balanced_500_integration'))
    p.add_argument('--output-dir',default=str(ROOT/'quant_data/processed/v2/factor_audit_20261024'))
    args=p.parse_args();directory=Path(args.features_dir).resolve();output=Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT/'quant_data/processed/v2') or output==directory:raise ValueError('Separate V2 audit directory required')
    output.mkdir(parents=True,exist_ok=True)
    config=json.loads((directory/'build_config.json').read_text(encoding='utf-8'))
    report=json.loads((directory/'feature_report.json').read_text(encoding='utf-8'))
    first=config['args']['start_year'];last=config['args']['end_year']
    if last>2023:raise ValueError('This development audit excludes 2024 confirmation and 2025+ Test')
    for items in config['inputs'].values():
        for item in items:
            years=re.findall(r'(?:feature_year|year)=(\d{4})',item['path'])
            if not years or max(map(int,years))>2023:raise ValueError('Unsafe input partition')
    fps=[f for year in range(first,last+1) for f in sorted(directory.glob(f'feature_year={year}/*.parquet'))]
    frame=pd.concat([pd.read_parquet(f) for f in fps],ignore_index=True).sort_values(['ts_code','signal_date']).reset_index(drop=True)
    if set(COLUMNS)-set(frame):raise ValueError('Missing candidate columns')
    if len(frame)>200000:raise ValueError('Audit currently bounded to 200k rows; extend aggregation before larger runs')
    print(f'Auditing {len(frame)} rows, {frame.ts_code.nunique()} stocks, {len(COLUMNS)} candidates',flush=True)
    blocked=set(report['blocked_features']);provenance=[]
    for name,digest in config['sources'].items():
        path=Path(__file__).parent/('research_design/'+name if name.endswith('.json') else name)
        provenance.append(dict(source=name,hash_matches=hashlib.sha256(path.read_bytes()).hexdigest()==digest))
    changed_inputs=[]
    for table,items in config['inputs'].items():
        for item in items:
            f=Path(item['path']);st=f.stat()
            if st.st_size!=item['bytes'] or st.st_mtime_ns!=item['mtime_ns']:changed_inputs.append(str(f))
    record_by_name={r['name']:r for r in RECORDS};annual=[];quality=[]
    for name in COLUMNS:
        s=frame[name].astype(float);finite=np.isfinite(s);v=s[finite];lo,hi=factor_bounds(name,record_by_name[name]['family'])
        violations=((v<lo-1e-6) if lo is not None else pd.Series(False,index=v.index))|((v>hi+1e-6) if hi is not None else pd.Series(False,index=v.index))
        quant=v.quantile([.001,.01,.5,.99,.999]).tolist() if len(v) else [None]*5
        daily=frame.assign(_value=s.where(finite)).groupby('signal_date')._value.agg(['count','nunique'])
        stats=dict(name=name,family=record_by_name[name]['family'],coverage=float(finite.mean()),missing=int(s.isna().sum()),
            infinite=int(np.isinf(s).sum()),distinct=int(v.nunique()),minimum=float(v.min()) if len(v) else None,
            maximum=float(v.max()) if len(v) else None,q001=quant[0],q01=quant[1],median=quant[2],q99=quant[3],q999=quant[4],
            bound_violations=int(violations.sum()),constant_cross_section_days=int(((daily['count']>=20)&(daily['nunique']<=1)).sum()))
        quality.append(stats)
        for year,idx in frame.groupby(frame.signal_date.dt.year).groups.items():
            block=s.loc[idx];f=np.isfinite(block);vv=block[f]
            annual.append(dict(name=name,year=int(year),rows=len(block),coverage=float(f.mean()),distinct=int(vv.nunique()),
                median=float(vv.median()) if len(vv) else None,q01=float(vv.quantile(.01)) if len(vv) else None,q99=float(vv.quantile(.99)) if len(vv) else None))
    print('Quality statistics complete; checking correlations and independent raw formulas',flush=True)
    # Pairwise complete observations, diagnostic only; no imputation or feature elimination.
    sample=frame[COLUMNS].sample(n=min(12000,len(frame)),random_state=42)
    corr=sample.corr(min_periods=1000)
    pairs=[]
    for i,a in enumerate(COLUMNS):
        for j in range(i+1,len(COLUMNS)):
            value=corr.iloc[i,j]
            if np.isfinite(value) and abs(value)>=.98:
                b=COLUMNS[j];pairs.append(dict(a=a,b=b,correlation=float(value),pairs=int((sample[a].notna()&sample[b].notna()).sum())))
    save_frame(output/'annual_quality.parquet',pd.DataFrame(annual));save_json(output/'high_correlations.json',pairs)
    con=duckdb.connect();con.execute('SET threads=4');con.execute("SET memory_limit='4GB'")
    checks,raw_quality=independent_checks(con,frame,config,output);con.close()
    mismatches={r['name']:r for r in checks if r['mismatches']}
    review=set(mismatches)
    # Propagate a numerical issue through explicit cross-sectional parent dependencies.
    for rec in RECORDS:
        if rec['family']=='I' and any(x in review for x in rec['source'].split('; ')):review.add(rec['name'])
    for stats in quality:
        name=stats['name'];reasons=[]
        if name in blocked:status='隔离';reasons.append('财务历史版本及可比报告口径未验证')
        elif name=='negative_earnings_flag':status='隔离';reasons.append('daily_basic亏损PE为空，pe_ttm<0不能识别亏损；不得把缺失直接认作亏损')
        elif name in review:status='待修复或统一口径';reasons.append('独立复算差异或依赖受影响因子')
        elif stats['infinite'] or stats['bound_violations']:status='待核查';reasons.append('无穷值或理论边界异常')
        elif stats['coverage']<.8 or stats['distinct']<=1:status='待核查';reasons.append('覆盖率低于审计提示阈值80%或样本内常数')
        else:status='样本统计初检通过';reasons.append('不是全市场可用性或alpha有效性认证')
        stats.update(status=status,reasons=reasons,independently_recomputed=name in {r['name'] for r in checks})
    save_frame(output/'factor_quality.parquet',pd.DataFrame(quality));save_json(output/'factor_quality.json',quality)
    counts=pd.Series([r['status'] for r in quality]).value_counts().to_dict()
    summary=dict(scope='existing 80-stock 2022-2023 build; not full-market certification',rows=len(frame),stocks=int(frame.ts_code.nunique()),
        years=[first,last],candidate_count=500,status_counts=counts,duplicate_keys=int(frame.duplicated(['ts_code','signal_date']).sum()),
        infinite_cells=int(np.isinf(frame[COLUMNS].to_numpy(float)).sum()),source_hashes=provenance,changed_input_metadata=changed_inputs,
        independent_checks=len(checks),checks_with_mismatches=len(mismatches),mismatch_factors=sorted(mismatches),high_correlation_pairs=len(pairs),
        audit_only=True,labels_read=False,test_read=False,training_or_selection_performed=False,
        raw_quality=raw_quality,feature_files=[dict(path=str(f),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in fps])
    summary['audit_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary['feature_valid_count_mismatches']=int((frame.feature_valid_count.to_numpy()!=np.isfinite(frame[COLUMNS].to_numpy(float)).sum(axis=1)).sum())
    summary['universe_audit']=json.loads((output/'universe_audit.json').read_text(encoding='utf-8'))
    save_json(output/'summary.json',summary)
    lines=['# 500因子数据审计：逐项清单','','范围：现有80只股票、2022–2023；不是全市场审计。跨年统计仅作质量诊断，不用于选alpha或拟合预处理。', '',
        '| 因子 | 方向 | 覆盖率 | 唯一值数 | 边界异常 | 状态 | 独立原始数据复算 |','|---|---|---:|---:|---:|---|---|']
    for q in quality:lines.append(f"| {q['name']} | {q['family']} | {q['coverage']:.2%} | {q['distinct']} | {q['bound_violations']} | {q['status']} | {'是' if q['independently_recomputed'] else '否'} |")
    (output/'FACTOR_CHECKLIST.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('feature_files','raw_quality')},ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
