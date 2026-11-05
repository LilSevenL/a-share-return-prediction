"""Read-only artifact verification against V1 and an independent 20-day example."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import json
import sys
from pathlib import Path
import duckdb
import numpy as np
import pyarrow.parquet as pq
from features_500 import ROOT, BASE_FEATURES, FEATURE_COLUMNS


def main():
    output = Path(sys.argv[1]).resolve()
    files = sorted(output.glob('feature_year=*/*.parquet'))
    assert files, 'No feature outputs'
    con = duckdb.connect()
    con.read_parquet([str(p) for p in files], hive_partitioning=False).create_view('result')
    schema = pq.read_schema(files[0])
    assert [n for n in schema.names if n in FEATURE_COLUMNS] == FEATURE_COLUMNS
    rows, distinct = con.execute('SELECT count(*),count(DISTINCT (ts_code,signal_date)) FROM result').fetchone()
    assert rows == distinct
    finite_expr = '+'.join(f'CAST(coalesce(isfinite({n}),false) AS INT)' for n in FEATURE_COLUMNS)
    assert con.execute(f'SELECT count(*) FROM result WHERE feature_valid_count != ({finite_expr})').fetchone()[0] == 0
    years = sorted({int(p.parent.name.split('=')[1]) for p in files})
    original = [str(p) for year in years for p in (ROOT/f'quant_data/processed/features_50/feature_year={year}').glob('*.parquet')]
    con.read_parquet(original, hive_partitioning=False).create_view('original')
    unequal = ' OR '.join(f'r.{name} IS DISTINCT FROM o.{name}' for name in BASE_FEATURES)
    matched, mismatches = con.execute(f'''SELECT count(*),sum(CASE WHEN {unequal} THEN 1 ELSE 0 END)
        FROM result r JOIN original o USING(ts_code,signal_date)''').fetchone()
    assert matched == rows and mismatches == 0, (matched,rows,mismatches)
    # Independently select the last 20 complete stock observations in the real
    # sample. This example requires 20 consecutive MARKET sessions too.
    code, day, actual = con.execute('''SELECT ts_code,signal_date,cond_volume_high_close_sum_20d
        FROM result WHERE cond_volume_high_close_sum_20d IS NOT NULL
        ORDER BY signal_date DESC,ts_code LIMIT 1''').fetchone()
    daily = [str(p) for year in (day.year-1,day.year) for p in (ROOT/f'quant_data/processed/parquet/daily/year={year}').glob('*.parquet')]
    history = [str(p) for year in (day.year-1,day.year) for p in (ROOT/f'quant_data/processed/features_50/feature_year={year}').glob('*.parquet')]
    con.read_parquet(daily,hive_partitioning=False).create_view('daily')
    con.read_parquet(history,hive_partitioning=False).create_view('history')
    records = con.execute('''SELECT h.signal_date,d.vol,h.logret_1d FROM history h JOIN daily d
        ON h.ts_code=d.ts_code AND h.signal_date=d.trade_date
        WHERE h.ts_code=? AND h.signal_date<=? ORDER BY h.signal_date DESC LIMIT 20''',[code,day]).fetchall()
    assert len(records)==20
    records = sorted(records)
    sessions = con.execute('SELECT count(DISTINCT trade_date) FROM daily WHERE trade_date BETWEEN ? AND ?',[records[0][0],records[-1][0]]).fetchone()[0]
    assert sessions == 20
    chosen = sorted(records, key=lambda row:(row[1],row[0]))[-10:]
    expected = sum(r[2] for r in chosen)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    report = json.loads((output/'feature_report.json').read_text(encoding='utf-8'))
    verification = dict(status='passed',rows=rows,feature_count=500,unique_keys=True,
        original_50_unchanged=True,valid_counts_correct=True,
        columns_with_any_finite_value=report['columns_with_any_finite_value'],
        independent_example=dict(stock=code,signal_date=str(day),factor='cond_volume_high_close_sum_20d',
                                 actual=actual,expected=expected,selected_dates=[str(r[0]) for r in chosen]))
    (output/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(verification,ensure_ascii=False,indent=2))
    con.close()


if __name__ == '__main__':
    main()
