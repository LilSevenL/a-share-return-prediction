import unittest
from unittest.mock import patch
from types import SimpleNamespace
import pandas as pd
from repair_historical_universe import historical_universe,eligible_on,SafeSDK


class UniverseRepairTests(unittest.TestCase):
    def test_delisted_included_only_during_historical_interval(self):
        frame=pd.DataFrame([
            dict(ts_code='old',list_date='20000101',delist_date='20220601',list_status='D'),
            dict(ts_code='new',list_date='20230101',delist_date=None,list_status='L'),
            dict(ts_code='future',list_date='20250101',delist_date=None,list_status='L')])
        universe,excluded=historical_universe([frame]);self.assertTrue(excluded.empty)
        self.assertEqual(set(eligible_on(universe,'2022-01-01').ts_code),{'old'})
        self.assertEqual(set(eligible_on(universe,'2023-06-01').ts_code),{'new'})

    def test_stale_active_metadata_does_not_erase_delisting(self):
        old=pd.DataFrame([dict(ts_code='x',list_date='20000101',delist_date=None,list_status='L')])
        latest=old.copy();latest['delist_date']='20220101';latest['list_status']='D'
        universe,_=historical_universe([old,latest])
        self.assertTrue(eligible_on(universe,'2023-01-01').empty)

    def test_missing_delisting_date_is_not_silently_open_ended(self):
        frame=pd.DataFrame([dict(ts_code='x',list_date='20000101',delist_date=None,list_status='D')])
        universe,excluded=historical_universe([frame])
        self.assertTrue(universe.empty);self.assertEqual(len(excluded),1)

    def test_conflicting_listing_dates_require_review(self):
        frame=pd.DataFrame([
            dict(ts_code='x',list_date='20000101',delist_date=None,list_status='L'),
            dict(ts_code='x',list_date='20100101',delist_date=None,list_status='L')])
        with self.assertRaisesRegex(ValueError,'Conflicting listing dates'):historical_universe([frame])

    def test_sdk_403_cannot_be_recorded_as_empty_dataset(self):
        import tushare as ts
        import tushare.pro.client as client
        sdk=SafeSDK.__new__(SafeSDK);sdk.pro=ts.pro_api('synthetic-test-only')
        sdk.last=0.;sdk.last_http_status=None
        with patch.object(client.requests,'post',return_value=SimpleNamespace(status_code=403)) as request:
            with self.assertRaisesRegex(RuntimeError,'http=403'):sdk.fetch('daily',ts_code='test')
            self.assertEqual(request.call_count,1)


if __name__=='__main__':unittest.main()
