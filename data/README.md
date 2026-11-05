# Local data requirements

Vendor market data are not included. Obtain your own authorized data access and prepare the files locally under `quant_data/`. Source scripts document the field-level requirements. The historical local cache is not an unbiased whole-market sample.

Expected main locations:

```text
quant_data/combined/stock_basic.csv
quant_data/processed/parquet/<interface>/year=YYYY/*.parquet
quant_data/processed/features_50/feature_year=YYYY/*.parquet
quant_data/processed/targets_5d/signal_year=YYYY/*.parquet
quant_data/processed/v2/local_features_2011_2023_stable/feature_year=YYYY/*.parquet
quant_data/processed/v2/<experiment>/...
```

Execution uses daily prices, adjustment factors, price limits and stock metadata. Other factor groups need their corresponding turnover, money-flow, valuation and financial fields. Read the feature audit and builder code for exact availability rules.

The scripts retain research-stage dependencies on source manifests, per-fold preprocessing, saved sample keys and preceding experiment outputs. These are not bundled. Full historical reproduction therefore requires rebuilding the sequence in the research index and adapting locally recorded paths in generated manifests.

Do not fabricate missing delisted histories or financial data. Preserve the documented missing-data gates. Never commit vendor credentials or raw caches.
