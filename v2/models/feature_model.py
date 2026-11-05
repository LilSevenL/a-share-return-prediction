"""Bind the balanced-500 schema, fitted preprocessing and coefficients together."""

# Support both direct script execution and package imports after directory grouping.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT as _REPO_ROOT, source_path as _source_path

import hashlib
import json
import pandas as pd
from balanced_features import CATALOG,COLUMNS
from research_controls import transform_features
from linear_losses import predict_linear_loss


def catalog_hash():
    return hashlib.sha256(json.dumps(CATALOG,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()


def validate_input(frame):
    if not isinstance(frame,pd.DataFrame):raise ValueError('Named DataFrame required for the 500-factor model')
    if frame.columns.duplicated().any():raise ValueError('Duplicate input columns')
    missing=set(COLUMNS)-set(frame.columns)
    if missing:raise ValueError(f'Balanced-500 input missing {len(missing)} factors: {sorted(missing)[:8]}')


def bundle_model(model,preprocessing):
    if preprocessing['names']!=COLUMNS:raise ValueError('Cannot bind old-50 preprocessing to balanced-500 model')
    active=[n for n,keep in zip(COLUMNS,preprocessing['active']) if keep]
    if active!=preprocessing['active_names'] or len(active)!=len(model['coefficient']):
        raise ValueError('Active factor/coefficient mismatch')
    return dict(format='balanced_500_linear_v1',catalog_sha256=catalog_hash(),candidate_count=len(COLUMNS),
        active_feature_names=active,preprocessing=preprocessing,linear_model=model,
        coefficient_by_name=dict(zip(active,model['coefficient'])))


def predict_frame(bundle,frame):
    validate_input(frame)
    if bundle.get('format')!='balanced_500_linear_v1' or bundle.get('catalog_sha256')!=catalog_hash():
        raise ValueError('Model/catalog version mismatch; refit or load its exact version')
    stats=bundle['preprocessing'];model=bundle['linear_model']
    verified=bundle_model(model,stats)
    if verified['active_feature_names']!=bundle['active_feature_names'] or verified['coefficient_by_name']!=bundle['coefficient_by_name']:
        raise ValueError('Model feature metadata mismatch')
    return predict_linear_loss(model,transform_features(frame,stats))
