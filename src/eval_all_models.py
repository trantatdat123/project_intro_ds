import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

import src.config as config
from src.features.feature_builder import build_ablation_features
from src.evaluation.regression_metrics import compute_regression_metrics

df_reg = pd.read_csv(config.PROCESSED_DATA_DIR / 'regression_data.csv')
sbert_all = np.load(config.EMBEDDINGS_DIR / 'sbert_embeddings.npy')
cleaned_df = pd.read_csv(config.PROCESSED_DATA_DIR / 'cleaned_jobs.csv')

id_to_idx = {uid: idx for idx, uid in enumerate(cleaned_df['unique_job_id'])}
reg_indices = [id_to_idx[uid] for uid in df_reg['unique_job_id']]
sbert_reg = sbert_all[reg_indices]

y = (df_reg['salary_mean_vnd'] / 1_000_000.0).values
indices = np.arange(len(df_reg))
train_idx, temp_idx = train_test_split(indices, test_size=0.20, random_state=42)
val_rel_idx, test_rel_idx = train_test_split(np.arange(len(temp_idx)), test_size=0.50, random_state=42)
val_idx = temp_idx[val_rel_idx]
test_idx = temp_idx[test_rel_idx]

df_train = df_reg.iloc[train_idx].reset_index(drop=True)
df_val = df_reg.iloc[val_idx].reset_index(drop=True)
df_test = df_reg.iloc[test_idx].reset_index(drop=True)

y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
sbert_train, sbert_val, sbert_test = sbert_reg[train_idx], sbert_reg[val_idx], sbert_reg[test_idx]

ablation_sets = build_ablation_features(
    df_train, df_val, df_test, sbert_train, sbert_val, sbert_test
)

all_factories = {
    'Dummy (Median Baseline)': lambda: DummyRegressor(strategy='median'),
    'Linear Regression': lambda: LinearRegression(),
    'Ridge Regression': lambda: Ridge(alpha=10.0, random_state=42),
    'Random Forest Regressor': lambda: RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1),
    'Extra Trees Regressor': lambda: ExtraTreesRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1),
    'XGBoost Regressor': lambda: XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)
}

print('=== 1. TESTING ALL 6 MODELS ON E1, E2, E3, E4 ===')
abl_rows = []
for c_key in ['E1', 'E2', 'E3', 'E4']:
    X_tr = ablation_sets[c_key]['train']
    X_va = ablation_sets[c_key]['val']
    X_te = ablation_sets[c_key]['test']
    for m_name, factory in all_factories.items():
        m = factory()
        m.fit(X_tr, y_train)
        val_p = m.predict(X_va)
        val_m = compute_regression_metrics(y_val, val_p, m_name, config_name=c_key)
        test_p = m.predict(X_te)
        test_m = compute_regression_metrics(y_test, test_p, m_name, config_name=c_key)
        test_m['Val MAE (Million VND)'] = val_m['MAE (Million VND)']
        test_m['Val R2 Score'] = val_m['R2 Score']
        abl_rows.append(test_m)

abl_df = pd.DataFrame(abl_rows)
print(abl_df.to_string(index=False))

print('\n=== 2. TESTING ALL 6 MODELS ON TEMPORAL VALIDATION (<=2025 -> 2026) ===')
temporal_train_mask = (df_reg['year_clean'] <= 2025).values
temporal_test_mask = (df_reg['year_clean'] == 2026).values

df_temp_tr = df_reg[temporal_train_mask].reset_index(drop=True)
df_temp_te = df_reg[temporal_test_mask].reset_index(drop=True)
y_temp_tr = y[temporal_train_mask]
y_temp_te = y[temporal_test_mask]
sbert_temp_tr = sbert_reg[temporal_train_mask]
sbert_temp_te = sbert_reg[temporal_test_mask]

temp_features = build_ablation_features(
    df_temp_tr, df_temp_te, df_temp_te, sbert_temp_tr, sbert_temp_te, sbert_temp_te
)
X_temp_tr_e4 = temp_features['E4']['train']
X_temp_te_e4 = temp_features['E4']['test']

temp_rows = []
for m_name, factory in all_factories.items():
    tm = factory()
    tm.fit(X_temp_tr_e4, y_temp_tr)
    preds_tm = tm.predict(X_temp_te_e4)
    tm_m = compute_regression_metrics(y_temp_te, preds_tm, m_name, config_name='Temporal (<=2025 -> 2026)')
    temp_rows.append(tm_m)

temp_df = pd.DataFrame(temp_rows)
print(temp_df.to_string(index=False))
