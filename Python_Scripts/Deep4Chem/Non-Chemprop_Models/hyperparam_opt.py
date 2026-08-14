import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform



# loading Deep4Chem (descriptors) training data subset

print('Loading and preprocessing dataset...')
d4c_train = pd.read_csv('d4c_train_rdkit_descriptors_filtered.csv')



# checking for non-finite and excessively large descriptor columns

problem_cols = []

for col in d4c_train.columns:
    if not np.isfinite(d4c_train[col]).all():
        problem_cols.append(col)
    elif d4c_train[col].abs().max() > 1e6:
        problem_cols.append(col)

print('Removing columns:')
print(problem_cols)

# removing problematic descriptors

d4c_train = d4c_train.drop(columns=problem_cols)

# splitting Deep4Chem training subset after cleaning

feature_cols = d4c_train.columns[:-2]
target_cols = ['Absorption max (nm)', 'log(e/mol-1 dm3 cm-1)']

X = d4c_train[feature_cols]
y = d4c_train[target_cols]



# Random Forest model

print('Optimising Random Forest hyperparameters...')

rf = MultiOutputRegressor(
    RandomForestRegressor(
        random_state=0,
        n_jobs=-1
    )
)

rf_hyperparams = {
    'estimator__n_estimators': randint(100, 1001),
    'estimator__max_features': ['sqrt', 'log2', 0.3, 0.5],
    'estimator__max_depth': randint(1, 41),
    'estimator__min_samples_split': randint(2, 11),
    'estimator__min_samples_leaf': randint(1, 5)
}

rf_search = RandomizedSearchCV(
    rf,
    rf_hyperparams,
    n_iter=30,
    cv=10,
    scoring='neg_root_mean_squared_error',
    random_state=0,
    n_jobs=-1
)

rf_search.fit(X, y)

print(rf_search.best_params_)
print(-rf_search.best_score_)



# LightGBM model

print('\nOptimising LightGBM hyperparameters...')

lgbm = MultiOutputRegressor(
    LGBMRegressor(
        random_state=0,
        n_jobs=-1
    )
)

lgbm_hyperparams = {
    'estimator__n_estimators': randint(100, 1001),
    'estimator__learning_rate': uniform(0.01, 0.30),
    'estimator__num_leaves': randint(1, 501),
    'estimator__max_depth': randint(1, 41),
    'estimator__subsample': uniform(0.5, 1.0),
    'estimator__colsample_bytree': uniform(0.5, 1.0)
}

lgbm_search = RandomizedSearchCV(
    lgbm,
    lgbm_hyperparams,
    n_iter=30,
    cv=10,
    scoring='neg_root_mean_squared_error',
    random_state=0,
    n_jobs=-1
)

lgbm_search.fit(X, y)

print(lgbm_search.best_params_)
print(-lgbm_search.best_score_)



# XGBoost model

print('\nOptimising XGBoost hyperparameters...')

xgb = MultiOutputRegressor(
    XGBRegressor(
        random_state=0,
        n_jobs=-1
    )
)

xgb_hyperparams = {
    'estimator__n_estimators': randint(100, 1001),
    'estimator__learning_rate': uniform(0.01, 0.30),
    'estimator__max_depth': randint(1, 41),
    'estimator__subsample': uniform(0.5, 1.0),
    'estimator__colsample_bytree': uniform(0.5, 1.0),
    'estimator__min_child_weight': uniform(0.1, 10.0),
    'estimator__gamma': uniform(0.0, 5.0)
}

xgb_search = RandomizedSearchCV(
    xgb,
    xgb_hyperparams,
    n_iter=30,
    cv=10,
    scoring='neg_root_mean_squared_error',
    random_state=0,
    n_jobs=-1
)

xgb_search.fit(X, y)

print(xgb_search.best_params_)
print(-xgb_search.best_score_)



# saving random search results

results = pd.DataFrame({
    'Model':[
        'Random Forest',
        'LightGBM',
        'XGBoost'
    ],

    'Best RMSE':[
        -rf_search.best_score_,
        -lgbm_search.best_score_,
        -xgb_search.best_score_
    ],

    'Best Parameters':[
        str(rf_search.best_params_),
        str(lgbm_search.best_params_),
        str(xgb_search.best_params_)
    ]
})

results.to_csv('hyperparam_opt_results.csv', index=False)

print(results)
print('Finished')
