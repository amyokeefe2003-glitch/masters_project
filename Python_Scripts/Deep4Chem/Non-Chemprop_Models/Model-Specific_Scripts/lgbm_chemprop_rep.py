# imports

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import (mean_squared_error, mean_absolute_error)
import shap
import matplotlib.pyplot as plt



# loading Deep4Chem (descriptors) datasets

print('Loading and preprocessing datasets...')
d4c_train = pd.read_csv('d4c_train_rdkit_descriptors_filtered.csv')
d4c_test = pd.read_csv('d4c_test_rdkit_descriptors_filtered.csv')



# checking for non-finite and excessively large descriptor columns

train_problem_cols = []
test_problem_cols = []

for col in d4c_train.columns:
    if not np.isfinite(d4c_train[col]).all():
        train_problem_cols.append(col)
    elif d4c_train[col].abs().max() > 1e6:
        train_problem_cols.append(col)

for col in d4c_test.columns:
    if not np.isfinite(d4c_test[col]).all():
        test_problem_cols.append(col)
    elif d4c_test[col].abs().max() > 1e6:
        test_problem_cols.append(col)

print('Removing columns:')
print(train_problem_cols)
print(test_problem_cols)

# removing problematic descriptors

d4c_train = d4c_train.drop(columns=train_problem_cols)
d4c_test = d4c_test.drop(columns=test_problem_cols)

# splitting Deep4Chem datasets after cleaning

feature_cols = d4c_train.columns[:-2]
target_cols = ['Absorption max (nm)', 'log(e/mol-1 dm3 cm-1)']

X_train = d4c_train[feature_cols]
y_train = d4c_train[target_cols]

X_test = d4c_test[feature_cols]
y_test = d4c_test[target_cols]



# initialising lists to store evaluation metrics

print('Initialising metric lists and model ...')
lgbm_wavelength_rmse = []
lgbm_wavelength_mae = []
lgbm_log_mec_rmse = []
lgbm_log_mec_mae = []

# initialising lists to store feature importance metrics

wavelength_lgbm_importance = []
log_mec_lgbm_importance = []
wavelength_lgbm_shap = []
log_mec_lgbm_shap = []



# 10-fold cross-validation

kf = KFold(
    n_splits=10,
    shuffle=True,
    random_state=0
)

# iterating over folds

for fold, (train_idx, test_idx) in enumerate(kf.split(X_train)):
    print(f'\nFold {fold}')

    # splitting dataset

    X_fold_train = X_train.iloc[train_idx]
    X_fold_test = X_train.iloc[test_idx]
    y_fold_train = y_train.iloc[train_idx]
    y_fold_test = y_train.iloc[test_idx]



   # initialising LightGBM model

    lgbm = MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=929,
            learning_rate=0.28,
            max_depth=4,
            num_leaves=271,
            colsample_bytree=0.9,
            subsample=0.95,
            random_state=fold,
            n_jobs=-1
        )
    )
    
    
    
    # training
    
    print('Training...')
    lgbm.fit(X_fold_train, y_fold_train)

    # saving feature importance

    wavelength_lgbm_importance.append(lgbm.estimators_[0].feature_importances_)
    log_mec_lgbm_importance.append(lgbm.estimators_[1].feature_importances_)



    # predicting
    
    print('Predicting...')
    pred = lgbm.predict(X_fold_test)

    

    # calculating evaluation metrics

    print('Calculating metrics...')
    
    # calculating RMSE
    
    lgbm_wavelength_rmse.append(
        np.sqrt(
            mean_squared_error(
                y_fold_test.iloc[:,0],
                pred[:,0]
            )
        )
    )
    
    lgbm_log_mec_rmse.append(
        np.sqrt(
            mean_squared_error(
                y_fold_test.iloc[:,1],
                pred[:,1]
            )
        )
    )
    
    # calculating MAE
    
    lgbm_wavelength_mae.append(
        mean_absolute_error(
            y_fold_test.iloc[:,0],
            pred[:,0]
        )
    )
    
    lgbm_log_mec_mae.append(
        mean_absolute_error(
            y_fold_test.iloc[:,1],
            pred[:,1]
        )
    )
    


    # saving predictions
    
    prediction_df = pd.DataFrame({
        'True_wavelength': y_fold_test.iloc[:,0],
        'Pred_wavelength': pred[:,0],
        'True_log_mec': y_fold_test.iloc[:,1],
        'Pred_log_mec': pred[:,1]
    })
    
    prediction_df.to_csv(f'lgbm_fold{fold}_predictions.csv', index=False)


    
    # initialising SHAP analysis

    print('Analysing feature importance...')
    wavelength_explainer = shap.TreeExplainer(lgbm.estimators_[0])
    log_mec_explainer = shap.TreeExplainer(lgbm.estimators_[1])

    # choosing sample data for SHAP analysis

    shap_samples = X_fold_test.sample(n=min(500, len(X_fold_test)), random_state=fold)

    # computing SHAP values for samples

    wavelength_shap = wavelength_explainer.shap_values(shap_samples)
    log_mec_shap = log_mec_explainer.shap_values(shap_samples)

    # storing SHAP values

    wavelength_lgbm_shap.append(np.abs(wavelength_shap).mean(axis=0))
    log_mec_lgbm_shap.append(np.abs(log_mec_shap).mean(axis=0))

    # producing summary plots for the final (representative) fold

    if fold == 9:
        
        plt.figure()
        shap.summary_plot(wavelength_shap, shap_samples, show=False)
        plt.tight_layout()
        plt.savefig('lgbm_wavelength_shap_summary.png', dpi=300)
        plt.close()
    
        plt.figure()
        shap.summary_plot(log_mec_shap, shap_samples, show=False)
        plt.tight_layout()
        plt.savefig('lgbm_log_mec_shap_summary.png', dpi=300)
        plt.close()



#  saving individual and average (mean) LightGBM model metrics

training_summary = pd.DataFrame({
    'Task':[
        'Absorption max (nm)',
        'log(e/mol-1 dm3 cm-1)'
    ],
    
    'Mean rmse':[
        np.mean(lgbm_wavelength_rmse),
        np.mean(lgbm_log_mec_rmse)
    ],

    'Standard deviation rmse':[
        np.std(lgbm_wavelength_rmse),
        np.std(lgbm_log_mec_rmse)
    ],
})

# adding RMSE for each fold
    
for i in range(10):
    training_summary[f'Fold {i} rmse'] = [
        lgbm_wavelength_rmse[i],
        lgbm_log_mec_rmse[i]
    ]

# adding MAE mean and standard deviation

training_summary['Mean mae'] = [
    np.mean(lgbm_wavelength_mae),
    np.mean(lgbm_log_mec_mae)
]

training_summary['Standard deviation mae'] = [
    np.std(lgbm_wavelength_mae),
    np.std(lgbm_log_mec_mae)
]

# adding MAE for each fold

for i in range(10):
    training_summary[f'Fold {i} mae'] = [
        lgbm_wavelength_mae[i],
        lgbm_log_mec_mae[i]
    ]

training_summary.to_csv('lgbm_training_summary.csv', index=False)
print(training_summary)



# averaging LightGBM predictions

all_predictions = []

for model_no in range(10):
    lgbm = MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=929,
            learning_rate=0.28,
            max_depth=4,
            num_leaves=271,
            colsample_bytree=0.9,
            subsample=0.95,
            random_state=model_no,
            n_jobs=-1
        )
    )
    
    lgbm.fit(X_train, y_train)
    pred = lgbm.predict(X_test)
    all_predictions.append(pred)

all_predictions = np.array(all_predictions)
mean_predictions = np.mean(all_predictions, axis=0)



# creating file of average property predictions

print('Generating averaged LightGBM model predictions...')
ensemble_preds = pd.DataFrame({
    'Absorption max (nm)': y_test.iloc[:,0],
    'log(e/mol-1 dm3 cm-1)': y_test.iloc[:,1],
    'Predicted Absorption max (nm)': mean_predictions[:,0],
    'Predicted log(e/mol-1 dm3 cm-1)': mean_predictions[:,1],
})

ensemble_preds.to_csv('lgbm_ensemble_preds.csv', index=False)
print(ensemble_preds)



# initialising LightGBM ensemble evaluation metric lists

print('Calculating ensemble metrics...')
lgbm_ensemble_wavelength_rmse = []
lgbm_ensemble_log_mec_rmse = []
lgbm_ensemble_wavelength_mae = []
lgbm_ensemble_log_mec_mae = []



for model_no in range(10):
    
    pred = all_predictions[model_no]
    
    # calculating ensemble RMSE
    
    lgbm_ensemble_wavelength_rmse.append(
        np.sqrt(
            mean_squared_error(
                y_test.iloc[:,0],
                pred[:,0]
            )
        )    
    )
    
    lgbm_ensemble_log_mec_rmse.append(
        np.sqrt(
            mean_squared_error(
                y_test.iloc[:,1],
                pred[:,1]
            )
        )
    )
    
    # calculating ensemble MAE
    
    lgbm_ensemble_wavelength_mae.append(
        mean_absolute_error(
            y_test.iloc[:,0],
            pred[:,0]
        )
    )
        
    lgbm_ensemble_log_mec_mae.append(
        mean_absolute_error(
            y_test.iloc[:,1],
            pred[:,1]
        )
    )
        


# creating file to summarise LightGBM outputs

print('Generating average LightGBM model metrics summary...')
summary = pd.DataFrame({
    'Metric':[
        'Absorption max (nm) RMSE',
        'log(e/mol-1 dm3 cm-1) RMSE',
        'Absorption max (nm) MAE',
        'log(e/mol-1 dm3 cm-1) MAE'
    ],
    
    'Mean':[
        np.mean(lgbm_ensemble_wavelength_rmse),
        np.mean(lgbm_ensemble_log_mec_rmse),
        np.mean(lgbm_ensemble_wavelength_mae),
        np.mean(lgbm_ensemble_log_mec_mae)
    ],

    'Standard deviation':[
        np.std(lgbm_ensemble_wavelength_rmse),
        np.std(lgbm_ensemble_log_mec_rmse),
        np.std(lgbm_ensemble_wavelength_mae),
        np.std(lgbm_ensemble_log_mec_mae)
    ]
})

summary.to_csv('lgbm_metrics_summary.csv', index=False)
print(summary)



# averaging and calculating standard deviations for LightGBM feature importance

print('Analysing average feature importance for each target...')
wavelength_lgbm_mean = np.mean(wavelength_lgbm_importance, axis=0)
wavelength_lgbm_sd = np.std(wavelength_lgbm_importance, axis=0)
log_mec_lgbm_mean = np.mean(log_mec_lgbm_importance, axis=0)
log_mec_lgbm_sd = np.std(log_mec_lgbm_importance, axis=0)

# averaging and calculating standard deviations for LightGBM SHAP analysis

wavelength_lgbm_shap_mean = np.mean(wavelength_lgbm_shap, axis=0)
wavelength_lgbm_shap_sd = np.std(wavelength_lgbm_shap, axis=0)
log_mec_lgbm_shap_mean = np.mean(log_mec_lgbm_shap, axis=0)
log_mec_lgbm_shap_sd = np.std(log_mec_lgbm_shap, axis=0)



# analysing feature importance for wavelengths 

wavelength_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Mean_LGBM_Importance': wavelength_lgbm_mean,
    'SD_LGBM_Importance': wavelength_lgbm_sd,
    'Mean_SHAP': wavelength_lgbm_shap_mean,
    'SD_SHAP': wavelength_lgbm_shap_sd
})

# sorting by importance and saving ranking for wavelengths

wavelength_importance.sort_values('Mean_SHAP', ascending=False, inplace=True)
wavelength_importance.to_csv('lgbm_wavelength_feature_summary.csv', index=False)



# analysing feature importance for logE

log_mec_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Mean_LGBM_Importance': log_mec_lgbm_mean,
    'SD_LGBM_Importance': log_mec_lgbm_sd,
    'Mean_SHAP': log_mec_lgbm_shap_mean,
    'SD_SHAP': log_mec_lgbm_shap_sd
})

# sorting by importance and saving ranking for logE

log_mec_importance.sort_values('Mean_SHAP', ascending=False, inplace=True)
log_mec_importance.to_csv('lgbm_log_mec_feature_summary.csv', index=False)

print('Finished')