import os

cmd = (
    f'chemprop_predict '
    f'--test_path d4c_test.csv '
	f'--checkpoint_dir /users/rmb20171/project/results/deep4chem_chemprop/reproduce_d4c_chemprop_full/d4c_training/folds/ '
    f'--number_of_molecules 2 '
    f'--num_workers 16 '
    f'--preds_path d4c_ensemble_preds.csv '
    f'--individual_ensemble_predictions'
    )

print(cmd)

print('Predicting...')

os.system(cmd)

print('Finished')
