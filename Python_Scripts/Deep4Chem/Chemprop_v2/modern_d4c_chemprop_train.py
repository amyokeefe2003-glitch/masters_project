import os

cmd = (
    f'chemprop train '
	f'--config-path /users/rmb20171/project/results/chemprop_v2/modern_d4c_hpopt/best_config.toml '
    f'--data-path d4c_train.csv '
    f'--smiles-column Chromophore Solvent '
    f'--target-columns "Absorption max (nm)" "log(e/mol-1 dm3 cm-1)" '
    f'--task-type regression '
    f'--num-replicates 10 '
    f'--data-seed 0 '
    f'--epochs 200 '
    f'--metric rmse '
    f'--num-workers 16 '
    f'--output-dir modern_d4c_training'
    )

print(cmd)

print('Training...')

os.system(cmd)

print('Finished')
