import os

cmd = (
    f'chemprop hpopt '
    f'--data-path reaxys_train_hpopt.csv '
	f'--task-type regression '
    f'--smiles-columns Chromophore Solvent '
	f'--target-columns "Absorption max (nm)" "log(e/mol-1 dm3 cm-1)" '
    f'--search-parameter-keywords basic '
    f'--hpopt-save-dir modern_reaxys_hpopt '
	f'--raytune-num-cpus 16'
    )

print(cmd)

print('Optimising...')

os.system(cmd)

print('Finished')
