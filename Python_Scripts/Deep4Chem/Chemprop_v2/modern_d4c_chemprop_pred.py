import os

cmd = (
    f'chemprop predict '
    f'--test-path d4c_test.csv '
	f'--model-paths '
    f'/users/rmb20171/project/results/chemprop_modern/modern_d4c_training/replicate_0/model_0/best.pt '
    f'/users/rmb20171/project/results/chemprop_modern/modern_d4c_training/replicate_1/model_0/best.pt '
    f'/users/rmb20171/project/results/chemprop_modern/modern_d4c_training/replicate_2/model_0/best.pt '
    f'/users/rmb20171/project/results/chemprop_modern/modern_d4c_training/replicate_3/model_0/best.pt '
    f'/users/rmb20171/project/results/chemprop_modern/modern_d4c_training/replicate_4/model_0/best.pt '
    f'/users/rmb20171/project/results/chemprop_modern/modern_d4c_training/replicate_5/model_0/best.pt '
    f'/users/rmb20171/project/results/chemprop_modern/modern_d4c_training/replicate_6/model_0/best.pt '
    f'/users/rmb20171/project/results/chemprop_modern/modern_d4c_training/replicate_7/model_0/best.pt '
    f'/users/rmb20171/project/results/chemprop_modern/modern_d4c_training/replicate_8/model_0/best.pt '
    f'/users/rmb20171/project/results/chemprop_modern/modern_d4c_training/replicate_9/model_0/best.pt '
    f'--smiles-columns Chromophore Solvent '
    f'--num-workers 16 '
    f'--preds-path ensemble_preds.csv'
    )

print(cmd)

print('Predicting...')

os.system(cmd)

print('Finished')
