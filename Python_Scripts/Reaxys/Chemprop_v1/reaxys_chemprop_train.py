import os

cmd = (
    f'chemprop_train '
    f'--data_path reaxys_data.csv '
    f'--dataset_type regression '
    f'--epochs 100 '
    f'--number_of_molecules 2 '
    f'--split_type random '
    f'--seed 0 '
    f'--depth 5 '
    f'--hidden_size 1900 '
    f'--ffn_hidden_size 2300 '
    f'--ffn_num_layers 3 '
    f'--dropout 0.15 '
    f'--aggregation mean '
    f'--batch_size 50 '
    f'--metric rmse '
    f'--num_workers 16 '
    f'--save_dir reaxys_train'
    )

print(cmd)

print('Training...')

os.system(cmd)

print('Finished')
