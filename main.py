import os
import sys
import subprocess
import re

DATASETS = ['ETTh1']

HYPER_PARAMS = {
    'model': 'GTC',
    'seq_len': 96,
    'pred_len': 96,
    'enc_in': 7,
    'e_layers': 8,
    'a_layers': 0,
    'b_layers': 0,
    'batch_size': 256,
    'learning_rate': 0.005,
    'train_epochs': 30,
    'patience': 5,
    'miss_rate': 0.3,
    'features': 'M',
    'use_hour_index': 1,
    'hour_length': 24,
    'period_len': 24,
    'rec_lambda': 0.,
    'auxi_lambda': 1,
    'itr': 1,
    'random_seed': 2024,
    'miss_type': 'MAR',
}

PATH_CONFIG = {
    'root_path': './dataset/',
    'data_cache': 'data/',
    'ckpoint': './ckpoint/',
    'is_training': 1
}

def get_base_args():
    args = []
    for k, v in PATH_CONFIG.items():
        args.extend([f'--{k}', str(v)])
    for k, v in HYPER_PARAMS.items():
        args.extend([f'--{k}', str(v)])
    return args

def run_task(data_name):
    model_id = f'{data_name}_{HYPER_PARAMS["seq_len"]}_{HYPER_PARAMS["pred_len"]}'

    cmd = [sys.executable, 'run.py'] + get_base_args()
    cmd.extend(['--data', data_name])
    cmd.extend(['--data_path', f'{data_name}.csv'])
    cmd.extend(['--model_id', model_id])

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8'
        )

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line, end='')

        if process.wait() != 0:
            return None

    except Exception:
        return None

if __name__ == '__main__':
    if not os.path.exists('run.py'):
        sys.exit(1)

    for data in DATASETS:
        run_task(data)