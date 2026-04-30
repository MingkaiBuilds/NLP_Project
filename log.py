import os
from pathlib import Path
import datetime
import json

def log(model, mode, runtime_type):
    '''
    Logs the runtime of a model in a specific mode to a file in the results directory.
    The log file is named according to the model and the number of existing log files for that model.'''
    if model not in ['gpt', 'llama', 'mistral']:
        raise ValueError("Model must be one of 'gpt', 'llama', or 'mistral'")
    if mode not in ['empty prompt', 'common crawl']:
        raise ValueError("Mode must be either 'empty prompt' or 'common crawl'")
    
    # Count number of files in directory
    N = sum(1 for _ in Path('results/' + model).glob('*.log')) + 1
    log_name = f'{model}_{N}.txt'
    log_dir = 'results/' + model + '/' + log_name
    log_path = os.path.join(log_dir, log_name)

    data = {
        'model': model,
        'N': N,
        'mode': mode,
        'runtime_type': runtime_type,
        'timestamp': datetime.datetime.now().isoformat()}
    with open(log_path, 'w') as f:
        json.dump(data, f)