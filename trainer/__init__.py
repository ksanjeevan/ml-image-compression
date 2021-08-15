
from pathlib import Path

def setup_logging(logging_path='logs'):

    log_path = Path(logging_path)
    
    if not log_path.exists(): log_path.mkdir()

    check_names = lambda y: y if y.isdigit() else -1
    get_ind = lambda x: int(check_names(x.name.split('_')[1]))
    
    run_counter = max([get_ind(p) for p in log_path.glob('*/') if p.is_dir()], default=-1) + 1

    run_path = log_path.joinpath('run_%s'%run_counter)
    
    run_path.mkdir()

    print(f'Logging set up, to monitor training run:\n'
        f'\t\'tensorboard --logdir={run_path}\'\n')

    return run_path



from .trainer import Trainer