from pathlib import Path
import numpy as np
import tensorflow as tf


# PSNR?


def setup_logging(logging_path):

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


class Logger:

    def __init__(self, name, path='logs'):

        log_path = setup_logging(path)

        #tf.keras.callbacks.TensorBoard('logs').set_model(self.Re)
        self.writer = tf.summary.create_file_writer(str(log_path.joinpath('train')))

        self.scalars = {
                            'loss_re' : tf.keras.metrics.Mean(),
                            'loss_cr' : tf.keras.metrics.Mean(),
                            'ssim' : tf.keras.metrics.Mean()
                        }

    
    def log_scalars(self, epoch : int):
        
        with self.writer.as_default():
            for n, m in self.scalars.items():
                tf.summary.scalar(n, m.result(), step=epoch)

        for metric in self.scalars.values():
            metric.reset_states()

    def update_scalars(self, updates : dict):
        for k, u in updates.items(): self.scalars[k](u)


    def log_images(self, images : tf.Tensor, outputs : tf.Tensor, epoch : int, num:int=4):
        ims = images.numpy()
        outs = outputs.numpy()

        if outs.max() <= 1: outs *= 255
        if ims.max() <= 1: ims *= 255

        outs = np.maximum(np.minimum(outs, 255), 0).round().astype('uint8')
        ims = ims.round().astype('uint8')

        for i in range(4):
            np.save('debug/im_%d.npy'%i, ims[i])
            np.save('debug/out_%d.npy'%i, outs[i])

        display = np.concatenate([ims, outs], axis=2)

        with self.writer.as_default():
            tf.summary.image('results', display, step=epoch, max_outputs=num)


class EmptyLogger(Logger):

    def __init__(self, name=None, path=None):
        pass

    def log_scalars(self, epoch : int):
        pass
    
    def update_scalars(self, updates : dict):
        pass

    def log_images(self, images : tf.Tensor, outputs : tf.Tensor, epoch : int, num:int=4):
        pass