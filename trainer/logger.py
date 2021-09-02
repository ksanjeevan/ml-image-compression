from pathlib import Path
import json
import numpy as np
import tensorflow as tf

# --samples_per_plugin images=100

# PSNR?

def setup_logging(logging_path):

  log_path = Path(logging_path)
  
  if not log_path.exists(): log_path.mkdir()

  get_ind = lambda x: int(check_names(x.name.repalce('run_', '')))
  run_counter = max([get_ind(p) for p in log_path.glob('run_*') if p.is_dir()], default=-1) + 1

  run_path = log_path.joinpath('run_%s'%run_counter)
  
  run_path.mkdir()

  print(f'Logging set up, to monitor training run:\n'
    f'\t\'tensorboard --logdir={run_path}\'\n')

  return run_path


class Logger:

  def __init__(self, path='logs'):

    self.log_path = setup_logging(path)
    #tf.keras.callbacks.TensorBoard('logs').set_model(self.Re)
    self.writer = tf.summary.create_file_writer(str(self.log_path))

    self.scalars = {}

  
  def log_scalars(self, epoch : int):
    
    with self.writer.as_default():
      for n, m in self.scalars.items():
        tf.summary.scalar(n, m.result(), step=epoch)

    for metric in self.scalars.values():
      metric.reset_states()

  def update_scalars(self, updates : dict):
    for k, u in updates.items():
      if k not in self.scalars:
        self.scalars[k] = tf.keras.metrics.Mean() 
      self.scalars[k](u)


  def log_images(self, images : tf.Tensor, outputs : tf.Tensor,epoch : int, num:int=4):
    ims = images.numpy()
    outs = outputs.numpy()

    if outs.max() > 1: outs /= 255
    if ims.max() > 1: ims /= 255

    #outs = np.maximum(np.minimum(outs, 255), 0).round().astype('uint8')
    #ims = ims.round().astype('uint8')

    #for i in range(4):
    #    np.save('debug/im_%d.npy'%i, ims[i])
    #    np.save('debug/out_%d.npy'%i, outs[i])

    display = np.concatenate([ims, outs], axis=2)

    with self.writer.as_default():
      tf.summary.image('results', display, step=epoch, max_outputs=num)


  # https://www.tensorflow.org/guide/keras/save_and_serialize
  def log_model(self, name : str, model : tf.keras.Model):
    model.save_weights(self.log_path.joinpath(name, 'model'))


  def log_hist(self, data : dict, epoch : int):

    with self.writer.as_default():
      for name, weights in data.items():
        tf.summary.histogram(f'hist_{name}', 
                   weights, 
                   step=epoch, 
                   buckets=None)


  def log_config(self, config : dict):
    with open(self.log_path.joinpath('config.json'), 'w') as wj:
      json.dump(config, wj, indent=4)


class EmptyLogger(Logger):

  def __init__(self, name=None, path=None):
    self.scalars = {}

  def log_scalars(self, epoch : int):
    pass
  
  def log_images(self, images : tf.Tensor, outputs : tf.Tensor, epoch : int, num:int=4):
    pass

  def log_model(self, name : str, model : tf.keras.Model):
    pass
  
  def log_config(self, config : dict):
    pass

  def log_hist(self, data : dict, epoch : int):
    pass
