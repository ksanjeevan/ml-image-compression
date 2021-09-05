

from pathlib import Path
import tensorflow as tf

from data import ClicData

from tqdm import tqdm
from .logger import Logger, EmptyLogger

from net import Compressor, rate_loss


class Trainer:
  MAX_VAL = 1.0

  def __init__(self, config : dict):

      self.model = Compressor(config)
      self._config = config

      self.gamma = config['gamma']
      self.rate_0 = config['rate']

      self.log = EmptyLogger() if config['no_log'] else Logger(config['logs'])
      self.log.log_config(config)

      self.L_D = tf.keras.losses.MeanSquaredError()
      self.L_R = rate_loss

      self.optimizer = tf.keras.optimizers.Adam(learning_rate=config['lr'])

      dataset = ClicData(config)
      self.ds_train = dataset.get_train()
      self.ds_val = dataset.get_val()

      
  def run(self):
      
    epochs = int(self._config['epochs'])

    print('Starting training....')
    best_ssim = float('-inf')

    for epoch in tqdm(range(1,epochs+1), desc='Epochs'):

      _train_iter = tqdm(self.ds_train, leave=False, desc='train')
      for batch_idx, images in enumerate(_train_iter):
          results = self.train_step(images)
          self.log.update_scalars(results['metrics'])
  
      # self.log.log_hist(results['tensors']['weights'], epoch)

      # _val_iter = tqdm(self.ds_val, leave=False, desc='val')
      # for batch_idx, images in enumerate(_val_iter):
      #     val_results = self.val_step(images)
      #     self.log.update_scalars(val_results['metrics'])

      # val_ssim = self.log.scalars['ssim_val'].result().numpy()
      # if val_ssim > best_ssim:
      #     best_ssim = val_ssim
      #     # self.log.log_model('cr', self.Cr)

      self.log.log_scalars(epoch)
      # self.log.log_images(val_results['tensors']['images'],
      #                     val_results['tensors']['outputs'],
      #                     epoch)



  def train_step(self, images : tf.Tensor):

    with tf.GradientTape() as tape:

      c = self.model.call_pre(images)
      # c, Px = self.model(images)

      ld = self.L_D(images, c)
      lr = 0
      # lr = self.L_R(Px, self.rate_0)
      
      # print(tf.reduce_mean(Px).numpy(), lr.numpy(), self.gamma * lr.numpy())

      loss = ld + self.gamma * lr

    gradients = tape.gradient(loss, self.model.trainable_variables)

    # for i, m in enumerate(self.model.trainable_variables):
    #   if gradients[i] is not None:
    #     g = gradients[i].shape
    #     space = '' 
    #   else:
    #     g = "NONE"
    #     space = "\n"
    #   print(space, m.name, m.shape, g, space)

    self.optimizer.apply_gradients(zip(gradients, 
                                   self.model.trainable_variables))

    c = tf.minimum(tf.maximum(c, 0.0), Trainer.MAX_VAL)
    ssim = tf.image.ssim(images, c, max_val=Trainer.MAX_VAL)

    # Some hist debugging
    # hist_weights1 = {l.name:l for l in self.Cr.bn.trainable_variables}

    return {
                'metrics' : {
                                 'loss'    : loss,
                                 'L_D'     : ld,
                                 'L_R'     : lr,
                                 'ssim'    : ssim,
                                 'grad'    : tf.linalg.global_norm(gradients),
                             },

                'tensors' : {   
                                'images'  : images,
                                'outputs' : c,
                             } 
            }


  # def val_step(self, images : tf.Tensor):

  #   # out = tf.minimum(tf.maximum(out, 0), MAX_VAL)
  #   # ssim = tf.image.ssim(images, out, max_val=MAX_VAL)

  #   return {
  #               'metrics' : {'loss_val':loss_cr, 
  #                            'ssim_val':ssim},

  #               'tensors' : {'images':images,
  #                            'outputs':out}
  #           }
        
          
          


