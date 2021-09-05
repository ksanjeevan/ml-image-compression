from pathlib import Path
import tensorflow as tf

# Follow example on how multiple losses https://www.tensorflow.org/tutorials/generative/dcgan
# https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch

from data import ClicData
from net import load_model

from tqdm import tqdm
from .logger import Logger, EmptyLogger

from net import ComCNN, RecCNN, ImageCodec

MAX_VAL = 1.0

class Trainer:

  def __init__(self, config : dict):

      self.Cr = ComCNN(num_channels=3)
      self.Re = RecCNN(num_channels=3)
      self.Co = ImageCodec()

      self._config = config

      self.log = EmptyLogger() if config['no_log'] else Logger(config['logs'])
      self.log.log_config(config)

      # self.loss_obj = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
      self.loss_obj = tf.keras.losses.MeanSquaredError()
      self.optimizer_re = tf.keras.optimizers.Adam(learning_rate=config['lr'])
      self.optimizer_cr = tf.keras.optimizers.Adam(learning_rate=config['lr'])

      dataset = ClicData(config)
      self.ds_train = dataset.get_train()
      self.ds_val = dataset.get_val()

      if config['resume_path'] is not None:
          self._resume_weights()

  def _resume_weights(self):

    input_shape = self.ds_train.element_spec.shape

    self.Cr = load_model('cr', self._config['resume_path'], input_shape)
    self.Re = load_model('re', self._config['resume_path'], input_shape)

    print('[RESUME TRAINING]')


  def run(self):
      
    epochs = int(self._config['epochs'])

    print('Starting training....')
    best_ssim = float('-inf')

    for epoch in tqdm(range(1,epochs+1), desc='Epochs'):

      _train_iter = tqdm(self.ds_train, leave=False, desc='train')
      for batch_idx, images in enumerate(_train_iter):
          results = self.train_step(images)
          self.log.update_scalars(results['metrics'])
  
      self.log.log_hist(results['tensors']['weights'], epoch)

      _val_iter = tqdm(self.ds_val, leave=False, desc='val')
      for batch_idx, images in enumerate(_val_iter):
          val_results = self.val_step(images)
          self.log.update_scalars(val_results['metrics'])

      val_ssim = self.log.scalars['ssim_val'].result().numpy()
      if val_ssim > best_ssim:
          best_ssim = val_ssim
          self.log.log_model('cr', self.Cr)
          self.log.log_model('re', self.Re)

      self.log.log_scalars(epoch)
      self.log.log_images(self.Co(val_results['tensors']['images']),
                          val_results['tensors']['outputs'],
                          epoch)

  def train_step(self, images : tf.Tensor):

    # ------------ Update Re -------------
    with tf.GradientTape() as tape_re:

      out_cr = self.Cr(images, training=False)

      out_codec = self.Co(out_cr)        
      out_re = self.Re(out_codec, training=True)

      loss_re = self.loss_obj(images, out_re)
      
      # Paper loss !?
      #x_hat = self.Re.compact_upscaled(out_codec)
      #residual = self.Re.residual(x_hat, training=True)
      #loss_re = self.loss_obj(residual, images - x_hat)


    
    gradients_re = tape_re.gradient(loss_re, self.Re.trainable_variables)
    self.optimizer_re.apply_gradients(zip(gradients_re, 
                                   self.Re.trainable_variables))
    # ------------ End -------------



    # ------------ Update Cr -------------
    with tf.GradientTape() as tape_cr:

      out_cr = self.Cr(images, training=True)

      out_re_aprox = self.Re(out_cr, training=False)

      loss_cr = self.loss_obj(images, out_re_aprox)

    gradients_cr = tape_cr.gradient(loss_cr, self.Cr.trainable_variables)
    self.optimizer_cr.apply_gradients(zip(gradients_cr, 
                                  self.Cr.trainable_variables))
    # ------------ End -------------


    out = self.Re(self.Co(self.Cr(images)))
    out = tf.minimum(tf.maximum(out, 0), MAX_VAL)
    ssim = tf.image.ssim(images, out, max_val=MAX_VAL)

    # Some hist debugging
    hist_weights1 = {l.name:l for l in self.Cr.bn.trainable_variables}
    hist_weights2 = {l.name:l for l in self.Re.conv1.trainable_variables}
    hist_weights3 = {l.name:l for l in self.Re.convs.layers[1].trainable_variables}

    hist_weights = {**hist_weights1, **hist_weights2, **hist_weights3}

    return {
                'metrics' : {
                                 'loss_cr' : loss_cr, 
                                 'loss_re' : loss_re, 
                                 'ssim'    : ssim,
                                 'grad_re' : tf.linalg.global_norm(gradients_re),
                                 'grad_cr' : tf.linalg.global_norm(gradients_cr)
                             },

                'tensors' : {   
                                'images'  : images,
                                'outputs' : out,
                                'weights' : hist_weights
                             } # should be a special func
            }

  def val_step(self, images : tf.Tensor):

    out_cr = self.Cr(images, training=False)
    out_codec = self.Co(out_cr)        
    out = self.Re(out_codec, training=False)

    loss_re = self.loss_obj(images, out)

    out_re_aprox = self.Re(out_cr, training=False)
    loss_cr = self.loss_obj(images, out_re_aprox)

    out = tf.minimum(tf.maximum(out, 0), MAX_VAL)
    ssim = tf.image.ssim(images, out, max_val=MAX_VAL)

    return {
                'metrics' : {'loss_cr_val':loss_cr, 
                             'loss_re_val':loss_re, 
                             'ssim_val':ssim},
                'tensors' : {'images':images,
                             'outputs':out} # should be a special func
            }
        
          
          


