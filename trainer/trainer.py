from pathlib import Path
import tensorflow as tf

# Follow example on how multiple losses https://www.tensorflow.org/tutorials/generative/dcgan
# https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch

from data import ClicData

from tqdm import tqdm
from .logger import Logger, EmptyLogger


class Trainer:

    def __init__(self, Cr : tf.keras.Model, 
                       Re : tf.keras.Model, 
                       Co : tf.keras.Model,
                       config : dict):

        self.Cr, self.Re, self.Co = Cr, Re, Co
        self._config = config

        self.log = EmptyLogger() if config['no_log'] else Logger(config['logs'])
        
        self.loss_obj = tf.keras.losses.MeanSquaredError()
        self.optimizer_re = tf.keras.optimizers.Adam(learning_rate=config['lr_re'])
        self.optimizer_cr = tf.keras.optimizers.Adam(learning_rate=config['lr_cr'])

        self.ds_train = ClicData().get_train()
        self.ds_val = ClicData().get_val()

        if config['resume_path'] is not None:
            self._resume_train_prep()

    def _resume_train_prep(self):
        self.Cr(tf.zeros(self.ds_train.element_spec.shape))
        self.Re(tf.zeros(self.ds_train.element_spec.shape))

        cr_path = Path(self._config['resume_path']).joinpath('cr/model')
        re_path = Path(self._config['resume_path']).joinpath('re/model')
        self.Cr.load_weights(cr_path)
        self.Re.load_weights(re_path)

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

        out = tf.minimum(tf.maximum(out, 0), 255.0)
        ssim = tf.image.ssim(images, out, max_val=255)

        return {
                    'metrics' : {'loss_cr':loss_cr, 
                                 'loss_re':loss_re, 
                                 'ssim':ssim},
                    'tensors' : {'images':images,
                                 'outputs':out} # should be a special func
                }

    def val_step(self, images : tf.Tensor):

        out_cr = self.Cr(images, training=False)
        out_codec = self.Co(out_cr)        
        out = self.Re(out_codec, training=True)

        loss_re = self.loss_obj(images, out)

        out_re_aprox = self.Re(out_cr, training=False)
        loss_cr = self.loss_obj(images, out_re_aprox)

        out = tf.minimum(tf.maximum(out, 0), 255.0)
        ssim = tf.image.ssim(images, out, max_val=255)

        return {
                    'metrics' : {'loss_cr_val':loss_cr, 
                                 'loss_re_val':loss_re, 
                                 'ssim_val':ssim},
                    'tensors' : {'images':images,
                                 'outputs':out} # should be a special func
                }
            
            
            


