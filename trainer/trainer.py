
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

        self.Cr = Cr
        self.Re = Re
        self.Co = Co
        self._config = config

        self.loss_obj = tf.keras.losses.MeanSquaredError()


        self.train_log = EmptyLogger() if config['no_log'] else Logger(name='train', path=config['logs'])


        print(config['lr'], type(config['lr']))
        exit()

        self.optimizer_cr = tf.keras.optimizers.Adam(learning_rate=config['lr'])
        self.optimizer_re = tf.keras.optimizers.Adam(learning_rate=config['lr'])

        self.ds_train = ClicData().get_train()

        
    def run(self):
        
        epochs = int(self._config['epochs'])

        print('Starting training....')
        for epoch in tqdm(range(epochs), desc='Epochs'):

            for batch_idx, images in enumerate(tqdm(self.ds_train, leave=False)):
                results = self.train_step(images)
                self.train_log.update_scalars(results['metrics'])


            self.train_log.log_images(results['tensors']['images'],
                                      results['tensors']['outputs'],
                                      epoch)
    

            self.train_log.log_scalars(epoch)


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

            out_re_ap = self.Re(out_cr, training=False)

            loss_cr = self.loss_obj(images, out_re_ap)


        gradients_cr = tape_cr.gradient(loss_cr, self.Cr.trainable_variables)
        self.optimizer_cr.apply_gradients(zip(gradients_cr, 
                                      self.Cr.trainable_variables))
        # ------------ End -------------


        #out = self.Re(self.Co(self.Cr(images)))
        ssim = tf.image.ssim(images, out_re, 255)

        return {
                    'metrics' : {'loss_cr':loss_cr, 
                                 'loss_re':loss_re, 
                                 'ssim':ssim},
                    'tensors' : {'images':images,
                                 'outputs':out_re} # should be a special func
                }

        #def val_step(self, images : tf.Tensor):

        


