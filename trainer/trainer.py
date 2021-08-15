
import tensorflow as tf

# Follow example on how multiple losses https://www.tensorflow.org/tutorials/generative/dcgan
# https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch

from data import ClicData

from trainer import setup_logging

from tqdm import tqdm

class Trainer:

    def __init__(self, Cr : tf.keras.Model, 
                       Re : tf.keras.Model, 
                       Co : tf.keras.Model):

        self.Cr = Cr
        self.Re = Re
        self.Co = Co

        self.loss_obj = tf.keras.losses.MeanSquaredError()

        
        self.train_loss_cr = tf.keras.metrics.Mean()
        self.train_loss_re = tf.keras.metrics.Mean()
        self.train_ssim = tf.keras.metrics.Mean()


        self.optimizer_cr = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.optimizer_re = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self.ds_train = ClicData().get_train()

        log_path = setup_logging('logs')
        self.tb_train_writer = tf.summary.create_file_writer(str(log_path.joinpath('train')))

        #tf.keras.callbacks.TensorBoard('logs').set_model(self.Re)


    def train(self):
        
        epochs = 10

        print('Starting training....')
        for epoch in range(epochs):

            print(f'Epoch {epoch}')

            for batch_idx, images in enumerate(tqdm(self.ds_train, disable=True)):
                loss_cr, loss_re, = self.train_step(images)

                if batch_idx % 5 == 0:
                    print('\t Loss Cr: %.2f, Loss Re %.2f'%(loss_cr, loss_re)) 

            print(f'Loss Cr {self.train_loss_cr.result()}')
            print(f'Loss Re {self.train_loss_re.result()}')
            print('--------------------------------------')

            with self.tb_train_writer.as_default():
                tf.summary.scalar('loss_cr', self.train_loss_cr.result(), step=epoch)
                tf.summary.scalar('loss_re', self.train_loss_re.result(), step=epoch)
                tf.summary.scalar('ssim', self.train_ssim.result(), step=epoch)
                
                #ims = images.numpy().round().astype('uint8')
                #tf.summary.image('sample', ims, max_outputs=16, step=epoch)

            self.train_loss_cr.reset_states()
            self.train_loss_re.reset_states()
            self.train_ssim.reset_states()


    def train_step(self, images : tf.Tensor):

        # ------------ Update Re -------------
        with tf.GradientTape() as tape_re:

            out_cr = self.Cr(images, training=False)
            out_codec = self.Co(out_cr)
            out_re = self.Re(out_codec, training=True)
            loss_re = self.loss_obj(images, out_re)


        #print(out_re)
        #print(tf.reduce_max(out_re))
        #print(tf.reduce_min(out_re))
        #exit()

        ssim = tf.image.ssim(images, out_re, 255)

        gradients_re = tape_re.gradient(loss_re, self.Re.trainable_variables)
        self.optimizer_re.apply_gradients(zip(gradients_re, 
                                       self.Re.trainable_variables))
        # ------------ End -------------

    
        with tf.GradientTape() as tape_cr:

            out_cr = self.Cr(images, training=True)

            out_re_ap = self.Re(out_cr, training=False)

            loss_cr = self.loss_obj(images, out_re_ap)


        gradients_cr = tape_cr.gradient(loss_cr, self.Cr.trainable_variables)
        self.optimizer_cr.apply_gradients(zip(gradients_cr, 
                                      self.Cr.trainable_variables))


        tf.image

        self.train_loss_cr(loss_cr)
        self.train_loss_re(loss_re)
        self.train_ssim(ssim)

        return float(loss_cr), float(loss_re)
