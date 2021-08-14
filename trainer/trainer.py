
import tensorflow as tf

# Follow example on how multiple losses https://www.tensorflow.org/tutorials/generative/dcgan
# https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch

from data import ClicData

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


        self.optimizer_cr = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.optimizer_re = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self.ds_train = ClicData().get_train()


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


    def train_step(self, images : tf.Tensor):
    
        with tf.GradientTape() as tape_cr:

            outputs_cr = self.Cr(images, training=True)

            outputs = self.Re(outputs_cr, training=False)

            loss_cr = self.loss_obj(images, outputs)


        gradients_cr = tape_cr.gradient(loss_cr, self.Cr.trainable_variables)
        self.optimizer_cr.apply_gradients(zip(gradients_cr, 
                                      self.Cr.trainable_variables))


        with tf.GradientTape() as tape_re:

            decomp = self.Co(outputs_cr)
            outputs = self.Re(decomp, training=True)
            loss_re = self.loss_obj(images, outputs)


        gradients_re = tape_re.gradient(loss_re, self.Re.trainable_variables)
        self.optimizer_re.apply_gradients(zip(gradients_re, 
                                       self.Re.trainable_variables))


        self.train_loss_cr(loss_cr)
        self.train_loss_re(loss_re)


        return float(loss_cr), float(loss_re)
