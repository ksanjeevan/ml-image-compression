
import tensorflow as tf
import tensorflow.keras.layers as nn

class RecCNN(tf.keras.Model):

    def __init__(self, num_channels=3):
        super(RecCNN, self).__init__()

        self.conv1 = nn.Conv2D(64, 3, activation='relu', padding='same')
        convs = []
        for _ in range(18):
            convs.append( nn.Conv2D(64, 3, activation='relu', padding='same') )
            convs.append( nn.BatchNormalization() )

        self.convs = tf.keras.Sequential(convs)
        self.conv20 = nn.Conv2D(num_channels, 3, activation=None, padding='same')

    def residual(self, x, training=False):

        x = self.conv1(x)
        x = self.convs(x, training)
        x = self.conv20(x)
        return x

    def compact_upscaled(self, x):
        # return tf.image.resize(x, 
        #                        size=[2*x.shape[1], 
        #                              2*x.shape[2]], 
        #                        method='bicubic')

        return tf.image.resize(x, 
                               size=[180, 180], 
                               method='bicubic')




    def call(self, x, training=False):
        x_up = self.compact_upscaled(x)
        x = self.residual(x_up, training) + x_up
        #x = tf.minimum(tf.maximum(x, 0), 255.0)
        #x = 255.0 * tf.math.sigmoid(x)
        return x