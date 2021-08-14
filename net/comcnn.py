
#from net import

import tensorflow as tf
import tensorflow.keras.layers as nn

class ComCNN(tf.keras.Model):

    def __init__(self, num_channels=3):
        super(ComCNN, self).__init__()
        tf.keras.Sequential()
        self.conv1 = nn.Conv2D(64, 3, activation='relu', padding='same')
        self.conv2 = nn.Conv2D(64, 3, strides=2, activation='relu', padding='same')
        self.bn = nn.BatchNormalization(axis=-1)
        self.conv3 = nn.Conv2D(num_channels, 3, activation='relu', padding='same')

    def call(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.conv3(x)

        return x