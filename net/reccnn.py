
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
        self.conv20 = nn.Conv2D(num_channels, 3, activation='relu', padding='same')

    def call(self, x):

        x = self.conv1(x)
        x = self.convs(x)
        x = self.conv20(x)

        x = tf.image.resize(x, [2*x.shape[1], 2*x.shape[2]], method='bicubic')

        return x