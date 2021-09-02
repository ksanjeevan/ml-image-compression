
import tensorflow as tf
import tensorflow.keras.layers as nn

from net.content_weight import pad, ResBlock, DepthToSpace

class Decoder(nn.Layer):

  def __init__(self, n=128):
    super(Decoder, self).__init__()
    self.conv1 = nn.Conv2D(512, 1, activation='relu')

    self.res1 = ResBlock(512)
    self.res2 = ResBlock(512)

    self.d2s1 = DepthToSpace(2)
    
    self.conv2 = nn.Conv2D(256, 3, padding='same', activation='relu')
    self.res3 = ResBlock(256)

    self.d2s2 = DepthToSpace(4)

    self.conv3 = nn.Conv2D(32, 3, padding='same', activation='relu')
    self.conv4 = nn.Conv2D(3, 3, padding='same', activation=None)


  def call(self, x):
    x = self.conv1(x)
    x = self.res1(x)
    x = self.res2(x)

    x = self.d2s1(x)
    x = self.conv2(x)

    x = self.res3(x)
    x = self.d2s2(x)

    x = self.conv3(x)
    x = self.conv4(x)
    return x


