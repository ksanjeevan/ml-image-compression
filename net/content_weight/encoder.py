
import tensorflow as tf
import tensorflow.keras.layers as nn

from net.content_weight import pad, ResBlock


class Encoder(nn.Layer):

  def __init__(self, n=128):
    super(Encoder, self).__init__()
    self.conv1 = nn.Conv2D(128, 8, strides=4, activation='relu')
    self.res1 = ResBlock(128)

    self.conv2 = nn.Conv2D(256, 4, strides=2, activation='relu')
    self.res2 = ResBlock(256)
    self.res3 = ResBlock(256)
    
    self.conv3 = nn.Conv2D(n, 1, activation='sigmoid')


  def call(self, x):
    x = self.conv1(pad(x, 2))
    x = self.res1(x)

    x = self.conv2(pad(x, 1))
    x = self.res2(x)
    Fx = self.res3(x) # here is where F(x) is

    Ex = self.conv3(Fx)

    return Ex, Fx



