
import tensorflow as tf
import tensorflow.keras.layers as nn

def pad(x, p=1):
  return tf.pad(x, 
          tf.constant([[0,0],[p,p],[p,p],[0,0]]),
          "CONSTANT")

class ResBlock(nn.Layer):

  def __init__(self, num_filters):
    super(ResBlock, self).__init__()
    self.num_filters = num_filters

  def build(self, input_shape):
    params = dict(activation='relu', padding='same')
    self.conv1 = nn.Conv2D(self.num_filters, 3, **params)
    self.conv2 = nn.Conv2D(int(input_shape[-1]), 3, **params)

  def call(self, x):
    x_res = x
    
    x = self.conv1(x)
    x = self.conv2(x)

    return x + x_res

class DepthToSpace(nn.Layer):

  def __init__(self, block_size):
    super(DepthToSpace, self).__init__()
    self.block_size = block_size

  def call(self, x):
    return tf.nn.depth_to_space(x, block_size=self.block_size)


from .model import *