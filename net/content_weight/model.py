
import tensorflow as tf
from .encoder import Encoder
from .decoder import Decoder
from .importance import Binarizer, ImportanceMap, Quantizer, Mask


class Compressor(tf.keras.Model):

  def __init__(self, config : dict):
    super(Compressor, self).__init__()

    n, L = (config[k] for k in ('n', 'L'))

    self.encoder = Encoder(n=n)
    self.decoder = Decoder(n=n)

    self.importance_map = ImportanceMap()
    
    self.quantizer = Quantizer(L=L)
    self.binarizer = Binarizer()
    self.mask = Mask(n=n, L=L)


  def call(self, x : tf.Tensor):

    Ex, Fx = self.encoder(x)

    binary_fmap = self.binarizer(Ex)

    p = self.importance_map(Fx)
    q = self.quantizer(p)
    m = self.mask(q)

    binary_codes = binary_fmap * m

    c = self.decoder(binary_codes)

    return c, p


  def call_pre(self, x : tf.Tensor):

    Ex, _ = self.encoder(x)

    binary_fmap = self.binarizer(Ex)

    c = self.decoder(binary_fmap)

    return c



def rate_loss(Px, r_0):
  L_R_0 = tf.reduce_sum(Px)
  
  # According to section 5.1 threhsold r is r_0 * h * w
  # with r_0 being the desired bpp
  r = r_0 * Px.shape[1] * Px.shape[2]

  # return L_R_0
  return tf.maximum(L_R_0 - r, 0)
