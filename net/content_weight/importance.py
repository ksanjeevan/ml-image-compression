import tensorflow as tf
import tensorflow.keras.layers as nn

class ImportanceMap(nn.Layer):

  def __init__(self):
    super(ImportanceMap, self).__init__()

    # layers = [nn.Conv2D(128, 3, padding='same', activation='relu'),
    #       nn.Conv2D(128, 3, padding='same', activation='relu'),
    #       nn.Conv2D(1, 1, activation='sigmoid')]

    # self.model = tf.keras.Sequential(layers)

    self.conv1 = nn.Conv2D(128, 3, padding='same', activation='relu')
    self.conv2 = nn.Conv2D(128, 3, padding='same', activation='relu')
    self.conv3 = nn.Conv2D(1, 1, activation='sigmoid')

  def call(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    return x


class Binarizer:

  @tf.custom_gradient
  def __call__(self, x):
    def grad(y):
      return tf.maximum(tf.minimum(y, 1.0), 0.0)
    return tf.where(x > 0.5, 1.0, 0.0), grad


class Quantizer:
  """
  Paper Eq. (4)
  """
  def __init__(self, L):
    self.L = L
  
  @tf.custom_gradient
  def __call__(self, p):

    def grad(y):
      return y

    # I hate tensorflow why can't we do boolean mask assignment???
    ret = []
    for l in range(1, self.L+1):
      cond1 = (l - 1) <= p * self.L
      cond2 = l > p * self.L
      x = tf.where(cond1&cond2, l - 1, 0)
      ret.append( x )

    qp = tf.math.add_n(ret)
    return qp, grad


class Mask:
  """
  Paper Eq. (5)
  """
  def __init__(self, n, L):
    self.n = n
    self.L = L

  @tf.custom_gradient
  def __call__(self, Qp):

    # since the derivative of Q is taken as I this is valid (???)
    def grad(p):
      ret = []
      for k in range(1, self.n + 1):
        cond1 = (self.L * p - 1 <= math.ceil(k*self.L/self.n))
        cond2 = (math.ceil(k*self.L/self.n) < self.L * p + 1)
        val = self.L * tf.cast(cond1 & cond2, dtype='int32')
        ret.append( val ) 

      return tf.concat(ret, axis=-1)
    
    ret = []
    for k in range(1, self.n + 1):
      ret.append( k <= (self.n * Qp / self.L) ) 


    # after k > n * Q(p) / L, increasing values of k will
    # mean m_ij = 0 and will be discarded
    m = tf.cast(tf.concat(ret, axis=-1), dtype='float32')
    
    return m, grad



if __name__ == '__main__':

  '''
  Toy example to verify 
  https://github.com/limuhit/ImageCompression/issues/2#issuecomment-478662006

  '''

  import math
  def Q(p, L):
    """
    Eq. (4) in the paper
    """
    for l in range(1,1+L):
      cond = ((l-1)/L <= p)
      cond2 = (p < l/L)
      if cond and cond2:
        return l-1

  def M(p, L, n):
    """
    Eq. (5) in the paper
    """
    ret = []
    for k in range(1,1+n):
      cond1 = (k <= n*Q(p,L)/L)
      ret.append(1 if cond1 else 0)
    return ret


  def M_2(p, L, n):
    """
    Eq. (6) in the paper
    """
    ret = []
    for k in range(1,1+n):
      cond1 = (math.ceil(k*L/n)<L*p)
      ret.append(1 if cond1 else 0)

    return ret


  def M_2_der(p, L, n):
    """
    Eq. (7) in the paper
    """
    ret = []
    for k in range(1,1+n):
      cond1 = (L*p-1 <= math.ceil(k*L/n))
      cond2 = (math.ceil(k*L/n) < L*p+1)
      ret.append(L if cond1 and cond2 else 0)

    return ret

  print(Q(0.5, 8))
  print(M(0.5, 8, 8))
  print(M_2_der(0.5, 8, 8))