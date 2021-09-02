import tensorflow as tf

class ImageCodec:

  def __init__(self, **encode_params):
    self._encode_params = encode_params
  
  def encode(self, images):
    images = tf.cast(tf.round(images * 255.0), tf.uint8)
    return tf.map_fn(lambda x: tf.io.encode_jpeg(x, **self._encode_params), 
             images, 
             fn_output_signature=tf.string)


  def decode(self, bits):
    image_jpeg = tf.map_fn(lambda x: tf.io.decode_jpeg(x), 
                 bits, 
                 fn_output_signature=tf.uint8)
    return tf.cast(image_jpeg, tf.float32) / 255.0

  def __call__(self, images):

    bits = self.encode(images)
    images_jpeg = self.decode(bits)

    return images_jpeg


  def with_sizes(self, images):
    
    bits = self.encode(images)
      
    sizes = tf.map_fn(lambda x: len(x.numpy()), bits, fn_output_signature=tf.int16)

    images_jpeg = self.decode(bits)

    return images_jpeg, sizes
