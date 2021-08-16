import tensorflow as tf

class ImageCodec:

    def __init__(self, **encode_params):
        self._encode_params = encode_params
    
    def encode(self, images):
        return tf.map_fn(lambda x: tf.io.encode_jpeg(x, **self._encode_params), 
                         images, 
                         fn_output_signature=tf.string)


    def decode(self, bits):
        return tf.map_fn(lambda x: tf.io.decode_jpeg(x), 
                         bits, 
                         fn_output_signature=tf.uint8)

    def __call__(self, images):

        images = tf.cast(tf.round(images), tf.uint8)

        bits = self.encode(images)
        images_jpeg = self.decode(bits)

        #images_jpeg = tf.cast(images_jpeg, tf.float32) / 255.0
        images_jpeg = tf.cast(images_jpeg, tf.float32)
        #return tf.stop_gradient(images_jpeg)
        return images_jpeg