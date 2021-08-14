import tensorflow as tf

class ImageCodec:

    def __init__(self):
        pass
    
    def encode(self, images):
        return tf.map_fn(lambda x: tf.io.encode_jpeg(x), 
                         tf.cast(tf.round(images), tf.uint8), 
                         fn_output_signature=tf.string)


    def decode(self, bits):
        return tf.map_fn(lambda x: tf.io.decode_jpeg(x), 
                         bits, 
                         fn_output_signature=tf.uint8)

    def __call__(self, images):

        bits = self.encode(images)
        images_jpeg = self.decode(bits)
        return tf.stop_gradient(images_jpeg)