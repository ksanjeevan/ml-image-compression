
#https://www.tensorflow.org/datasets/api_docs/python/tfds/load#args

import tensorflow as tf
import tensorflow_datasets as tfds

'''
@tf.function
def scale_and_resize(image : tf.Tensor):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (180, 180))
    return image
'''

class ClicData:

    def __init__(self):
        splits = ['train', 'test', 'validation']
        ds, self.ds_info = tfds.load('clic', 
                                     split=splits, 
                                     download=True, 
                                     shuffle_files=False,
                                     with_info=True)

        self._ds = dict(zip(splits, ds))


    def get_train(self):
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset
        
        ds_train = self._ds['train']

        ds_train = ds_train.map(lambda x: tf.image.resize(x['image'], (180, 180)), 
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds_train = ds_train.cache() # random transforms after cache
        ds_train = ds_train.batch(16)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        return ds_train



'''

print(ds_train)
print(type(ds_train))
exit()

#dataset = train.batch(64).prefetch(10).take(5)

for im in ds_train:
    #print(im['image'].shape)
    print(im.shape)
    break
'''