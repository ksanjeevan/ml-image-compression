
#https://www.tensorflow.org/datasets/api_docs/python/tfds/load#args

import tensorflow as tf
import tensorflow_datasets as tfds

# https://knowyourdata-tfds.withgoogle.com/#dataset=clic&tab=STATS
# https://www.tensorflow.org/guide/data_performance#prefetching

class ClicData:
    # https://www.tensorflow.org/datasets/catalog/clic
    def __init__(self):
        splits = ['train', 'test', 'validation']
        ds, self.ds_info = tfds.load('clic', 
                                     split=splits, 
                                     download=True, 
                                     shuffle_files=False,
                                     with_info=True)

        self._ds = dict(zip(splits, ds))


    def _common_pipeline(self, ds):
        ds = ds.map(lambda x: tf.image.resize(x['image'], (180, 180)), 
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

        #ds = ds.map(lambda x: tf.image.convert_image_dtype.resize(x, dtype=tf.float32), 
        #                num_parallel_calls=tf.data.experimental.AUTOTUNE)


        ds = ds.cache() # random transforms after cache
        ds = ds.batch(16, drop_remainder=True)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds

    def get_train(self):
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset
        
        ds_train = self._common_pipeline(self._ds['train'])
        return ds_train

    def get_val(self):
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset
        
        ds_val = self._common_pipeline(self._ds['validation'])
        return ds_val


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