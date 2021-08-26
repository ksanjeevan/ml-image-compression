
#https://www.tensorflow.org/datasets/api_docs/python/tfds/load#args

import tensorflow as tf
import tensorflow_datasets as tfds

# https://knowyourdata-tfds.withgoogle.com/#dataset=clic&tab=STATS
# https://www.tensorflow.org/guide/data_performance#prefetching
#https://www.tensorflow.org/datasets/performances


class ClicData:
    # https://www.tensorflow.org/datasets/catalog/clic
    def __init__(self, config : dict={}):
        splits = ['train', 'test', 'validation']
        ds, self.ds_info = tfds.load('clic', 
                                     split=splits, 
                                     download=True, 
                                     shuffle_files=False,
                                     with_info=True,
                                     #read_config=tfds.ReadConfig(shuffle_seed=0),
                                     )


        self._ds = dict(zip(splits, ds))
        self.batch_size = config.get('batch_size', 16)

        self.random_crop = tf.keras.layers.RandomCrop(40, 40, seed=0)


    def image_transforms(self, image_dic):
        image = image_dic['image']
        image = tf.image.resize(image, (180, 180))
        image = image / 255.0
        #image = tf.image.rgb_to_grayscale(image)
        return image

    def augmentations(self, image):
        image = self.random_crop(image)
        return image

    def pipeline(self, ds, train=False):
        ds = ds.map(lambda x: self.image_transforms(x),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds = ds.cache() # random transforms after cache
        if train:
            ds = ds.map(lambda x: self.augmentations(x),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
            #ds = ds.shuffle(self.ds_info.splits['train'].num_examples)


        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds

    def get_train(self):
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset
        
        return self.pipeline(self._ds['train'], train=True)

    def get_val(self):
        return self.pipeline(self._ds['test'])


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