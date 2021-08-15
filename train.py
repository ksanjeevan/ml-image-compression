


import tensorflow as tf

from trainer import Trainer

from net import ComCNN, RecCNN, ImageCodec

from data import ClicData

ds_train = ClicData().get_train()

d, = ds_train.take(1)



Cr = ComCNN()

Re = RecCNN()



Co = ImageCodec()

t = Trainer(Cr, Re, Co)

t.train()




