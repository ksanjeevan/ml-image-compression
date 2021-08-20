


import tensorflow as tf

from trainer import Trainer

from net import ComCNN, RecCNN, ImageCodec

from data import ClicData


if __name__ == '__main__':


    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epochs', default=10)
    argparser.add_argument('--lr-re', default=1e-3, type=float)
    argparser.add_argument('--lr-cr', default=1e-3, type=float)
    argparser.add_argument('--logs', default='logs')
    argparser.add_argument('--resume-path', default=None)
    argparser.add_argument('--no-log', action='store_true')

    args = argparser.parse_args()


    #ds_train = ClicData().get_train()
    #d, = ds_train.take(1)


    tf.random.set_seed(0)

    Cr = ComCNN(num_channels=3)

    Re = RecCNN(num_channels=3)

    Co = ImageCodec()

    t = Trainer(Cr, Re, Co, config=vars(args))

    t.run()




