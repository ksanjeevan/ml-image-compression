

from trainer import Trainer

from net import ComCNN, RecCNN, ImageCodec

import argparse

def get_train_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epochs', default=10)
    argparser.add_argument('--lr-re', default=1e-3, type=float)
    argparser.add_argument('--lr-cr', default=1e-3, type=float)
    argparser.add_argument('--lr', default=None, type=float)
    argparser.add_argument('--batch-size', default=16, type=int)
    argparser.add_argument('--logs', default='logs')
    argparser.add_argument('--resume-path', default=None)
    argparser.add_argument('--no-log', action='store_true')

    args = argparser.parse_args()

    if args.lr is not None:
        args.lr_re = args.lr
        args.lr_cr = args.lr

    return vars(argparser.parse_args())



if __name__ == '__main__':

    config = get_train_args()

    # from data import ClicData
    # ds_train = ClicData().get_train()
    # d, = ds_train.take(1)


    import tensorflow as tf
    tf.random.set_seed(0)

    Cr = ComCNN(num_channels=3)

    Re = RecCNN(num_channels=3)

    Co = ImageCodec()

    t = Trainer(Cr, Re, Co, config=config)

    t.run()




