
import argparse

def get_train_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--mode')
    argparser.add_argument('--epochs', default=10)
    argparser.add_argument('--lr', default=0.001, type=float)
    
    # "γ in the range [0.0001, 0.2]"
    argparser.add_argument('--gamma', default=0.005, type=float)
    argparser.add_argument('--rate', default=0.5, type=float)
    argparser.add_argument('--L', default=16, type=int)
    argparser.add_argument('--n', default=64, type=int)

    argparser.add_argument('--batch-size', default=16, type=int)
    argparser.add_argument('--logs', default='logs')
    argparser.add_argument('--resume-path', default=None)
    argparser.add_argument('--no-log', action='store_true')

    args = argparser.parse_args()

    return vars(argparser.parse_args())



if __name__ == '__main__':

    config = get_train_args()

    # from data import ClicData
    # ds_train = ClicData().get_train()
    # d, = ds_train.take(1)


    import tensorflow as tf
    tf.random.set_seed(0)

    if config['mode'] == 'content':
        from trainer.content_weight import Trainer

    elif config['mode'] == 'full':
        from trainer.full_cnn import Trainer
        names = ['gamma', 'rate', 'L', 'n']
        for n in names:
            del config[n]
        
    else:
        raise ValueError('Enter a valid `mode`')


    t = Trainer(config=config)
    t.run()


