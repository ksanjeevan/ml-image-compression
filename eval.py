import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from net import ImageCodec
from train import get_train_args
from trainer import Trainer


class Evaluator:

    def __init__(self, config : dict):

        self._trainer = Trainer(Cr=None, Re=None, Co=ImageCodec(), config=config)

        self.loss_re = tf.keras.metrics.Mean() 
        self.loss_cr = tf.keras.metrics.Mean() 
        self.ssim = tf.keras.metrics.Mean() 

    def get_metrics(self):

        _val_iter = tqdm(self._trainer.ds_val, leave=False)
        for batch_idx, images in enumerate(_val_iter):
            val_results = self._trainer.val_step(images)

            self.loss_re(val_results['metrics']['loss_re_val'])
            self.loss_cr(val_results['metrics']['loss_cr_val'])
            self.ssim(val_results['metrics']['ssim_val'])

        return {
                    'loss_cr' : self.loss_cr.result().numpy(),
                    'loss_re' : self.loss_re.result().numpy(),
                    'ssim' : self.ssim.result().numpy()
        }

    def get_all(self):
        _val_iter = tqdm(self._trainer.ds_val.unbatch())
        
        image_map = {}
        ret = []
        
        for index, image in enumerate(_val_iter):
            val_results = self._trainer.val_step(image[None])
            
            image_map[index] = {'original' : image.numpy(),
                                'output'   : val_results['tensors']['outputs'][0].numpy()}


            ret.append([val_results['metrics']['loss_cr_val'].numpy(),
                        val_results['metrics']['loss_re_val'].numpy(),
                        val_results['metrics']['ssim_val'].numpy()[0]])
            
        cols = ['loss_cr', 'loss_re', 'ssim']
        df = pd.DataFrame(ret, columns=cols)

        return {'image_map'     : image_map, 
                'image_metrics' : df.sort_values('ssim', ascending=False)}


if __name__ == '__main__':

    config = get_train_args()
    config['no_log'] = True

    e = Evaluator(config)
    #metrics = e.get_metrics()
    # print(metrics)
    e.get_all()


    