from .comcnn import *
from .reccnn import *
from .codec import *

from pathlib import Path
import tensorflow as tf

def load_model(model_name : str, model_path : str, input_shape : list):

  model = ComCNN() if model_name == 'cr' else RecCNN()
  model(tf.zeros(input_shape))
  path = Path(model_path).joinpath(f'{model_name}/model')
  model.load_weights(path)
  return model


def get_full_model(model_path : str, input_shape : list):
  Cr = load_model('cr', model_path, input_shape)
  Co = ImageCodec()
  Re = load_model('re', model_path, input_shape)
  return lambda x: Re(Co(Cr(x)))


