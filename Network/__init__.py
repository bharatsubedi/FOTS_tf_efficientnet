import os

import tensorflow as tf

from utils import config

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_list
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

try:
    os.makedirs(config.output_dir)
except OSError as e:
    if e.errno != 17:
        raise
