import tensorflow as tf
#tf.config.set_visible_devices([], 'GPU')
from summary_maker import *
import consts
mirrored_strategy = tf.distribute.MirroredStrategy()
gpus = tf.config.experimental.list_logical_devices('GPU') # 멀티 gpu 세팅.
RANGE=consts.BATCH_SIZE*10

TRAIN_FILE="_sm_train_whole"
VALID_FILE="_sm_valid_whole"
FURTHER_TRAIN=True
summary_maker(START=0,RANGE=RANGE,seq_length=7,file=TRAIN_FILE,is_model_or_given_dataset=True)
summary_maker(START=0,RANGE=RANGE,seq_length=7,file=VALID_FILE,is_model_or_given_dataset=True)
