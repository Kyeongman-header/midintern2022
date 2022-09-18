import tensorflow as tf
#tf.config.set_visible_devices([], 'GPU')
from summary_maker import *
import consts
mirrored_strategy = tf.distribute.MirroredStrategy()
gpus = tf.config.experimental.list_logical_devices('GPU') # 멀티 gpu 세팅.
RANGE=consts.BATCH_SIZE*30000

TRAIN_FILE="_train"
VALID_FILE="_valid"
FURTHER_TRAIN=True
summary_maker(START=consts.BATCH_SIZE*3000,RANGE=RANGE,length=750,file=TRAIN_FILE,is_model_or_given_dataset=True)
#summary_maker(START=consts.BATCH_SIZE*3000,RANGE=RANGE,length=750,file=VALID_FILE,is_model_or_given_dataset=True)
