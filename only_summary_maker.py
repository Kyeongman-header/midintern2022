#-*- coding:utf-8 -*-
import tensorflow as tf
#tf.config.set_visible_devices([], 'GPU')
from summary_maker import *
import consts
import torch
 
 #  Returns a bool indicating if CUDA is currently available.
print(torch.cuda.is_available())
print(torch.cuda.device_count())
torch.cuda.device(0)
#mirrored_strategy = tf.distribute.MirroredStrategy()
#gpus = tf.config.experimental.list_logical_devices('GPU') # 멀티 gpu 세팅.
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

RANGE=consts.BATCH_SIZE * 5000 # if '0', then whole range is target.

TRAIN_FILE="_sm_train_whole"
VALID_FILE="_sm_valid_whole"
FURTHER_TRAIN=True
summary_maker(START=0,RANGE=RANGE,seq_length=0,file=TRAIN_FILE,is_model_or_given_dataset=True,device=0)
summary_maker(START=0,RANGE=RANGE,seq_length=0,file=VALID_FILE,is_model_or_given_dataset=True,device=0)
