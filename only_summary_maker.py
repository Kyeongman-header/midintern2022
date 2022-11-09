#-*- coding:utf-8 -*-
import tensorflow as tf
#tf.config.set_visible_devices([], 'GPU')
from summary_maker import *
from bert_summary_maker import *
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

RANGE=consts.BATCH_SIZE * 10000 # if '0', then whole range is target.

TRAIN_FILE="ext_18_train"
VALID_FILE="ext_18_valid"
FURTHER_TRAIN=True
bert_summary_maker(START=0,RANGE=RANGE,seq_length=18,file=TRAIN_FILE,is_model_or_given_dataset=True,device=0)
bert_summary_maker(START=0,RANGE=RANGE,seq_length=18,file=VALID_FILE,is_model_or_given_dataset=True,device=0)
#summary_maker(START=0,RANGE=RANGE,seq_length=18,file=TRAIN_FILE,is_model_or_given_dataset=True,device=0)
#summary_maker(START=0,RANGE=RANGE,seq_length=18,file=VALID_FILE,is_model_or_given_dataset=True,device=0)
