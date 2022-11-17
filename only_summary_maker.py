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

RANGE=consts.BATCH_SIZE * 5 # if '0', then whole range is target.

EXT_TRAIN_GPT_FILE="ext_18_train_gpt"
EXT_VALID_GPT_FILE="ext_18_valid_gpt"
ABS_TRAIN_GPT_FILE="abs_18_train_gpt"
ABS_VALID_GPT_FILE="abs_18_valid_gpt"

EXT_TRAIN_BART_FILE="ext_18_train_bart"
EXT_VALID_BART_FILE="ext_18_valid_bart"
ABS_TRAIN_BART_FILE="abs_18_train_bart"
ABS_VALID_BART_FILE="abs_18_valid_bart"

FURTHER_TRAIN=True
bert_summary_maker(START=0,RANGE=RANGE,is_abs_or_ext=False,seq_length=18,file=EXT_TRAIN_GPT_FILE,is_model_or_given_dataset=True,device=0)
bert_summary_maker(START=0,RANGE=RANGE,is_abs_or_ext=False,seq_length=18,file=EXT_VALID_GPT_FILE,is_model_or_given_dataset=True,device=0)
bert_summary_maker(START=0,RANGE=RANGE,is_abs_or_ext=True,seq_length=18,file=ABS_TRAIN_GPT_FILE,is_model_or_given_dataset=True,device=0)
bert_summary_maker(START=0,RANGE=RANGE,is_abs_or_ext=True,seq_length=18,file=ABS_VALID_GPT_FILE,is_model_or_given_dataset=True,device=0)

#summary_maker(START=0,RANGE=RANGE,is_abs_or_ext=False,seq_length=18,file=EXT_TRAIN_BART_FILE,is_model_or_given_dataset=True,device=0)
#summary_maker(START=0,RANGE=RANGE,is_abs_or_ext=False,seq_length=18,file=EXT_VALID_BART_FILE,is_model_or_given_dataset=True,device=0)
#summary_maker(START=0,RANGE=RANGE,is_abs_or_ext=True,seq_length=18,file=ABS_TRAIN_BART_FILE,is_model_or_given_dataset=True,device=0)
#summary_maker(START=0,RANGE=RANGE,is_abs_or_ext=True,seq_length=18,file=ABS_VALID_BART_FILE,is_model_or_given_dataset=True,device=0)
