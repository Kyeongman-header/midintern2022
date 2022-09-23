from re import T
from datasets import load_dataset
import time
import tensorflow as tf
import os
import numpy as np
import torch
#tf.config.set_visible_devices([], 'GPU') # CPU로 학습하기.
from summary_maker import *
from models import *
import consts
from tqdm import tqdm, trange

mirrored_strategy = tf.distribute.MirroredStrategy()
gpus = tf.config.experimental.list_logical_devices('GPU') # 멀티 gpu 세팅.
# tf.debugging.set_log_device_placement(True)
RANGE=consts.BATCH_SIZE*2900

TRAIN_FILE="_sm_train"
VALID_FILE="_sm_valid"
#위의 두개는 #bart large cnn 압축 데이터이다.
FURTHER_TRAIN=False
#summary_maker(RANGE=RANGE,length=800,file=TRAIN_FILE,is_model_or_given_dataset=False)
#summary_maker(RANGE=RANGE,length=800,file=VALID_FILE,is_model_or_given_dataset=False)

token_summary=np.load("./npdata/"+TRAIN_FILE +"/token_summary.npy")[:RANGE]
token_target=np.load("./npdata/"+TRAIN_FILE +"/token_target.npy")[:RANGE]
valid_token_summary=np.load("./npdata/"+VALID_FILE +"/token_summary.npy")[:RANGE]
valid_token_target=np.load("./npdata/"+VALID_FILE +"/token_target.npy")[:RANGE]

def make_batch(inp1,inp2,batch_size):
    #inp는 2차원(whole_size, length)이다.
    #목표는 3차원 (whole/batch, batch, length)로 만드는 것이다.
    inp1=np.reshape(inp1,(-1,batch_size,inp1.shape[1]))
    inp2=np.reshape(inp2,(-1,batch_size,inp2.shape[1]))
    return (inp1,inp2)

token_summary,token_target=make_batch(token_summary,token_target,consts.BATCH_SIZE)
val_token_summary,val_token_target=make_batch(valid_token_summary,valid_token_target,consts.BATCH_SIZE)

summary_length=token_summary.shape[1]
target_length=token_target.shape[1]
valid_summary_length=valid_token_summary.shape[1]
valid_target_length=valid_token_target.shape[1]

from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
bart = TFAutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
#bart=TFAutoModelForSeq2SeqLM.from_pretrained("patrickvonplaten/t5-tiny-random")
bart_optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5)
disc_optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5)

filename="NOGAN_PRET_SM"
#ckpt_manager=model_saver(bart,bart_optimizer,filename=filename)

ae_loss=SparseCategorical_Loss(LAMBDA=consts.LAMBDA)

#print(token_summary[3].shape)
#print(token_target[3][:-1].shape)

#print(token_summary[3])
#print(token_target[3,:,:-1])
print(tokenizer("<pad>")['input_ids'])
bart_t_output=bart({"input_ids" : token_summary[3],"decoder_input_ids": token_target[3,:,:-1]}).logits
sparse_loss=ae_loss.reconstruction_loss(token_target[3,:,1:],bart_t_output)

bart_t_output=tf.argmax(bart_t_output,axis=2,output_type=token_target.dtype)[0]
bart_g_output=bart.generate(token_summary[3])
origin=tokenizer.decode(token_target[3][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
summary= tokenizer.decode(token_summary[3][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
g_output=tokenizer.decode(bart_g_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
t_output=tokenizer.decode(bart_t_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(bart_t_output)
print(t_output)
print(bart_g_output)
print(g_output)
print()
print(origin)
print()
print(summary)
