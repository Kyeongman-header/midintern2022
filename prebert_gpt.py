from transformers import TFGPT2LMHeadModel
from bert_summary_maker import *
from models import *
import consts
from tqdm import tqdm, trange
import time
import tensorflow as tf
import os
import numpy as np
import torch



RANGE=consts.BATCH_SIZE*3500


TRAIN_FILE="ext_18_train"
VALID_FILE="ext_18_valid"
#위의 두개는 extractive 압축 데이터이다.
FURTHER_TRAIN=False
#summary_maker(RANGE=RANGE,length=800,file=TRAIN_FILE,is_model_or_given_dataset=False)
#summary_maker(RANGE=RANGE,length=800,file=VALID_FILE,is_model_or_given_dataset=False)

token_summary_prefix_target=np.load("./npdata/"+TRAIN_FILE +"/token_summary_prefix_target.npy")[:RANGE]
valid_token_summary_prefix_target=np.load("./npdata/"+VALID_FILE +"/token_summary_prefix_target.npy")[:RANGE]


summary_prefix_target_length=token_summary_prefix_target.shape[1]
valid_summary_prefix_target_length=valid_token_summary_prefix_target.shape[1]


def make_batch(inp1,inp2,batch_size):
    #inp는 2차원(whole_size, length)이다.
    #목표는 3차원 (whole/batch, batch, length)로 만드는 것이다.
    inp1=np.reshape(inp1,(-1,batch_size,inp1.shape[1]))
    inp2=np.reshape(inp2,(-1,batch_size,inp2.shape[1]))
    return (inp1,inp2)

inp,val_inp=make_batch(token_summary_prefix_target,valid_token_summary_prefix_target,consts.BATCH_SIZE)
print(inp.shape)
print(val_inp.shape)


from transformers import TFGPT2Model, TFAutoModelForSeq2SeqLM
gpt = TFGPT2Model.from_pretrained("gpt2")
gpt_optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5)

filename="EXTRACTIVE_PRET_18"
ckpt_manager=model_saver(gpt,gpt_optimizer,filename=filename)
scce_loss=SparseCategorical_Loss(LAMBDA=consts.LAMBDA,PAD=tokenizer("<pad>")['input_ids'][1])

gpt.compile(
     optimizer=gpt_optimizer,
     metrics=tf.metrics.SparseCategoricalAccuracy(),
     loss=scce_loss,
)
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/third/' + filename + current_time

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model",save_best_only=True)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)

if FURTHER_TRAIN :
    gpt = keras.models.load_model('best_model')

gpt.fit(x={"input_ids" : token_summary,"decoder_input_ids": token_target[:,:-1]},
y=token_target[:,1:],
validation_data=({"input_ids" : valid_token_summary,"decoder_input_ids":valid_token_target[:,:-1]},valid_token_target[:,1:]),
batch_size=2, epochs=20,
callbacks=[tensorboard_callback,checkpoint,stop_early])

