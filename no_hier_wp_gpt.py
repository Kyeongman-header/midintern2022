from bert_summary_maker import *
from models import *
import consts
from tqdm import tqdm, trange
import time
import tensorflow as tf
import os
import numpy as np
import torch

tf.config.set_visible_devices([], 'GPU') # CPU로 학습하기.
# train_RANGE=consts.BATCH_SIZE*20000
valid_RANGE=consts.BATCH_SIZE*100 # whole dataset. 물론, 이 중 1024 token을 넘거나 200 token도 안되는 애들은 날려버렸기 때문에
# 실제는 좀 더 적다.


TRAIN_FILE="abs_whole_train_gpt"
VALID_FILE="abs_whole_valid_gpt"
#위의 두개는 extractive 압축 데이터이다.
FURTHER_TRAIN=False
#summary_maker(RANGE=RANGE,length=800,file=TRAIN_FILE,is_model_or_given_dataset=False)
#summary_maker(RANGE=RANGE,length=800,file=VALID_FILE,is_model_or_given_dataset=False)

summary=np.load("./npdata/"+TRAIN_FILE +"/mother.npy")[:]
inp_1=np.load("./npdata/"+TRAIN_FILE +"/middle_tokens.npy")[:]
inp_2=np.load("./npdata/"+TRAIN_FILE +"/final_tokens.npy")[:]
print()
print("whole size : " + str(len(summary)) + " , " + str(len(inp_1)) + " , "+ str(len(inp_2)))

valid_summary=summary[-valid_RANGE:]
valid_inp_1=inp_1[-valid_RANGE:]
valid_inp_2=inp_2[-valid_RANGE*5:]

summary=summary[:-valid_RANGE]
inp_1=inp_1[:-valid_RANGE]
inp_2=inp_2[:-valid_RANGE*5]
# training 데이터와 valid 데이터의 slice.

print()
print("길이 정보. 차례로 summary, inp_1, inp_2")
print(len(summary))
print(len(inp_1))
print(len(inp_2))
print("valid : ")
print(len(valid_summary))
print(len(valid_inp_1))
print(len(valid_inp_2)) # inp1과 inp2는 언제나 5배 관계여야 한다.


def make_batch(inp1,inp2,batch_size):
    #inp는 2차원(whole_size, length)이다.
    #목표는 3차원 (whole/batch, batch, length)로 만드는 것이다.
    inp1=np.reshape(inp1,(-1,batch_size,inp1.shape[1]))
    inp2=np.reshape(inp2,(-1,batch_size,inp2.shape[1]))
    return (inp1,inp2)

# print(inp.dtype)
# print(val_inp.dtype)

#dataset = tf.data.Dataset.from_tensor_slices((inp[:,:-1],inp[:,1:]))
#val_dataset =  tf.data.Dataset.from_tensor_slices((val_inp[:,:-1], val_inp[:,1:]))

from transformers import GPT2Config,TFGPT2LMHeadModel, TFAutoModelForSeq2SeqLM
print("vocab size : " + str(tokenizer.vocab_size)) # special token 더해준거 개수를 직접 더해야 한다...!!!
"""
config = GPT2Config(
  vocab_size=tokenizer.vocab_size+5,
  bos_token_id=tokenizer.bos_token_id,
  eos_token_id=tokenizer.eos_token_id,
  pad_token_id=tokenizer.pad_token_id,
)
"""
#gpt = TFGPT2LMHeadModel(config) # 이렇게 안 하면 eos token 등등이 없는 GPT 모델을 불러온다!
# 이렇게 하면 이제 PRETRAINED 된 모델을 사용하지 못한다
gpt = TFGPT2LMHeadModel.from_pretrained("gpt2")
gpt_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
gpt_large=TFGPT2LMHeadModel.from_pretrained("gpt2") # large로 학습하려 했으나, 그냥 메모리에 로드 자체가 안된다....
gpt_large_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)

filename="REEDSY_HIER"
#ckpt_manager=model_saver(gpt,gpt_optimizer,filename=filename)
SCL=SparseCategorical_Loss(LAMBDA=consts.LAMBDA,PAD=tokenizer.pad_token_id)
loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# from nltk.translate.bleu_score import sentence_bleu

if FURTHER_TRAIN :
    #gpt.load_weights("./MY_checkpoints/"+filename+"/best_model")
    #gpt_large.load_weights("./MY_checkpoints/"+filename+"/best_model_large")
    gpt = tf.keras.models.load_model("./MY_checkpoints/"+filename+"/best_model",custom_objects={"reconstruction_accuracy_function" : SCL.reconstruction_accuracy_function,'reconstruction_loss' : SCL.reconstruction_loss})
    gpt_large= tf.keras.models.load_model("./MY_checkpoints/"+filename+"/best_model_large",custom_objects={"reconstruction_accuracy_function" : SCL.reconstruction_accuracy_function,"reconstruction_loss" : SCL.reconstruction_loss})


gpt.compile(
    optimizer=gpt_optimizer,
     metrics=SCL.reconstruction_accuracy_function,
     #metrics=[SCL.reconstruction_accuracy_function, SCL.bleu_function], # accuracy도 masking 된 acc 여야 한다.
     #run_eagerly = True,
     loss=SCL.reconstruction_loss,
)
gpt_large.compile(
    optimizer=gpt_large_optimizer,
     metrics=SCL.reconstruction_accuracy_function,
     #metrics=[SCL.reconstruction_accuracy_function, SCL.bleu_function], # accuracy도 masking 된 acc 여야 한다.
     #run_eagerly = True,
     loss=SCL.reconstruction_loss,
)

import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/forth/' + filename + current_time

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="./MY_checkpoints/"+filename+"/best_model",save_best_only=True)
checkpoint_large = tf.keras.callbacks.ModelCheckpoint(filepath="./MY_checkpoints/"+filename+"/best_model_large",save_best_only=True)
#stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# 일단은 꺼본다.

#print("//////////////////////////")
#print("sample:")
#print(tokenizer.decode(inp[0,:-1]))
#print(tokenizer.decode(inp[0,1:]))
#sampleout=gpt({"input_ids" : inp[0:1,:-1],"labels" : inp[0:1,1:]})
#print(sampleout.logits)
#print(tokenizer.decode(tf.argmax(sampleout.logits,axis=2,output_type=inp.dtype)[0]))
#print(sampleout.loss)
#print(SCL.reconstruction_loss(inp[0:1,1:],sampleout.logits))

import csv

f= open(filename+'.csv', 'w', newline='',encoding='utf-8')
wr=csv.writer(f)
wr.writerow(['original','summary','article_g','rouge-1','rouge-2','rouge-l'])
#wr.writerow(['this is ' + str(10404/(round(312/10))) + ' epoch generation task.'])
f2=open(filename+"_large.csv",'w',newline='',encoding='utf-8')
wr2=csv.writer(f2)
wr2.writerow(['original','summary','article_g','rouge-1','rouge-2','rouge-l'])

from hier_utility import *


for epoch in trange(100): # 20회씩.
    if (epoch) % 10==0 :
        #r_1_avg,r_2_avg,r_l_avg,ppl,outputs=generate_valid(model=gpt,valid_summary=valid_summary,wr=wr,epoch=epoch,tokenizer=tokenizer,val_inp=valid_inp_1, prefix=prefix_ver1)
    #if epoch>=0:
        #print("rouge_avg : " + str(r_1_avg)+" "+str(r_2_avg)+" "+str(r_l_avg))
        #print("ppl : " + str(ppl))
        #print("outputs num : " + str(len(outputs)))
        #print("outputs : ")
        #print(outputs)
        # hier 구조에서는 , 여기서 나온 outputs를 각각에 대하여 5등분하여서 valid_summary에 먹인다.
        #five_split_outputs=splitting_output(outputs,tokenizer)
        #print("five split outputs num : " + str(len(five_split_outputs)))
        #large_r_1_avg,large_r_2_avg,large_r_l_avg,large_ppl,large_second_outputs=generate_valid(model=gpt_large,mother_plots=valid_summary,valid_summary=five_split_outputs,wr=wr2,epoch=epoch,tokenizer=tokenizer,val_inp=valid_inp_2, prefix=prefix_ver2)
        large_r_1_avg,large_r_2_avg,large_r_l_avg,large_ppl,large_second_outputs=generate_valid(model=gpt_large,mother_plots=valid_summary,valid_summary=[],wr=wr2,epoch=epoch,tokenizer=tokenizer,val_inp=valid_inp_2, prefix=prefix_ver2)
    
    history=gpt.fit(x={"input_ids":inp_1[:,:-1]},y=inp_1[:,1:],
            
            validation_data=({"input_ids" : valid_inp_1[:,:-1]},valid_inp_1[:,1:]),
            batch_size=consts.BATCH_SIZE,
            callbacks=[tensorboard_callback,checkpoint,])
    
    print(history.history)
    
    history_large=gpt_large.fit(x={"input_ids":inp_2[:,:-1]},y=inp_2[:,1:],
            
            validation_data=({"input_ids" : valid_inp_2[:,:-1]},valid_inp_2[:,1:]),
            batch_size=consts.BATCH_SIZE,
            callbacks=[tensorboard_callback,checkpoint_large,])

    
    print(history_large.history)
