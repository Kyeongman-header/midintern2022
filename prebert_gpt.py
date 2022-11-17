from bert_summary_maker import *
from models import *
import consts
from tqdm import tqdm, trange
import time
import tensorflow as tf
import os
import numpy as np
import torch


training_RANGE=consts.BATCH_SIZE*10000
RANGE=consts.BATCH_SIZE*7500


TRAIN_FILE="ext_18_train"
VALID_FILE="ext_18_valid"
#위의 두개는 extractive 압축 데이터이다.
FURTHER_TRAIN=False
#summary_maker(RANGE=RANGE,length=800,file=TRAIN_FILE,is_model_or_given_dataset=False)
#summary_maker(RANGE=RANGE,length=800,file=VALID_FILE,is_model_or_given_dataset=False)
summary=np.load("./npdata/"+TRAIN_FILE +"/summary.npy")[:training_RANGE]
token_summary_prefix_target=np.load("./npdata/"+TRAIN_FILE +"/token_summary_prefix_target.npy")[:training_RANGE]
valid_summary=np.load("./npdata/"+VALID_FILE+"/summary.npy")[:RANGE]
valid_token_summary_prefix_target=np.load("./npdata/"+VALID_FILE +"/token_summary_prefix_target.npy")[:RANGE]


summary_prefix_target_length=token_summary_prefix_target.shape[1]
valid_summary_prefix_target_length=valid_token_summary_prefix_target.shape[1]


def make_batch(inp1,inp2,batch_size):
    #inp는 2차원(whole_size, length)이다.
    #목표는 3차원 (whole/batch, batch, length)로 만드는 것이다.
    inp1=np.reshape(inp1,(-1,batch_size,inp1.shape[1]))
    inp2=np.reshape(inp2,(-1,batch_size,inp2.shape[1]))
    return (inp1,inp2)

#inp,val_inp=make_batch(token_summary_prefix_target,valid_token_summary_prefix_target,consts.BATCH_SIZE)
inp=token_summary_prefix_target
val_inp=valid_token_summary_prefix_target

print(inp.dtype)
print(val_inp.dtype)

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

filename="EXTRACTIVE_PRET_18"
#ckpt_manager=model_saver(gpt,gpt_optimizer,filename=filename)
SCL=SparseCategorical_Loss(LAMBDA=consts.LAMBDA,PAD=tokenizer.pad_token_id)
loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
from nltk.translate.bleu_score import sentence_bleu



gpt.compile(
    optimizer=gpt_optimizer,
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
#stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# 일단은 꺼본다.
if FURTHER_TRAIN :
    gpt = keras.models.load_model("/MY_checkpoints/"+filename+"/best_model")

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
f= open(filename+'.csv', 'w', newline='')
wr=csv.writer(f)
wr.writerow(['original','summary','article_g','bleu'])
#wr.writerow(['this is ' + str(10404/(round(312/10))) + ' epoch generation task.'])
f2=open(filename+"_history.csv",'w',newline='')
wr2=csv.writer(f2)

for epoch in trange(20): # 20회씩.
    if (epoch+1) % 5==0 :
    #if epoch>=0:
        c=0
        wr.writerow(['this is ' + str(epoch) + 'epoch generation task.'])
        bleu_avg=0
        for val_sum in valid_summary:
            if (c+1)%10==0:
                print("val summary ===> " + val_sum)
                input_ids = tokenizer.encode(val_sum, return_tensors='tf')
                output = tokenizer.decode(gpt.generate(input_ids,max_length = 1024,do_sample=True,top_p=0.92,top_k=50)[0]) # no repeat은 dumb repeat을 방지할 수도 있다!
                original=tokenizer.decode(val_inp[c])
                print("output ===> ")
                print(output)
                bleu=sentence_bleu([original],output)
                bleu_avg=bleu_avg+bleu
                wr.writerow([original,val_sum,output,bleu])
            c=c+1
        wr.writerow(['this is avg bleu of generation: ' + str(bleu_avg/(round(c/10)))])
    
    history=gpt.fit(x={"input_ids":inp[:,:-1]},y=inp[:,1:],
            
            validation_data=({"input_ids" : val_inp[:,:-1]},val_inp[:,1:]),
            batch_size=consts.BATCH_SIZE, epochs=4,
            callbacks=[tensorboard_callback,checkpoint,])
        #callbacks=[tensorboard_callback,checkpoint,stop_early])
    #print(history)
    print(history.history)

