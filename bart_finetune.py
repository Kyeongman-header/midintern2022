from bert_summary_maker import *
from models import *
import consts
from tqdm import tqdm, trange
import time
import tensorflow as tf
import os
import numpy as np
import torch
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
#training_RANGE=consts.BATCH_SIZE*272600
#RANGE=consts.BATCH_SIZE*1 # whole dataset. 물론, 이 중 1024 token을 넘거나 200 token도 안되는 애들은 날려버렸기 때문에
# 실제는 좀 더 적다.




# TRAIN_FILE="abs_whole_train_gpt"
# VALID_FILE="abs_whole_valid_gpt"
#위의 두개는 extractive 압축 데이터이다.
FURTHER_TRAIN=False
#summary_maker(RANGE=RANGE,length=800,file=TRAIN_FILE,is_model_or_given_dataset=False)
#summary_maker(RANGE=RANGE,length=800,file=VALID_FILE,is_model_or_given_dataset=False)
# summary=np.load("./npdata/"+TRAIN_FILE +"/summary.npy")[:training_RANGE]
# token_summary_prefix_target=np.load("./npdata/"+TRAIN_FILE +"/token_summary_prefix_target.npy")[:training_RANGE]
# valid_summary=np.load("./npdata/"+VALID_FILE+"/summary.npy")[:RANGE]
# valid_token_summary_prefix_target=np.load("./npdata/"+VALID_FILE +"/token_summary_prefix_target.npy")[:RANGE]


# summary_prefix_target_length=token_summary_prefix_target.shape[1]
# valid_summary_prefix_target_length=valid_token_summary_prefix_target.shape[1]


# def make_batch(inp1,inp2,batch_size):
#     #inp는 2차원(whole_size, length)이다.
#     #목표는 3차원 (whole/batch, batch, length)로 만드는 것이다.
#     inp1=np.reshape(inp1,(-1,batch_size,inp1.shape[1]))
#     inp2=np.reshape(inp2,(-1,batch_size,inp2.shape[1]))
#     return (inp1,inp2)

# #inp,val_inp=make_batch(token_summary_prefix_target,valid_token_summary_prefix_target,consts.BATCH_SIZE)
# inp=token_summary_prefix_target
# val_inp=valid_token_summary_prefix_target

# print(inp.dtype)
# print(val_inp.dtype)

#dataset = tf.data.Dataset.from_tensor_slices((inp[:,:-1],inp[:,1:]))
#val_dataset =  tf.data.Dataset.from_tensor_slices((val_inp[:,:-1], val_inp[:,1:]))

if os.path.isfile("./npdata/"+"bartfinetune"+"/summary.npy") is not True:
    T="train"
    total_source=[]
    with open("writingPrompts/"+ T +".wp_source", encoding='UTF8') as f:
        stories = f.readlines()
        stories = [" ".join(i.split()[0:1000]) for i in stories]
        temp_stories=[]
        for story in stories:
            temp_stories.append(story.replace("<newline>",""))
        total_source.append(temp_stories)

    total_target=[]

    with open("writingPrompts/"+ T +".wp_target", encoding='UTF8') as f:
        stories = f.readlines()
        stories = [" ".join(i.split()[0:1000]) for i in stories]
        temp_stories=[]
        for story in stories:
            temp_stories.append(story.replace("<newline>",""))
        total_target.append(temp_stories)


    token_target=tokenizer(total_target[0],return_tensors="tf",padding="max_length",max_length=1024, truncation=True).input_ids
    token_summary=tokenizer(total_source[0],return_tensors="tf",padding="max_length",max_length=1024, truncation=True).input_ids


    nptoken_summary=token_summary.numpy()
    nptoken_target=token_target.numpy()
    
    createFolder("npdata/"+"bartfinetune")

    if os.path.isfile("./npdata/"+"bartfinetune"+"/token_summary.npy"):
        past=np.load("./npdata/"+"bartfinetune"+"/token_summary.npy")
        nptoken_summary=np.concatenate((past,nptoken_summary),axis=0)
    np.save("./npdata/"+"bartfinetune" +"/token_summary",nptoken_summary)

    if os.path.isfile("./npdata/"+"bartfinetune"+"/token_target.npy"):
        past=np.load("./npdata/"+"bartfinetune"+"/token_target.npy")
        nptoken_target=np.concatenate((past,nptoken_target),axis=0)
    np.save("./npdata/"+"bartfinetune" +"/token_target",nptoken_target)



token_target=np.load("./npdata/"+"bartfinetune" +"/token_target.npy")[:]
token_summary=np.load("./npdata/"+ "bartfinetune" +"/token_summary.npy")[:]
token_target=tf.cast(token_target,tf.int64)
token_summary=tf.cast(token_summary,tf.int64)


print(token_target.shape)
print(token_summary.shape)


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

filename="FINETUNE_BART"

from transformers import AutoTokenizer

#bart=TFAutoModelForSeq2SeqLM.from_pretrained("patrickvonplaten/t5-tiny-random")
bart_optimizer=tf.keras.optimizers.Adam(learning_rate=consts.LEARNING_RATE)

if FURTHER_TRAIN:
    gpt = TFGPT2LMHeadModel.from_pretrained("./MY_checkpoints/"+filename+"/bart")
    bart_optimizer=tf.keras.optimizers.Adam(learning_rate=consts.LEARNING_RATE)
    
else:
    bart = TFAutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    bart_optimizer=tf.keras.optimizers.Adam(learning_rate=consts.LEARNING_RATE)
    


#ckpt_manager=model_saver(gpt,gpt_optimizer,filename=filename)
SCL=SparseCategorical_Loss(LAMBDA=consts.LAMBDA,PAD=tokenizer.pad_token_id)
loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
from nltk.translate.bleu_score import sentence_bleu



bart.compile(
    optimizer=bart_optimizer,
     metrics=SCL.reconstruction_accuracy_function,
     #metrics=[SCL.reconstruction_accuracy_function, SCL.bleu_function], # accuracy도 masking 된 acc 여야 한다.
     #run_eagerly = True,
     loss=SCL.reconstruction_loss,
)
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/bart_finetune/' + filename + current_time

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="./MY_checkpoints/"+filename+"/best_model",save_best_only=True)
#stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# 일단은 꺼본다.
# if FURTHER_TRAIN :
#     gpt = keras.models.load_model("/MY_checkpoints/"+filename+"/best_model")

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
wr.writerow(['original','summary','article_g','rouge-1','rouge-2','rouge-l'])

from hier_utility import *


for epoch in trange(100): # 20회씩.
    # if (epoch) % 5==0 :
    #     r_1_avg,r_2_avg,r_l_avg,ppl,outputs=generate_valid(model=gpt,valid_summary=valid_summary,wr=wr,epoch=epoch,tokenizer=tokenizer,val_inp=val_inp, prefix=prefix_ver0)
    # #if epoch>=0:
    #     print("rouge_avg : " + str(r_1_avg)+" "+str(r_2_avg)+" "+str(r_l_avg))
    #     print("ppl : " + str(ppl))
    #     print("outputs num : " + str(len(outputs)))

    history=bart.fit(x={"input_ids":token_summary[:,:1024],"decoder_input_ids":token_target[:,:1023]},y=token_target[:,1:1024],
            
            #validation_data=({"input_ids" : val_inp[:,:-1]},val_inp[:,1:]),
            batch_size=consts.BATCH_SIZE,
            callbacks=[tensorboard_callback,checkpoint,])
    
    bart.save_pretrained("./MY_checkpoints/"+filename+"/bart") 
        #callbacks=[tensorboard_callback,checkpoint,stop_early])
    #print(history)
    print(history.history)

