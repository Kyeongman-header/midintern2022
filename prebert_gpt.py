from bert_summary_maker import *
from models import *
import consts
from tqdm import tqdm, trange
import time
import tensorflow as tf
import os
import numpy as np
import torch


training_RANGE=consts.BATCH_SIZE*272,600
RANGE=consts.BATCH_SIZE*15620 # whole dataset. 물론, 이 중 1024 token을 넘거나 200 token도 안되는 애들은 날려버렸기 때문에
# 실제는 좀 더 적다.


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

filename="EXTRACTIVE_PRET_WHOLE"
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
from rouge import Rouge
f= open(filename+'.csv', 'w', newline='')
wr=csv.writer(f)
wr.writerow(['original','summary','article_g','bleu'])
#wr.writerow(['this is ' + str(10404/(round(312/10))) + ' epoch generation task.'])
f2=open(filename+"_history.csv",'w',newline='')
wr2=csv.writer(f2)
rouge=Rouge()

def prefix_ver1(val_sum):
    return "The summary is : " +val_sum + " And the original text is : "
def prefix_ver2(val_2_sum,mother_plot, i):
    return "This is the second abstract plot. " + "The mother plot is : " + mother_plot + " and the page is : " + str(i) + " and the summary text is : " + val_2_sum[i] + " and the original text is : "

def generate_valid(model,valid_summary,wr,epoch,tokenizer,val_inp,prefix,mother_plots=None):
    c=0
    wr.writerow(['this is ' + str(epoch) + 'epoch generation task.'])
    rouge_avg=0
    outputs=[]
    for val_sum in valid_summary:
        if mother_plots==None:
            val_sum=prefix(val_sum) 
        else :
            val_sum=prefix(val_sum,mother_plots[int(math.ceil(c/5))],c%5) # 
            # c가 0~4까지는 c/5는 0이다. 5~9까지 c/5는 1이다. 이렇게 mother_plot을 5개씩 매칭한다.
            #페이지는 0~4까지 범위를 가진다.
        
        print("val summary ===> " + val_sum) # prefix 부분이다.
        # ~~~!! 적절한 prefix를 붙여줘야 한다!!
        input_ids = tokenizer.encode(val_sum, return_tensors='tf')
        # max length가 1024지, 실제로 1024개를 생성하는 경우는 없을 거다.
        output = tokenizer.decode(model.generate(input_ids,max_length = 1024,do_sample=True,top_p=0.92,top_k=50)[0],skip_special_tokens=True) # no repeat은 dumb repeat을 방지할 수도 있다!
        original=tokenizer.decode(val_inp[c],skip_special_tokens=True)
        
        # ~~~!! prefix는 없애줘야 한다!
        output=output.replace(val_sum,"")
        original=original.repace(val_sum,"") # prefix 부분은 없애준다.
        print("original : " + original)
        print("len of original : " + str(len(tokenizer(original).input_ids)))
        print()
        print("output : " + output)
        print("len of output : " + str(len(tokenizer(output).input_ids)))
        outputs.append(output) # prefix부분은 빠진 output.

        
        r=rouge.get_scores(output,original,avg=True)
        #bleu=sentence_bleu([original],output)
        rouge_avg=rouge_avg+r
        c=c+1
        wr.writerow([original,val_sum,output,r])
            
            
    # 여기서부턴 한꺼번에 perplexity 계산.
    encodings=tokenizer("\n\n".join(valid_summary),return_tensors="tf")
    print("encoding input shape : " )
    print(encodings.input_ids.shape) # (1,287664가 됨!! 여러 글이 한 덩어리로 합쳐짐.)

    max_length = model.config.n_positions
    stride = 512
    size=encodings.input_ids.shape[1]
    nlls = []
    print("model's output max_length : " + str(max_length))
    print("input id size : " + str(encodings.input_ids.shape[1]))
    print("stride : " + str(stride))
    prev_end_loc=0

    # input()

    for begin_loc in tqdm(range(0, size, stride)): # (0~287644)를 512씩 움직인다. 이 경우 
        end_loc = min(begin_loc + max_length, size) 
                # print("begin_loc : " + str(begin_loc))
                # print("end_loc : " + str(end_loc))


        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc] # begin loc은 512씩 가는데 end_loc은 거기서부터 1024까지의 길이이기때문에
                # 0~1024, 512~1536, 1024~2048 이런식으로 input은 겹치면서 나아간다
                # 고로 end에서 1024만큼 뺀 부분은 -100으로 target을 설정해 줌으로써 그부분의 loss는 계산에서 제외할 수 있대(어째서지...)
                # 그니까, 앞의 512개의 context도 전부 보면서, 뒤의 512개의 생성에 대한 perplexity를 계산하겠다는 거지. 
                # 그 와중에 앞의 512개의 perplexity를 또 계산하면 중복되니깐 그 부분은 -100으로 설정해서 없애겠다는거고.
                # 이 stride가 작을 수록 perplexity는 작아짐(좋아짐.) 왜냐하면 겹치면서 보는 context가 많아지니까?
                # print("input_ids : ") # 2차원임[[]]
                # print(input_ids.shape)
                # print("trg_len : " + str(trg_len))

                # target_ids[:, :-trg_len] = -100
        py_target=[]
        for i in range(input_ids.shape[1]):
            if i<input_ids.shape[1]-trg_len:
                py_target.append(-100)
            else:
                py_target.append(input_ids[0][i])


        target_ids=tf.stack([py_target])

                # print("target_ids : ")
                # print(target_ids.shape) # [[]]
                # input()

        outputs = model(input_ids, labels=target_ids)
                # print("outputs : ")
                # print(outputs[0])
                # print(outputs.loss)
        neg_log_likelihood = outputs.loss * trg_len

                # print("neg_log_likelihood : ")
                # print(neg_log_likelihood)
        prev_end_loc = end_loc
        nlls.append(neg_log_likelihood)

    ppl = tf.exp(tf.stack(nlls).sum() / end_loc)

    #print("ppl is : " + str(ppl))

    
    wr.writerow(['this is avg rouge of generation: ' + str(rouge_avg/(round(c/10)))])
    wr.writerow(['this is avg perplexity of generation : ' + str(ppl)])   
    return rouge_avg/(round(c/10)), ppl, outputs


import math
def splitting_output(outputs,tokenizer):
    five_split_outputs=[]
    for o in outputs:
        w=round(len(tokenizer(o).input_ids)/5)
        sl=len(o.split('.'))
        seq=[0]
        for split in o.split('.'):
            count=count+1
            if len(seq)>=6:
                # 5 덩어리에서 끊는다.
                continue
            tokens=tokenizer(split).input_ids
            l=l+len(tokens)
            if l>=w:
                seq.append(count)
                l=len(tokens)
        seq.append(count)
        
        if len(seq)>6:#6덩어리가 있는 상태.
            dist=seq[-1]-seq[-2]+1
            dist=math.ceil(dist/5)

            while seq[5]<sl:
                seq[1]+=dist
                seq[2]+=dist*2
                seq[3]+=dist*3
                seq[4]+=dist*4
                seq[5]+=dist*5
        
        for i in range(len(seq)):
            if i>0:
                mt=('.').join(o.split('.')[seq[i-1]:seq[i]])
                print("each middel target token length : " + str(len(tokenizer(mt).input_ids)))

                five_split_outputs.append(mt)
    return five_split_outputs


for epoch in trange(100): # 20회씩.
    if (epoch+1) % 5==0 :
        rouge_avg,ppl,outputs=generate_valid(model=gpt,valid_summary=valid_summary,wr=wr,epoch=epoch,tokenizer=tokenizer,val_inp=val_inp, prefix=prefix_ver1)
    #if epoch>=0:
        print("rouge_avg : " + str(rouge_avg))
        print("ppl : " + str(ppl))
        print("outputs num : " + str(len(outputs)))
        # hier 구조에서는 , 여기서 나온 outputs를 각각에 대하여 5등분하여서 valid_summary에 먹인다.
        five_split_outputs=splitting_output(outputs,tokenizer)
        print("five split outputs num : " + str(len(five_split_outputs)))
        # rouge_avg,ppl,second_outputs=generate_valid(model=gpt_large,mothe_plots=valid_summary,valid_summary=five_split_outputs,wr=wr2,epoch=epoch,tokenizer=tokenizer,val_inp=val_2_inp, prefix=prefix_ver2)

    history=gpt.fit(x={"input_ids":inp[:,:-1]},y=inp[:,1:],
            
            validation_data=({"input_ids" : val_inp[:,:-1]},val_inp[:,1:]),
            batch_size=consts.BATCH_SIZE,
            callbacks=[tensorboard_callback,checkpoint,])
        #callbacks=[tensorboard_callback,checkpoint,stop_early])
    #print(history)
    print(history.history)

