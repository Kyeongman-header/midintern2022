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

for epoch in trange(100): # 20회씩.
    if (epoch+1) % 5==0 :
    #if epoch>=0:
        c=0
        wr.writerow(['this is ' + str(epoch) + 'epoch generation task.'])
        rouge_avg=0
        for val_sum in valid_summary:
            if (c+1)%10==0:
                val_sum="The summary is : " + val_sum + " And the original text is : "
                print("val summary ===> " + val_sum)
                # ~~~!! 적절한 prefix를 붙여줘야 한다!!
                input_ids = tokenizer.encode(val_sum, return_tensors='tf')
                output = tokenizer.decode(gpt.generate(input_ids,max_length = 1024,do_sample=True,top_p=0.92,top_k=50)[0]) # no repeat은 dumb repeat을 방지할 수도 있다!
                original=tokenizer.decode(val_inp[c])
                
                # ~~~!! prefix는 없애줘야 한다!
                output=output.replace(val_sum,"")
                original=original.repace(val_sum,"") # summarize 한 prefix 부분은 없애준다.
                print("original : " + original)
                print("len of original : " + str(len(tokenizer(original).input_ids)))
                print()
                print("output : " + output)
                print("len of output : " + str(len(tokenizer(output).input_ids)))

                # 개선점. 일단 , 이 replace가 잘 작동 안 할 수도 있으니깐 한번 확인 해주고
                # perplexity는 generate이 아니라 model에 직접 넣은 결과를 봐야함
                # 그리고 이 generate도 웬만하면 병렬로 하는게 좋을 듯. 즉 input_ids를 []로 한번 감싸서 2차원으로 만든다음에
                # before=np.reshape(before,input_ids,axis=0)
                # 그리고서 generate에다가 넣는다.
                # 근데 생각해보니 어차피 병렬로 해봤자 batch size 2정도네... 걍 해도 될듯...
                
                # 그리고 hier 구조는 총 두개의 모델을 학습해야 함.
                # middle_tokens을 학습할 gpt,
                # final_tokens을 학습할 gpt-large
                # 이거는 걍 fit 두번 하는 거지만
                # generate이 좀 복잡함. gpt가 먼저 생성한 다음에,(그게 어느정도 길이가 되는지 먼저 관찰하자. 약 600문장 이상이 되여야 적절하다) 
                # 그 값을 5등분 하여서
                # gpt-large에게 차례로 적절한 prefix와 함께 먹인다.
                # 그 최종적인 결과를 마침내 reedsy writing prompt와 rouge 비교한다.
                
                #그리고 perplexity는 각각의 모델을 따로 계산한다. 
                r=rouge.get_scores(output,original,avg=True)
                #bleu=sentence_bleu([original],output)
                rouge_avg=rouge_avg+r
                c=c+1
                wr.writerow([original,val_sum,output,r])
                
                
        
        encodings=tokenizer("\n\n".join(valid_summary),return_tensors="tf")
        print(encodings.input_ids.shape) # (1,287664가 됨!! 여러 글이 한 덩어리로 합쳐짐.)

        max_length = model.config.n_positions
        stride = 512
        size=encodings.input_ids.shape[1]
        nlls = []
        print("max_length : " + str(max_length))
        print("input id size : " + str(encodings.input_ids.shape[1]))
        prev_end_loc=0

        input()

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

        print("ppl is : " + str(ppl))

        
        wr.writerow(['this is avg rouge of generation: ' + str(rouge_avg/(round(c/10)))])
        wr.writerow(['this is avg perplexity of generation : ' + str(ppl)])        

    history=gpt.fit(x={"input_ids":inp[:,:-1]},y=inp[:,1:],
            
            validation_data=({"input_ids" : val_inp[:,:-1]},val_inp[:,1:]),
            batch_size=consts.BATCH_SIZE,
            callbacks=[tensorboard_callback,checkpoint,])
        #callbacks=[tensorboard_callback,checkpoint,stop_early])
    #print(history)
    print(history.history)

