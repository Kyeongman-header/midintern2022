# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from summarizer import Summarizer
from tqdm import tqdm,trange
import torch
import csv
import sys
import ctypes as ct
import math
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))

#  Returns a bool indicating if CUDA is currently available.
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# torch.cuda.device(0)

def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error Creating directory. ' + directory)


# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token=tokenizer.eos_token
tokenizer.pad_token_id=tokenizer.eos_token_id

def hier_summary_maker(START=0,RANGE=10,report=False, is_abs_or_ext=False, seq_length=100,file="train",is_model_or_given_dataset=True,device=0):
    if is_abs_or_ext :
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn",device=device)
    else :
        summarizer = Summarizer()

    RANGE=800

    #total_source=[]
    
    total_target=[]
    f = open('reedsy_wp.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    # print(rdr)
    for line in rdr:
        #total_target.append(print(line[1]))
        #total_source.append(line[1])
        total_target.append(line[1])
            #input()
        

    #f.close()
    #print(len(total_target))
    
    
    #T=file
    #if "train" in file:
    #    T="train"
    #elif "valid" in file:
    #    T="valid"
    
    """
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
    """

    #if RANGE != 0:
    #    whole_data=total_target[0][START:START+RANGE]
    #else:
    #    whole_data=total_target[0]
    if RANGE !=0:
        whole_data=total_target[START:START+RANGE]
    else:
        whole_data=total_target

    token_len=[]
    seq_len=[]
    word_len=[]

    middle_token_len=[]
    middle_seq_len=[]
    middle_word_len=[]
    mother_token_len=[]
    mother_seq_len=[]
    mother_word_len=[]

    #whole_target=total_target[:RANGE]
    #whole_source=total_source[:RANGE]

    #print(len(whole_target[0]))
    #print(whole_target[0])
    #print(len(whole_target[1]))
    #print(whole_target[1])
    middle_summary_prefix_target=[]
    final_summary_prefix_target=[]
    whole_summary_set=[]
    whole_target_set=[]
    real_summary_set=[]
    middle_summary_set=[]
    middle_target_set=[]

    c=0
    seq_length=0
    for t in tqdm(whole_data):
            # t에서 ...은 다 없애야
            t=t.split(' ')
            
            
            c=c+1
            continue_flag=False
            t=t.replace("......",".")
            t=t.replace(".....",".")
            t=t.replace("....",".")
            t=t.replace("...",".")
            t=t.replace("..",".")
            t=t.replace(";",";.")
            t=t.replace("!","!.")
            t=t.replace("?","?.")
            t=t.replace("\""," ")
            t=t.replace("\'"," ")
            
            if seq_length<=0:
               tt=t
            else:
               tt=('.').join(t.split('.')[:seq_length])
            # seq length로 조절하는 것은 무의미하고,
            # 그 전에 words 개수로 '자른다'

            # tt=(' ').join(t.split(' '))
        
            if len(tt)==0:
                continue
            
            #print(tt)
            #print('len: '+str(len(tt)))

            tl=len(tokenizer(tt).input_ids)
            sl=len(tt.split('.'))
            wl=len(tt.split(' '))
            seq_len.append(sl)
            token_len.append(tl)
            word_len.append(wl)
            #print("---$$---")
            #print("sl: " + str(sl))
            #print("wl: "+str(wl))
            #print("tl: "+str(tl))

            ### 최대 1024 토큰 이하가 되도록 덩어리 나누기.
            seq=[0]
            l=0
            count=0
            Words=int(round(tl/5))

            for split in tt.split('.'):
                count=count+1
                if len(seq)>=6:
                    # 5 덩어리에서 끊는다.
                    continue
                tokens=tokenizer(split).input_ids
                l=l+len(tokens)
                if l>=Words:
                    seq.append(count)
                    l=len(tokens)
                
            
            seq.append(count)
            ### seq엔 나눠진 덩이리들(문장의 인덱스)가 있음
            if len(seq)>6:#6덩어리가 있는 상태.
                dist=seq[-1]-seq[-2]+1
                dist=math.ceil(dist/5)

                while seq[5]<sl:
                    seq[1]+=dist
                    seq[2]+=dist*2
                    seq[3]+=dist*3
                    seq[4]+=dist*4
                    seq[5]+=dist*5 # 마지막 seq[5]가 실제 len을 넘어도 큰 상관은 없다.

            # else: # 이론상 5개가 무조건 되야 한다. 
            #     dist=seq[-1]-seq[-2]+1
            #     dist=math.ceil(dist/5)
            #     while seq[-1]<sl:
            #         for s in range(len(seq)):
            #             if s>0:
            #                 seq[s]+=dist*s

            print("seq 배열 : ")
            print(seq)


            
            middle_target=[]
            for i in range(len(seq)):
                if i>0:
                    mt=('.').join(tt.split('.')[seq[i-1]:seq[i]])
                    if(len(tokenizer(mt).input_ids)>1023):
                        continue_flag=True

                    #print("each middel target token length : " + str(len(tokenizer(mt).input_ids)))

                    middle_target.append(mt)
            
            if continue_flag:
                continue
            #print("middle target length : "+str(len(middle_target)))
            
            mt_summary=[]
            middle_summary=""
            if len(middle_target) > 5: # 구조상 나오면 안되지만, 혹시 모르니까.
                print("middle target length is over than 5")
                continue

            for mt in middle_target:
                middle_token_len.append(len(tokenizer(mt).input_ids))
                middle_seq_len.append(len(mt.split('.')))
                middle_word_len.append(len(mt.split(' ')))
                try :
                    s=summarizer(mt,max_length=200, min_length=150)
                    if is_abs_or_ext :
                        s=s[0]["summary_text"]
                    
                    middle_summary=middle_summary+". "+s
                    #mt_summary.append(s)
                    
                except:
                    continue
            if len(tokenizer(middle_summary).input_ids)>1023 or len(tokenizer(middle_summary).input_ids)<300:
                print("too short middle summary.")
                continue
            
            #print("middle_summary len: " + str(len(tokenizer(middle_summary).input_ids)))
            try :
                
                mother_plot=summarizer(middle_summary,max_length=200, min_length=50)

                if is_abs_or_ext :
                    mother_plot=mother_plot[0]["summary_text"]
                
                middle_summary_prefix_target.append("This is the most abstract plot. " + "The summary text is : " + mother_plot + " and the original text is : " + middle_summary) 
                # GPT 학습을 위한 PREFIX 데이터셋을 생성한다.

                mother_token_len.append(len(tokenizer(mother_plot).input_ids))
                mother_seq_len.append(len(mother_plot.split('.')))
                mother_word_len.append(len(mother_plot.split(' ')))
            except:
                continue
            
            for i in len(middle_target):
                final_summary_prefix_target.append("This is the second abstract plot. " + "The mother plot is : " + mother_plot + " and the page is : " + str(num+i) + " and the summary text is : " + mt_summary[i] + " and the original text is : " + middle_target[i])
            # GPT 학습을 위한 prefix 데이터셋을 생성한다.

            # middle_target-> original target이 분할됨, mt_summary->분할된 미들 서머리들, middle_summary-> middle summary 통째, 
            # mother_plot -> 최종 서머리, 가 전부 준비되었으므로, 데이터셋을 통째로 만든다.
            # middel_summary가 1차적인 decoder input이 된다.
            # mother_plot이 1차적인 encoder input이 된다.
            # mt_summary가 각각 2차적인 encoder input이 된다.
            # middle_target이 각각 2차적인 decoder input이 된다.

            #if len(seq)<=10:
            #    for i in range(11-len(seq)):
            #        mt_summary.append("<pad>")
            #        middle_target.append("<pad>")
                    # 강제로 10개가 되도록 맞춰준다.
            
            whole_summary_set.append(mother_plot)
            #whole_target_set.append(middle_summary)
            #middle_summary_set.append(mt_summary[:10]) # 아마 10개를 넘는 녀석은 없겠지만, 혹시 모르니 해준다.
            #middle_target_set.append(middle_target[:10])
            #real_summary_set.append(whole_source[c-1])
            # 차원을 통일해야 한다
            # 어차피 저장하는건 token이다. 
            # whole summary set, whole target set은 예전에 하듯이 토큰 만들면  된다. (size, 1024)와 (size, 100)이 나올거다
            # middle summary set은 기본적으로 [["~","~",...], ["~",~~] ] 이런 구조이다. 타겟도 마찬가지.
            # 얘네도 차원을 통일해 줘야 한다. 내부적으로 차원이 10개씩으로 반드시 맞추고,
            # 최종적으로 (size, 10, 100)이랑 (size,10,1024)가 나와야 한다.
            # 당연하지만 각각의 셋은 10개를 못넘는 애들도 있을 거다(10개를 넘는 애는 없을듯) 이런 애들은 "<pad>" 배열이라도 넣어야 한다.



    middle_summary_prefix_target=tokenizer(middle_summary_prefix_target,return_tensors="tf",padding="max_length",max_length=1024,truncation=True).input_ids
    final_summary_prefix_target=tokenizer(final_summary_prefix_target,return_tensors="tf",padding="max_length",max_length=1024,truncation=True).input_ids
    npmiddle_summary_prefix_target=middle_summary_prefix_target.numpy()
    npfinal_summary_prefix_target=final_summary_prefix_target.numpy()

    print(npmiddle_summary_prefix_target.shape)
    print(npfinal_summary_prefix_target.shape)

    whole_summary_set=tokenizer(whole_summary_set,return_tensors="tf",padding="max_length",max_length=200, truncation=True).input_ids
    #whole_target_set=tokenizer(whole_target_set,return_tensors="tf",padding="max_length",max_length=1024,truncation=True).input_ids
    #real_summary_set=tokenizer(real_summary_set,return_tensors="tf",padding="max_length",max_length=200,truncation=True).input_ids
    #mss=[]
    #for m in middle_summary_set:
    #    mss.append(tokenizer(m,return_tensors="tf",padding="max_length",max_length=200, truncation=True).input_ids)
    #mts=[]
    #for m in middle_target_set:
    #    mts.append(tokenizer(m,return_tensors="tf",padding="max_length",max_length=1024, truncation=True).input_ids)

    npwhole_summary_set=whole_summary_set.numpy()
    #npwhole_target_set=whole_target_set.numpy()
    #npmts=np.array(mts)
    #npmss=np.array(mss)
    #npreal_whole_summary_set=real_summary_set.numpy()
    print(npwhole_summary_set.shape)
    #print(npwhole_target_set.shape)
    #print(npreal_whole_summary_set.shape)
    #print(npmss.shape)
    #print(npmts.shape)

    #file="TOR"

    createFolder("npdata/"+file)
    print(os.path.isfile("./npdata/"+file+"/mother.npy"))
    #if os.path.isfile("./npdata/"+file+"/summary.npy"):
    #    past=np.load("./npdata/"+file+"/summary.npy")
    #    npsummary=np.concatenate((past,npsummary),axis=0)
    np.save("./npdata/"+file +"/mother",npwhole_summary_set)
    #if os.path.isfile("./npdata/"+file+"/token_summary.npy"):
    #    past=np.load("./npdata/"+file+"/token_summary.npy")
    #    nptoken_summary=np.concatenate((past,nptoken_summary),axis=0)
    np.save("./npdata/"+file +"/mother_target",npmiddle_summary_prefix_target)
    #if os.path.isfile("./npdata/"+file+"/token_target.npy"):
    #    past=np.load("./npdata/"+file+"/token_target.npy")
    #    nptoken_target=np.concatenate((past,nptoken_target),axis=0)
    np.save("./npdata/"+file +"/middle_summary",npfinal_summary_prefix_target)
    #np.save("./npdata/"+file +"/middle_target",npmts)
    #np.save("./npdata/"+file +"/real_mother",npreal_whole_summary_set)

    a=np.array(seq_len)
    b=np.array(token_len)
    c=np.array(word_len)
    d=np.array(middle_token_len)
    e=np.array(middle_seq_len)
    f=np.array(middle_word_len)
    h=np.array(mother_token_len)
    i=np.array(mother_seq_len)
    j=np.array(mother_word_len)
    def analyze(a,name):
        print("-----$$------")
        print("data name : " + name)
        print("mean : " + str(np.mean(a)))
        print("variance : " +str(np.var(a)))
        print("std : " + str(np.std(a)))
        print("median : " + str(np.median(a)))
        print("bincount : " + str(np.bincount(a).argmax()))
        print("min : " + str(np.min(a)))
        print("1 quantile : " + str(np.quantile(a,0.25)))
        print("2 quantile : " + str(np.quantile(a,0.5)))
        print("3 quantile : " + str(np.quantile(a,0.75)))
        print("max : " + str(np.max(a)))
    if report :
        analyze(a,"whole seq_len")
        analyze(b,"whole token_len")
        analyze(c,"whole word_len")
        analyze(e,"middle_seq_len")
        analyze(d,"middle_token_len")
        analyze(f,"middle_word_len")
        analyze(i,"mother_seq_len")
        analyze(h,"mother_token_len")
        analyze(j,"mother_word_len")

hier_summary_maker(0,0,report=True,is_abs_or_ext=False,file="sample_bookrik")
