import tensorflow as tf
import os
import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from tqdm import tqdm,trange
import torch
import csv
import sys
import ctypes as ct
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))
#  Returns a bool indicating if CUDA is currently available.
print(torch.cuda.is_available())
print(torch.cuda.device_count())
torch.cuda.device(0)

def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error Creating directory. ' + directory)

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn",device=0)

RANGE=800

total_source=[]
total_target=[]
f = open('tor_fictions.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
# print(rdr)
for line in rdr:
	#total_target.append(print(line[1]))
        total_source.append(line[1])
        total_target.append(line[2])
        #input()
     

f.close()
print(len(total_target))

#with open("writingPrompts/"+ T +".wp_source", encoding='UTF8') as f:
#    stories = f.readlines()
#    stories = [" ".join(i.split()[0:1000]) for i in stories]
#    temp_stories=[]
#    for story in stories:
#        temp_stories.append(story.replace("<newline>",""))
#    total_source.append(temp_stories)

#total_target=[]

#with open("writingPrompts/"+ T +".wp_target", encoding='UTF8') as f:
#    stories = f.readlines()
#    stories = [" ".join(i.split()[0:1000]) for i in stories]
#    temp_stories=[]
#    for story in stories:
#        temp_stories.append(story.replace("<newline>",""))
#    total_target.append(temp_stories)

seq_length=0

token_len=[]
seq_len=[]
word_len=[]

middle_token_len=[]
middle_seq_len=[]
middle_word_len=[]
mother_token_len=[]
mother_seq_len=[]
mother_word_len=[]

whole_target=total_target[:RANGE]
whole_source=total_source[:RANGE]
#print(len(whole_target[0]))
#print(whole_target[0])
#print(len(whole_target[1]))
#print(whole_target[1])

whole_summary_set=[]
whole_target_set=[]
real_summary_set=[]
middle_summary_set=[]
middle_target_set=[]

c=0
for t in tqdm(whole_target):
        # t?????? ...??? ??? ?????????
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
        t=t.replace("\""," ")
        if seq_length<=0:
            tt=t
        else:
            tt=('.').join(t.split('.')[:seq_length])
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

        ### ?????? 1024 ?????? ????????? ????????? ????????? ?????????.
        seq=[0]
        l=0
        count=0
        for split in tt.split('.'):
            count=count+1
            tokens=tokenizer(split).input_ids
            l=l+len(tokens)
            if l>=1024:
                seq.append(count)
                l=len(tokens)
        seq.append(count)
        ### seq??? ????????? ????????????(????????? ?????????)??? ??????
        
        
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
        if len(middle_target) > 10: # ?????? ??? ??? ???????????? middle target??? 23?????? ?????? ?????? ??????...
            #print("middle target length is over than 10")
            continue

        for mt in middle_target:
            middle_token_len.append(len(tokenizer(mt).input_ids))
            middle_seq_len.append(len(mt.split('.')))
            middle_word_len.append(len(mt.split(' ')))
            try :
                s=summarizer(mt,max_length=100, min_length=50, do_sample=True)
                middle_summary=middle_summary+" "+s[0]["summary_text"]
                mt_summary.append(s[0]["summary_text"])
            except:
                continue
        if len(tokenizer(middle_summary).input_ids)>1023:
            continue
        #print("middle_summary len: " + str(len(tokenizer(middle_summary).input_ids)))
        try :
            mother_plot=summarizer(middle_summary,max_length=100, min_length=10, do_sample=True)[0]["summary_text"]
        
            mother_token_len.append(len(tokenizer(mother_plot).input_ids))
            mother_seq_len.append(len(mother_plot.split('.')))
            mother_word_len.append(len(mother_plot.split(' ')))
        except:
            continue
        # middle_target-> original target??? ?????????, mt_summary->????????? ?????? ????????????, middle_summary-> middle summary ??????, 
        #mother_plot -> ?????? ?????????, ??? ?????? ?????????????????????, ??????????????? ????????? ?????????.
        # middel_summary??? 1????????? decoder input??? ??????.
        # mother_plot??? 1????????? encoder input??? ??????.
        # mt_summary??? ?????? 2????????? encoder input??? ??????.
        # middle_target??? ?????? 2????????? decoder input??? ??????.

        if len(seq)<=10:
            for i in range(11-len(seq)):
                mt_summary.append("<pad>")
                middle_target.append("<pad>")
                # ????????? 10?????? ????????? ????????????.
        whole_summary_set.append(mother_plot)
        whole_target_set.append(middle_summary)
        middle_summary_set.append(mt_summary[:10]) # ?????? 10?????? ?????? ????????? ????????????, ?????? ????????? ?????????.
        middle_target_set.append(middle_target[:10])
        real_summary_set.append(whole_source[c-1])
        # ????????? ???????????? ??????
        # ????????? ??????????????? token??????. 
        # whole summary set, whole target set??? ????????? ????????? ?????? ?????????  ??????. (size, 1024)??? (size, 100)??? ????????????
        # middle summary set??? ??????????????? [["~","~",...], ["~",~~] ] ?????? ????????????. ????????? ????????????.
        # ????????? ????????? ????????? ?????? ??????. ??????????????? ????????? 10???????????? ????????? ?????????,
        # ??????????????? (size, 10, 100)?????? (size,10,1024)??? ????????? ??????.
        # ??????????????? ????????? ?????? 10?????? ????????? ????????? ?????? ??????(10?????? ?????? ?????? ?????????) ?????? ????????? "<pad>" ??????????????? ????????? ??????.


whole_summary_set=tokenizer(whole_summary_set,return_tensors="tf",padding="max_length",max_length=200, truncation=True).input_ids
whole_target_set=tokenizer(whole_target_set,return_tensors="tf",padding="max_length",max_length=1024,truncation=True).input_ids
real_summary_set=tokenizer(real_summary_set,return_tensors="tf",padding="max_length",max_length=200,truncation=True).input_ids
mss=[]
for m in middle_summary_set:
    mss.append(tokenizer(m,return_tensors="tf",padding="max_length",max_length=200, truncation=True).input_ids)
mts=[]
for m in middle_target_set:
    mts.append(tokenizer(m,return_tensors="tf",padding="max_length",max_length=1024, truncation=True).input_ids)

npwhole_summary_set=whole_summary_set.numpy()
npwhole_target_set=whole_target_set.numpy()
npmts=np.array(mts)
npmss=np.array(mss)
npreal_whole_summary_set=real_summary_set.numpy()
print(npwhole_summary_set.shape)
print(npwhole_target_set.shape)
print(npreal_whole_summary_set.shape)
print(npmss.shape)
print(npmts.shape)

file="TOR"
createFolder("npdata/"+file)
print(os.path.isfile("./npdata/"+file+"/mother.npy"))
#if os.path.isfile("./npdata/"+file+"/summary.npy"):
#    past=np.load("./npdata/"+file+"/summary.npy")
#    npsummary=np.concatenate((past,npsummary),axis=0)
np.save("./npdata/"+file +"/mother",npwhole_summary_set)
#if os.path.isfile("./npdata/"+file+"/token_summary.npy"):
#    past=np.load("./npdata/"+file+"/token_summary.npy")
#    nptoken_summary=np.concatenate((past,nptoken_summary),axis=0)
np.save("./npdata/"+file +"/mother_target",npwhole_target_set)
#if os.path.isfile("./npdata/"+file+"/token_target.npy"):
#    past=np.load("./npdata/"+file+"/token_target.npy")
#    nptoken_target=np.concatenate((past,nptoken_target),axis=0)
np.save("./npdata/"+file +"/middle_summary",npmss)
np.save("./npdata/"+file +"/middle_target",npmts)
np.save("./npdata/"+file +"/real_mother",npreal_whole_summary_set)

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
    
analyze(a,"whole seq_len")
analyze(b,"whole token_len")
analyze(c,"whole word_len")
analyze(e,"middle_seq_len")
analyze(d,"middle_token_len")
analyze(f,"middle_word_len")
analyze(i,"mother_seq_len")
analyze(h,"mother_token_len")
analyze(j,"mother_word_len")
