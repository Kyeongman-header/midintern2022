import tensorflow as tf
from summarizer import Summarizer
import os
import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from tqdm import tqdm,trange

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token=tokenizer.eos_token
"""tokenizer.add_special_tokens({
            "pad_token": "<pad>",
              })
"""
tokenizer.pad_token_id=tokenizer.eos_token_id
 # gpt는 기본적으로 pad 가 없고 eos와 bos 만 존재한다. 그리고 이 두 토큰을 tokenizer가 자동으로 붙여주지도 않는다.

#pipeline("summarization", tokenizer=tokenizer,model=TFAutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn"),device=0)
#pipeline("summarization", model="facebook/bart-large-cnn",device=0)

def model_saver(model,optimizer,filename='pretbert_gpt'):
    checkpoint_path = "./MY_checkpoints/train_"+filename

    ckpt = tf.train.Checkpoint(bart=model,optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=None)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    else :
        print('This is New train!!(no checkpoint)')
    
    return ckpt_manager

def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error Creating directory. ' + directory)
def bert_summary_maker(START=0,RANGE=10,report=False, is_abs_or_ext=False, seq_length=100,file="train",is_model_or_given_dataset=True,device=0):
    if is_abs_or_ext :
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn",device=device)
    else :
        summarizer = Summarizer()
    #summarizer=pipeline("summarization", tokenizer=tokenizer,model=TFAutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn"),device=0)
    total_source=[]
    T=file
    if "train" in file:
        T="train"
    elif "valid" in file:
        T="valid"
    

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
   
    summary_prefix_target=[]
    summary=[]
    truncated_target=[]
    #truncated_target=[]
    whole_data=[]
    #print(START)
    #print(RANGE)
    #print(len(total_target[0]))
    #print(len(total_target[0][START:RANGE]))
    if RANGE != 0:
        whole_data=total_target[0][START:START+RANGE]
    else:
        whole_data=total_target[0]
    max_sum=0
    max_target=0
    print("whole data: " + str(len(whole_data)))
    seq_arr=[]

    for t in tqdm(whole_data):
        # t에서 ;이랑 .. 같은 애들 . 으로 바꿔야 겠다
        t.replace(".....",".")
        t.replace("....",".")
        t.replace("...",".")
        t.replace("..",".")
        t=t.replace(";",";.")
        t=t.replace("!","!.")
        t=t.replace("?","?.")
        t=t.replace("\""," ")
        t=t.replace("\""," ") 

        seq_arr.append(len(t.split('.')))
        # 전체 문장의 분포를 알기 위해.
        if seq_length<=0:
            tt=t
        else:
            tt=('.').join(t.split('.')[:seq_length])
        if len(tt)==0:
            continue
        #print('len: '+str(len(tt)))
        #print(len(tokenizer(tt).input_ids))
        _len=len(tokenizer(tt).input_ids)
        if is_model_or_given_dataset:
            if _len<1024:
                try :
                    
                    s=summarizer(tt,max_length=200, min_length=50)
                    if is_abs_or_ext :
                        s=s[0]["summary_text"]
                    truncated_target.append(tt)
                    summary.append(s)
                    summary_prefix_target.append("The summary is : " + s + " And the original text is : " + tt + tokenizer.eos_token) # result 자체가 문자열임
                    # 이렇게 자연어로 된 prefix 관련 제시를 해야 성능이 좋댄다.
                except:
                    continue
            else:
                continue
        
            
    print("pad token id : " + str(tokenizer.pad_token_id))
    print("bos token id : " + str(tokenizer.bos_token_id))
    print("eos token id : " + str(tokenizer.eos_token_id))
    print("summary prefix target length :" + str(len(summary_prefix_target)))
    print("summary length : " + str(len(summary)))
    print("truncated target length : " + str(len(truncated_target)))
    token_summary_prefix_target=tokenizer(summary_prefix_target,return_tensors="tf",padding="max_length",max_length=1024, truncation=True).input_ids
    print("tokn summary prefix target shape : ")
    print(token_summary_prefix_target.shape)
    token_target=tokenizer(truncated_target,return_tensors="tf",padding="max_length", max_length=1024,truncation=True).input_ids
    print("tokn target shape : ")
    print(token_target.shape)

    npsummary=np.array(summary)
    #nptarget=np.array(truncated_target)
    nptoken_summary_prefix_target=token_summary_prefix_target.numpy()
    nptoken_target=token_target.numpy()
    
    createFolder("npdata/"+file)

    print("is this npdata exists already? => ")
    print(os.path.isfile("./npdata/"+file+"/summary.npy"))
    
    if os.path.isfile("./npdata/"+file+"/summary.npy"):
        past=np.load("./npdata/"+file+"/summary.npy")
        npsummary=np.concatenate((past,npsummary),axis=0)
    np.save("./npdata/"+file +"/summary",npsummary)
    
    if os.path.isfile("./npdata/"+file+"/token_summary_prefix_target.npy"):
        past=np.load("./npdata/"+file+"/token_summary_prefix_target.npy")
        nptoken_summary_prefix_target=np.concatenate((past,nptoken_summary_prefix_target),axis=0)
    np.save("./npdata/"+file +"/token_summary_prefix_target",nptoken_summary_prefix_target)
    
    if os.path.isfile("./npdata/"+file+"/token_target.npy"):
        past=np.load("./npdata/"+file+"/token_target.npy")
        nptoken_target=np.concatenate((past,nptoken_target),axis=0)
    np.save("./npdata/"+file +"/token_target",nptoken_target)

    if report:
        a=np.array(seq_arr)
    
        print("-----$$------")
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
