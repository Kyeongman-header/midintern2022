import tensorflow as tf
import os
import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from tqdm import tqdm,trange

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

#pipeline("summarization", tokenizer=tokenizer,model=TFAutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn"),device=0)
#pipeline("summarization", model="facebook/bart-large-cnn",device=0)

def model_saver(model,optimizer,filename='pret5_bart'):
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
def summary_maker(START=0,RANGE=10, seq_length=100,file="train",is_model_or_given_dataset=True,device=0):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn",device=device)
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
   
    summary=[]

    truncated_target=[]
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
    for t in tqdm(whole_data):
        if seq_length<=0:
            tt=t
        else:
            tt=('.').join(t.split('.')[:seq_length])
        if len(tt)==0:
            continue
        #print('len: '+str(len(tt)))
        len(tokenizer(tt).input_ids)
        if is_model_or_given_dataset:
            if len(tokenizer(tt).input_ids)<1024:
                try :
                    s=summarizer(tt,max_length=130, min_length=50, do_sample=True)
                    truncated_target.append(tt)
                    summary.append(s[0]["summary_text"])
                    if (len(tt.split(' '))>max_target):
                        max_target=len(tt.split(' '))
                    #print(max_target)
                    if (len(s[0]["summary_text"].split(' '))>max_sum):
                        max_sum=len(s[0]["summary_text"].split(' '))
                    #print(max_sum)
                except:
                    continue
            else:
                continue
        else:
            truncated_target.append(tt)
        
    if is_model_or_given_dataset is False:
        if RANGE != 0:
            summary=total_source[0][START:START+RANGE]
        else:
            summary=total_source[0]
    
    print(len(summary))
    print(len(truncated_target))
    #token_summary=tokenizer(summary,padding="max_length",max_length=150, truncation=True).input_ids
    token_summary=tokenizer(summary,return_tensors="tf",padding="max_length",max_length=150, truncation=True).input_ids
    print(token_summary.shape)
    #token_target=tokenizer(truncated_target,padding="max_length", max_length=max_target+100,truncation=True).input_ids
    token_target=tokenizer(truncated_target,return_tensors="tf",padding="max_length", max_length=1024,truncation=True).input_ids
    print(token_target.shape)

    npsummary=np.array(summary)
    #nptarget=np.array(truncated_target)
    nptoken_summary=token_summary.numpy()
    nptoken_target=token_target.numpy()
    
    createFolder("npdata/"+file)
    print(os.path.isfile("./npdata/"+file+"/summary.npy"))
    if os.path.isfile("./npdata/"+file+"/summary.npy"):
        past=np.load("./npdata/"+file+"/summary.npy")
        npsummary=np.concatenate((past,npsummary),axis=0)
    np.save("./npdata/"+file +"/summary",npsummary)
    if os.path.isfile("./npdata/"+file+"/token_summary.npy"):
        past=np.load("./npdata/"+file+"/token_summary.npy")
        nptoken_summary=np.concatenate((past,nptoken_summary),axis=0)
    np.save("./npdata/"+file +"/token_summary",nptoken_summary)
    if os.path.isfile("./npdata/"+file+"/token_target.npy"):
        past=np.load("./npdata/"+file+"/token_target.npy")
        nptoken_target=np.concatenate((past,nptoken_target),axis=0)
    np.save("./npdata/"+file +"/token_target",nptoken_target)
