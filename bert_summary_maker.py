import tensorflow as tf
from summarizer import Summarizer
import os
import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from tqdm import tqdm,trange

tokenizer = AutoTokenizer.from_pretrained("gpt2")

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
def bert_summary_maker(START=0,RANGE=10, seq_length=100,file="train",is_model_or_given_dataset=True,device=0):
    summarizer=Summarizer()
    #summarizer = pipeline("summarization", model="facebook/bart-large-cnn",device=device)
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
    for t in tqdm(whole_data):
        # t에서 ;이랑 .. 같은 애들 . 으로 바꿔야 겠다
        t.replace(".....",".")
        t.replace("....",".")
        t.replace("...",".")
        t.replace("..",".")
        
        if seq_length<=0:
            tt=t
        else:
            tt=('.').join(t.split('.')[:seq_length])
        if len(tt)==0:
            continue
        #print('len: '+str(len(tt)))
        #print(len(tokenizer(tt).input_ids))
        if is_model_or_given_dataset:
            if len(tokenizer(tt).input_ids)<1024:
                try :
                    
                    s=summarizer(tt,max_length=200, min_length=50)
                    
                    summary_prefix_target.append(s + " : " + tt) # result 자체가 문자열임
                    
                except:
                    continue
            else:
                continue
        
        
    
    print("summary prefix target length :" + str(len(summary_prefix_target)))
    token_summary_prefix_target=tokenizer(summary_prefix_target,return_tensors="tf",padding="max_length",max_length=1024, truncation=True).input_ids
    print("tokn summary prefix target shape : ")
    print(token_summary_prefix_target.shape)
    

    npsummary_prefix_target=np.array(summary_prefix_target)
    #nptarget=np.array(truncated_target)
    nptoken_summary_prefix_target=token_summary_prefix_target.numpy()
    
    createFolder("npdata/"+file)

    print("is this npdata exists already? => ")
    print(os.path.isfile("./npdata/"+file+"/summary_prefix_target.npy"))
    if os.path.isfile("./npdata/"+file+"/summary_prefix_target.npy"):
        past=np.load("./npdata/"+file+"/summary_prefix_target.npy")
        npsummary_prefix_target=np.concatenate((past,npsummary_prefix_target),axis=0)
    np.save("./npdata/"+file +"/summary_prefix_target",npsummary_prefix_target)
    if os.path.isfile("./npdata/"+file+"/token_summary_prefix_target.npy"):
        past=np.load("./npdata/"+file+"/token_summary_prefix_target.npy")
        nptoken_summary_prefix_target=np.concatenate((past,nptoken_summary_prefix_target),axis=0)
    np.save("./npdata/"+file +"/token_summary_prefix_target",nptoken_summary_prefix_target)
    
