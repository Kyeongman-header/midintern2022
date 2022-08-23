import tensorflow as tf
import os
import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from tqdm import tqdm,trange

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def model_saver(model,optimizer,file_name='pret5_bart'):
    checkpoint_path = "./MY_checkpoints/train_"+file_name

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
def summary_maker(RANGE=10, length=800,file="train",is_model_or_given_dataset=True):
    total_source=[]
    with open("writingPrompts/"+ file +".wp_source", encoding='UTF8') as f:
        stories = f.readlines()
        stories = [" ".join(i.split()[0:1000]) for i in stories]
        temp_stories=[]
        for story in stories:
            temp_stories.append(story.replace("<newline>",""))
        total_source.append(temp_stories)

    total_target=[]

    with open("writingPrompts/"+ file +".wp_target", encoding='UTF8') as f:
        stories = f.readlines()
        stories = [" ".join(i.split()[0:1000]) for i in stories]
        temp_stories=[]
        for story in stories:
            temp_stories.append(story.replace("<newline>",""))
        total_target.append(temp_stories)
   
    summary=[]

    truncated_target=[]
    if RANGE != 0:
        whole_data=total_target[0][:RANGE]
    else:
        whole_data=total_target[0]
    for t in tqdm(whole_data):
        tt=(' ').join(t.split()[:length])
        #print(len(tt))
        truncated_target.append(tt)
        if is_model_or_given_dataset:
            s=summarizer(tt,max_length=130, min_length=30, do_sample=False)
            summary.append(s[0]["summary_text"])
        
    if is_model_or_given_dataset is False:
        if RANGE != 0:
            summary=total_source[0][:RANGE]
        else:
            summary=total_source[0]
    
    token_summary=tokenizer(summary,return_tensors="tf",padding='longest', truncation=True).input_ids
    print(token_summary.shape)
    token_target=tokenizer(truncated_target,return_tensors="tf",padding='longest', truncation=True).input_ids
    print(token_target.shape)


    npsummary=np.array(summary)
    nptoken_summary=token_summary.numpy()
    nptoken_target=token_target.numpy()
    createFolder("bart/"+file)
    np.save("./bart/"+file +"/summary",npsummary)
    np.save("./bart/"+file +"/token_summary",nptoken_summary)
    np.save("./bart/"+file +"/token_target",nptoken_target)
