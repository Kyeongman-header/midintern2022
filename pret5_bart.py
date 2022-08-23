from re import T
from datasets import load_dataset
import time
import tensorflow as tf
import os
import numpy as np

# mirrored_strategy = tf.distribute.MirroredStrategy()
# gpus = tf.config.experimental.list_logical_devices('GPU') # 멀티 gpu 세팅.
total_source=[]

with open("writingPrompts/"+ "test.wp_source", encoding='UTF8') as f:
    stories = f.readlines()
    stories = [" ".join(i.split()[0:1000]) for i in stories]
    temp_stories=[]
    for story in stories:
        temp_stories.append(story.replace("<newline>",""))
    total_source.append(temp_stories)

total_target=[]

with open("writingPrompts/"+ "test.wp_target", encoding='UTF8') as f:
    stories = f.readlines()
    stories = [" ".join(i.split()[0:1000]) for i in stories]
    temp_stories=[]
    for story in stories:
        temp_stories.append(story.replace("<newline>",""))
    total_target.append(temp_stories)

# print(len(total_source[0]))
# print(len(total_target[0]))

from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

bart = TFAutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary=[]

truncated_target=[]
for t in total_target[0][:10]:
    tt=(' ').join(t.split()[:800])
    print(len(tt))
    truncated_target.append(tt)
    
    s=summarizer(tt,max_length=130, min_length=30, do_sample=False)
    summary.append(s[0]["summary_text"])

token_summary=tokenizer(summary,return_tensors="tf",padding='longest', truncation=True).input_ids
token_target=tokenizer(truncated_target,return_tensors="tf",padding='longest', truncation=True).input_ids
print(token_summary.shape)
print(token_target.shape)



bart.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy(),
)


bart.fit(x=token_summary, y=token_target,batch_size=8, epochs=3)