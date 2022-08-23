from re import T
from datasets import load_dataset
import time
import tensorflow as tf
import os
import numpy as np
import torch
tf.config.set_visible_devices([], 'GPU') # CPU로 학습하기.
from summary_maker import *

# mirrored_strategy = tf.distribute.MirroredStrategy()
# gpus = tf.config.experimental.list_logical_devices('GPU') # 멀티 gpu 세팅.
# tf.debugging.set_log_device_placement(True)


TRAIN_FILE="train"
VALID_FILE="valid"
summary_maker(RANGE=0,length=800,file=TRAIN_FILE,is_model_or_given_dataset=False)
summary_maker(RANGE=0,length=800,file=VALID_FILE,is_model_or_given_dataset=False)

token_summary=np.load("./bart/"+TRAIN_FILE +"/token_summary.npy")
token_target=np.load("./bart/"+TRAIN_FILE +"/token_target.npy")
valid_token_summary=np.load("./bart/"+VALID_FILE +"/token_summary.npy")
valid_token_target=np.load("./bart/"+VALID_FILE +"/token_target.npy")

from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
bart = TFAutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
bart_optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5)

bart.compile(
    optimizer=bart_optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy(),
)
model_saver(bart,bart_optimizer)
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/fit/' + current_time
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

bart.fit(x={"input_ids" : token_summary,"decoder_input_ids": token_target[:,:-1]}, 
y=token_target[:,1:],
validation_data=({"input_ids" : valid_token_summary,"decoder_input_ids":valid_token_target[:,:-1]},valid_token_target[:,1:]),
batch_size=8, epochs=3,
callbacks=[tensorboard_callback])
