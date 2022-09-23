from re import T
from datasets import load_dataset
import time
import tensorflow as tf
import os
import numpy as np
import torch
#tf.config.set_visible_devices([], 'GPU') # CPU로 학습하기.
from summary_maker import *
from models import *
import consts
from tqdm import tqdm, trange

mirrored_strategy = tf.distribute.MirroredStrategy()
gpus = tf.config.experimental.list_logical_devices('GPU') # 멀티 gpu 세팅.
# tf.debugging.set_log_device_placement(True)
RANGE=consts.BATCH_SIZE*2900

#TRAIN_FILE="train"
#VALID_FILE="valid"
TRAIN_FILE="_sm_train"
VALID_FILE="_sm_valid"
#위의 두개는 #bart large cnn 압축 데이터이다.
FURTHER_TRAIN=False
#summary_maker(RANGE=RANGE,length=800,file=TRAIN_FILE,is_model_or_given_dataset=False)
#summary_maker(RANGE=RANGE,length=800,file=VALID_FILE,is_model_or_given_dataset=False)

token_summary=np.load("./npdata/"+TRAIN_FILE +"/token_summary.npy")[:RANGE]
token_target=np.load("./npdata/"+TRAIN_FILE +"/token_target.npy")[:RANGE]
valid_token_summary=np.load("./npdata/"+VALID_FILE +"/token_summary.npy")[:RANGE]
valid_token_target=np.load("./npdata/"+VALID_FILE +"/token_target.npy")[:RANGE]

summary_length=token_summary.shape[1]
target_length=token_target.shape[1]
valid_summary_length=valid_token_summary.shape[1]
valid_target_length=valid_token_target.shape[1]

def make_batch(inp1,inp2,batch_size):
    #inp는 2차원(whole_size, length)이다.
    #목표는 3차원 (whole/batch, batch, length)로 만드는 것이다.
    inp1=np.reshape(inp1,(-1,batch_size,inp1.shape[1]))
    inp2=np.reshape(inp2,(-1,batch_size,inp2.shape[1]))
    return (inp1,inp2)

token_summary,token_target=make_batch(token_summary,token_target,consts.BATCH_SIZE)
val_token_summary,val_token_target=make_batch(valid_token_summary,valid_token_target,consts.BATCH_SIZE)
print(token_summary.shape)
print(token_target.shape)

from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
bart = TFAutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
#bart=TFAutoModelForSeq2SeqLM.from_pretrained("patrickvonplaten/t5-tiny-random")
bart_optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5)
disc_optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5)

# bart.compile(
#     optimizer=bart_optimizer,
#     
#     metrics=tf.metrics.SparseCategoricalAccuracy(),
# )
filename="NOGAN_PRET_SM"
ckpt_manager=model_saver(bart,bart_optimizer,filename=filename)
# GAN / NOGAN
# PRET / NOPRET
# LARGE / TINY
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/third/' + current_time
train_summary_writer = tf.summary.create_file_writer(log_dir)
MAX_VOCAB = 50264
my_disc=My_Disc(vocab_size=MAX_VOCAB,length=target_length,dim=32)
ae_loss=SparseCategorical_Loss(LAMBDA=consts.LAMBDA)
gan_loss=DiscriminatorAndGenerator_Loss(ALPHA=consts.ALPHA)

train_cce_loss=tf.keras.metrics.Mean(name='train_cce_loss')
train_gan_loss=tf.keras.metrics.Mean(name='train_gan_loss')
train_cce_accuracy = tf.keras.metrics.Mean(name='train_cce_accuracy')
train_disc_loss=tf.keras.metrics.Mean(name='train_enc_disc_loss')
val_cce_loss=tf.keras.metrics.Mean(name='val_cce_loss')
val_gan_loss=tf.keras.metrics.Mean(name='val_gan_loss')
val_cce_accuracy = tf.keras.metrics.Mean(name='val_cce_accuracy')
val_disc_loss=tf.keras.metrics.Mean(name='val_enc_disc_loss')

@tf.function
def train_step(token_summary, token_target,val_token_summary,val_token_target): 
    
    with tf.GradientTape(persistent=True) as tape:
        bart_output=bart({"input_ids" : token_summary,"decoder_input_ids": token_target[:-1]}).logits
        sparse_loss=ae_loss.reconstruction_loss(token_target[1:],bart_output)
        #disc_fake=my_disc(bart_output,is_first=False)
        #disc_true=my_disc(token_target[1:],is_first=True)
        #_gan_loss=gan_loss.gen_loss(disc_fake)
        _gan_loss=0
        #cri_loss=gan_loss.critic_loss(disc_true,disc_fake)
        cri_loss=0
        #total_gen_loss=consts.GAMMA*_gan_loss+sparse_loss
        val_bart_output=bart({"input_ids" : val_token_summary,"decoder_input_ids": val_token_target[:-1]}).logits
        val_sparse_loss=ae_loss.reconstruction_loss(val_token_target[1:],val_bart_output)
        val_cri_loss=0
        _val_gan_loss=0

    train_cce_accuracy(ae_loss.reconstruction_accuracy_function(token_target[1:],bart_output))
    val_cce_accuracy(ae_loss.reconstruction_accuracy_function(val_token_target[1:],val_bart_output))
    train_cce_loss(sparse_loss)
    val_cce_accuracy(val_sparse_loss)
    train_gan_loss(_gan_loss)
    val_gan_loss(_val_gan_loss)
    train_disc_loss(cri_loss)
    val_disc_loss(val_cri_loss)

    #gen_gradients = tape.gradient(total_gen_loss, bart.trainable_variables)
    gen_gradients = tape.gradient(sparse_loss, bart.trainable_variables) #NO GAN
    #disc_gradients=tape.gradient(cri_loss,my_disc.trainable_variables)
    bart_optimizer.apply_gradients(zip(gen_gradients, bart.trainable_variables))
    #disc_optimizer.apply_gradients(zip(disc_gradients,my_disc.trainable_variables))

import csv
from nltk.translate.bleu_score import sentence_bleu
tensorboard_count=0
gen_count=0
f= open(filename+'.csv', 'w', newline='')
wr=csv.writer(f)
wr.writerow(['orig_article','summary','ariticle_t','article_g'])
f2= open(filename+'_loss.csv', 'w', newline='')
wr2=csv.writer(f2)
wr2.writerow(['cce_loss','cce_accuracy','disc_loss','gan_loss'])
for epoch in trange(consts.EPOCHS):
    start = time.time()
    train_cce_loss.reset_states()
    train_cce_accuracy.reset_states()
    train_gan_loss.reset_states()
    train_disc_loss.reset_states()
    print("")
    #print(token_summary.shape)
    #print(token_target.shape)
    for batch in trange(token_summary.shape[0]):
    #for (batch, (summary,target)) in enumerate(tqdm(zip(token_summary,token_target))):
        #print(batch)
        #print(summary.shape)
        #print(target.shape)
        summary=token_summary[batch]
        target=token_target[batch]
        val_summary=val_token_summary[batch]
        val_target=val_token_target[batch]
        train_step(summary,target,val_summary,val_target)
        #print(summary.shape)
        #print(target.shape)
        if batch % 100 == 0:
            with train_summary_writer.as_default():
                tf.summary.scalar('cce_loss', train_cce_loss.result(), step=tensorboard_count)
                tf.summary.scalar('cce_accuracy', train_cce_accuracy.result(), step=tensorboard_count)
                tf.summary.scalar('disc_loss', train_disc_loss.result(), step=tensorboard_count)
                tf.summary.scalar('gan_loss', train_gan_loss.result(), step=tensorboard_count)
                tf.summary.scalar('val_cce_loss', val_cce_loss.result(), step=tensorboard_count)
                tf.summary.scalar('val_cce_accuracy', val_cce_accuracy.result(), step=tensorboard_count)
                tf.summary.scalar('val_disc_loss', val_disc_loss.result(), step=tensorboard_count)
                tf.summary.scalar('val_gan_loss', val_gan_loss.result(), step=tensorboard_count)
                wr2.writerow([train_cce_loss.result(), train_cce_accuracy.result(),train_disc_loss.result(),train_gan_loss.result()])
                tensorboard_count=tensorboard_count+1
        if batch % 100 == 0:
            bart_t_output=bart({"input_ids" : summary,"decoder_input_ids": target[:-1]}).logits
            bart_t_output=tf.argmax(bart_t_output,axis=2,output_type=target.dtype)[0]
            bart_g_output=bart.generate(summary)
            origin=tokenizer.decode(target[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            summary= tokenizer.decode(summary[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            g_output=tokenizer.decode(bart_g_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            t_output=tokenizer.decode(bart_t_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            wr.writerow([origin,summary,t_output,g_output])
            with train_summary_writer.as_default():
                tf.summary.scalar('bleu_gen',sentence_bleu([origin],g_output),step=gen_count)
                tf.summary.scalar('bleu_teacher',sentence_bleu([origin],t_output),step=gen_count)
                gen_count=gen_count+1
            #print(f'\rSaving checkpoint for batch {batch} at {ckpt_save_path}',end="")
        #    print(f'Batch {batch} CCE_Loss: {train_cce_loss.result():.4f} | CCE_Accuracy: {train_cce_accuracy.result():.4f}| GAN_Loss: {train_gan_loss.result():.4f} | Disc_Loss: {train_disc_loss.result():.4f}')
 
        #ckpt_save_path = ckpt_manager.save()
        #print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
    print(f'Epoch {epoch + 1} CCE_Loss: {train_cce_loss.result():.4f} | CCE_Accuracy: {train_cce_accuracy.result():.4f}| GAN_Loss: {train_gan_loss.result():.4f} | Disc_Loss: {train_disc_loss.result():.4f}')
    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

ckpt_save_path = ckpt_manager.save()
print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
print(f'Epoch {epoch + 1} CCE_Loss: {train_cce_loss.result():.4f} | CCE_Accuracy: {train_cce_accuracy.result():.4f}| GAN_Loss: {train_gan_loss.result():.4f} | Disc_Loss: {train_disc_loss.result():.4f}')
print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model",
#                                                     save_best_only=True)
# stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)

# if FURTHER_TRAIN :
#     bart = keras.models.load_model('best_model')

# bart.fit(x={"input_ids" : token_summary,"decoder_input_ids": token_target[:,:-1]}, 
# y=token_target[:,1:],
# validation_data=({"input_ids" : valid_token_summary,"decoder_input_ids":valid_token_target[:,:-1]},valid_token_target[:,1:]),
# batch_size=2, epochs=20,
# callbacks=[tensorboard_callback,checkpoint,stop_early])



