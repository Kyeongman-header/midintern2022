from re import T
from datasets import load_dataset
import time
import tensorflow as tf
import os
import numpy as np
from models import *
from others import *
import consts

#tf.debugging.set_log_device_placement(True)
#tf.config.set_visible_devices([], 'GPU') # CPU로 학습하기.
# mirrored_strategy = tf.distribute.MirroredStrategy()
# gpus = tf.config.experimental.list_logical_devices('GPU') # 멀티 gpu 세팅.

from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM,TFAutoModel

t5_tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/t5-tiny-random")
t5_1 = TFAutoModelForSeq2SeqLM.from_pretrained("patrickvonplaten/t5-tiny-random")
t5_2 = TFAutoModelForSeq2SeqLM.from_pretrained("patrickvonplaten/t5-tiny-random")
bert_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-distilbert")
bert_1=TFAutoModel.from_pretrained("hf-internal-testing/tiny-random-bert")
bert_2=TFAutoModel.from_pretrained("hf-internal-testing/tiny-random-bert")

dataset = load_dataset("cnn_dailymail", '3.0.0') # cnn dailymail로 했지만 다른 데이터도 같은 데이터 형식(dictionary, "long" : ~, "short" : ~)
# print(dataset["train"][100]['highlights'])
# print(dataset["train"][100]['article'])

MAX_VOCAB = 32128
#len(t5_tokenizer.get_vocab()) # 뭐지 ㅅㅂ 이거랑 왜달라
print('VOCAB_SIZE :',  MAX_VOCAB)

def tokenize_function(examples):
    return {'input_ids' : t5_tokenizer(examples["article"],max_length=consts.LONG_MAX, padding='max_length', truncation=True)['input_ids'],'decoder_input_ids' : t5_tokenizer(examples["highlights"], max_length=consts.SHORT_MAX, padding='max_length', truncation=True)['input_ids']}

tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(50000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
from transformers import DefaultDataCollator
data_collator = DefaultDataCollator(return_tensors="tf")
tf_train_dataset = small_train_dataset.to_tf_dataset(
    columns=['input_ids', 'decoder_input_ids'],
    label_cols=['decoder_input_ids'],
    shuffle=True,
    collate_fn=data_collator,
    drop_remainder=True, # 무조건 batch size로 고정(마지막에 남는 애들은 버림)
    batch_size=consts.BATCH_SIZE,
) # train dataset을 batch 사이즈별로 제공해줌.
tf_validation_dataset = small_eval_dataset.to_tf_dataset(
    columns=['input_ids', 'decoder_input_ids'],
    label_cols=['decoder_input_ids'],
    shuffle=False,
    collate_fn=data_collator,
    drop_remainder=True,
    batch_size=consts.BATCH_SIZE,
)

my_decoder=My_Decoder_tiny_T5(model=t5_1,dim=consts.DIM,vocab_size=MAX_VOCAB,length=consts.LONG_MAX,inp_length=consts.SHORT_MAX)
my_encoder=My_Encoder_tiny_T5(model=t5_2,dim=consts.DIM,vocab_size=MAX_VOCAB,length=consts.SHORT_MAX,inp_length=consts.LONG_MAX)
my_enc_disc=My_Disc(model=bert_1,vocab_size=MAX_VOCAB,length=consts.SHORT_MAX,dim=consts.DIS_DIM)
my_dec_disc=My_Disc(model=bert_2,vocab_size=MAX_VOCAB,length=consts.LONG_MAX,dim=consts.DIS_DIM)
ae_loss=SparseCategorical_Loss(LAMBDA=consts.LAMBDA)
gan_loss=DiscriminatorAndGenerator_Loss(ALPHA=consts.ALPHA)
enc_optimizer = tf.keras.optimizers.Adam(consts.LEARNING_RATE, beta_1=0.9, beta_2=0.98,epsilon=1e-9) # summary의 teacher forcing을 위한 optimizer
dec_optimizer = tf.keras.optimizers.Adam(consts.LEARNING_RATE, beta_1=0.9,beta_2=0.98,epsilon=1e-9) # AE 구조의 recon loss를 위한 optimize
enc_disc_optimizer=tf.keras.optimizers.Adam(consts.LEARNING_RATE, beta_1=0.9, beta_2=0.98,epsilon=1e-9)
dec_disc_optimizer=tf.keras.optimizers.Adam(consts.LEARNING_RATE, beta_1=0.9, beta_2=0.98,epsilon=1e-9)

import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
# test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# test_summary_writer = tf.summary.create_file_writer(test_log_dir)

train_enc_gen_loss=tf.keras.metrics.Mean(name='train_enc_gen_loss')
train_enc_gen_accuracy = tf.keras.metrics.Mean(name='train_enc_gen_accuracy')
train_enc_disc_loss=tf.keras.metrics.Mean(name='train_enc_disc_loss')
train_dec_gen_loss=tf.keras.metrics.Mean(name='train_dec_gen_loss')
train_dec_gen_accuracy=tf.keras.metrics.Mean(name='train_dec_gen_accuracy')
train_dec_disc_loss=tf.keras.metrics.Mean(name='train_dec_disc_loss')
train_enc_cycled_loss=tf.keras.metrics.Mean(name='train_enc_cycled_loss')
train_enc_cycled_accuracy = tf.keras.metrics.Mean(name='train_enc_cycled_accuracy')

ckpt_manager=model_saver(my_encoder,my_decoder,my_enc_disc,my_dec_disc,enc_optimizer,dec_optimizer,enc_disc_optimizer,dec_disc_optimizer,file_name=consts.FILE_NAME)




# ForTest(my_encoder,my_enc_disc,my_decoder,my_dec_disc,ae_loss,gan_loss,consts.BATCH_SIZE,consts.SHORT_MAX,consts.LONG_MAX,t5_tokenizer)


@tf.function
def train_step(inp, sum,teacher): # 기본적으로, summary pair가 존재하면 teacher=true, pair가 존재하지 않고 literature domain의 summary만 존재할 경우, false
    start_tokens,end_token,pad_inp,pad_sum,loss_inp,loss_sum=padder(inp,sum,t5_tokenizer)
    
    with tf.GradientTape(persistent=True) as tape:
        
        enc_output=my_encoder([pad_inp,pad_sum],starts=start_tokens,end=end_token,teacher=teacher,is_first=True)
        t2=my_enc_disc(enc_output,is_first=False) # fake output
        t3=my_enc_disc(loss_sum,is_first=True) # real output

        if teacher is True:
            ae_enc_loss=ae_loss.summary_loss(loss_sum,enc_output)
        else:
            ae_enc_loss=0
        
        gan_enc_gen_loss=gan_loss.gen_loss(t2)
        gan_enc_cri_loss=gan_loss.critic_loss(t3,t2)

        dec_output=my_decoder([enc_output,pad_inp],starts=start_tokens,end=end_token,teacher=True,is_first=False) #기본적으로 teacher forcing
        t5=my_dec_disc(dec_output,is_first=False) #fake output
        t6=my_dec_disc(loss_inp,is_first=True) # real output

        cycled_ae_dec_loss=ae_loss.reconstruction_loss(loss_inp,dec_output)
        # cycled_gan_dec_gen_loss=gan_loss.gen_loss(t5)
        # cycled_gan_dec_cri_loss=gan_loss.critic_loss(t6,t5)
        
        if teacher is True: # teacher가 true라면, decoder도 summary pair를 이용해 cycle 학습을 해준다.
            dec_output=my_decoder([pad_sum,pad_inp],starts=start_tokens,end=end_token,teacher=True,is_first=True) #기본적으로 teacher forcing.
            t7=my_dec_disc(dec_output,is_first=False)
            t8=my_dec_disc(loss_inp,is_first=True)
            ae_dec_loss=ae_loss.reconstruction_loss(loss_inp,dec_output)
            gan_dec_gen_loss=gan_loss.gen_loss(t7)
            gan_dec_cri_loss=gan_loss.critic_loss(t8,t7)
            cycle_enc_output=my_encoder([dec_output,pad_sum],starts=start_tokens,end=end_token,teacher=teacher,is_first=False)
            # t9=my_enc_disc(cycle_enc_output,is_first=False) # fake output
            # t10=my_enc_disc(loss_sum,is_first=True) # real output
            cycled_ae_enc_loss=ae_loss.summary_loss(loss_sum,cycle_enc_output)
            # cycled_gan_enc_gen_loss=gan_loss.gen_loss(t9)
            # cycled_gan_enc_cri_loss=gan_loss.critic_loss(t10,t9)
        else :
            ae_dec_loss=0
            gan_dec_gen_loss=0
            gan_dec_cri_loss=0
            cycled_ae_enc_loss=0

        total_enc_gen_loss=ae_enc_loss+gan_enc_gen_loss+consts.GAMMA*(cycled_ae_dec_loss+cycled_ae_enc_loss)
        total_enc_disc_loss=gan_enc_cri_loss
        total_dec_gen_loss=ae_dec_loss+gan_dec_gen_loss+consts.GAMMA*(cycled_ae_enc_loss+cycled_ae_dec_loss) # 현재 구조는 no teacher 일때 decoder는 cycled dec loss만을 이용할 수 있다(사실 원래 그랬긴 하지)
        total_dec_disc_loss=gan_dec_cri_loss
        
    ae_enc_acc=ae_loss.summary_accuracy_function(loss_sum,enc_output)
    ae_dec_acc=ae_loss.reconstruction_accuracy_function(loss_inp,dec_output)

    if teacher is True:
        enc_cycled_acc=ae_loss.summary_accuracy_function(loss_sum,cycle_enc_output)

    enc_gradients = tape.gradient(total_enc_gen_loss, my_encoder.trainable_variables)
    enc_disc_gradients=tape.gradient(total_enc_disc_loss,my_enc_disc.trainable_variables)
    dec_gradients=tape.gradient(total_dec_gen_loss,my_decoder.trainable_variables)
    dec_disc_gradients=tape.gradient(total_dec_disc_loss,my_dec_disc.trainable_variables)

    enc_optimizer.apply_gradients(zip(enc_gradients, my_encoder.trainable_variables))
    enc_disc_optimizer.apply_gradients(zip(enc_disc_gradients,my_enc_disc.trainable_variables))
    dec_optimizer.apply_gradients(zip(dec_gradients, my_decoder.trainable_variables))
    dec_disc_optimizer.apply_gradients(zip(dec_disc_gradients,my_dec_disc.trainable_variables))

    train_enc_gen_loss(total_enc_gen_loss)
    train_enc_gen_accuracy(ae_enc_acc)
    train_enc_disc_loss(total_enc_disc_loss)
    train_dec_gen_loss(total_dec_gen_loss)
    train_dec_gen_accuracy(ae_dec_acc)
    train_dec_disc_loss(total_dec_disc_loss)

    if teacher is True:
        train_enc_cycled_loss(cycled_ae_enc_loss)
        train_enc_cycled_accuracy(enc_cycled_acc)
    else:
        train_enc_cycled_loss(0)
        train_enc_cycled_accuracy(0)



from tqdm import tqdm,trange
import csv
from rouge import Rouge
rouge=Rouge()
from nltk.translate.bleu_score import sentence_bleu



createFolder(consts.FILE_NAME)

f= open(consts.FILE_NAME+'/training_results_with_scores.csv', 'w', newline='')
wr=csv.writer(f)
wr.writerow(['orig_article','orig_summary','gen_article','gen_summary','art_rouge','art_bleu','sum_rouge','sum_bleu'])
f2= open(consts.FILE_NAME+'/training_loss_acc.csv', 'w', newline='')
wr2=csv.writer(f2)
wr2.writerow(['enc_gen_loss','enc_gen_acc','enc_disc_loss','dec_gen_loss','dec_gen_acc','dec_disc_loss',"enc_cycled_loss","enc_cycled_accuracy"])
#EPOCHS=0

# def generate():
    # generate_art=my_bart_decoder([set[0]['decoder_input_ids'],[]],training=False)
    # generate_sum=my_bart_encoder({'input_ids':set[0]['input_ids']},teacher=False)
    # #teacher=my_bart_encoder({'input_ids': tf.concat([summarize_tokens,set[0]['input_ids']],axis=-1),'decoder_input_ids' : set[0]['decoder_input_ids']},training=True)
    # #print(logits)
    # #print("generate : " + tokenizer.batch_decode(generate)[0])
    # #print("")
    # art_rouge = rouge.get_scores([tokenizer.decode(set[0]['input_ids'][0])],[tokenizer.decode(generate_art[0])])
    # art_bleu = sentence_bleu([tokenizer.decode(set[0]['input_ids'][0])], tokenizer.decode(generate_art[0]))
    # sum_rouge = rouge.get_scores([tokenizer.decode(set[0]['decoder_input_ids'][0])],[tokenizer.decode(tf.argmax(generate_sum[0],axis=-1))])
    # sum_bleu = sentence_bleu([tokenizer.decode(set[0]['decoder_input_ids'][0])],tokenizer.decode(tf.argmax(generate_sum[0],axis=-1)))
    # wr.writerow([tokenizer.decode(set[0]['input_ids'][0]), tokenizer.decode(set[0]['decoder_input_ids'][0]),tokenizer.decode(generate_art[0]),tokenizer.decode(tf.argmax(generate_sum[0],axis=-1)),art_rouge,art_bleu,sum_rouge,sum_bleu])



for epoch in trange(consts.EPOCHS):
    start = time.time()
    train_enc_gen_loss.reset_states()
    train_enc_gen_accuracy.reset_states()
    train_enc_disc_loss.reset_states()
    train_dec_gen_loss.reset_states()
    train_dec_gen_accuracy.reset_states()
    train_dec_disc_loss.reset_states()
    train_enc_cycled_loss.reset_states()
    train_enc_cycled_accuracy.reset_states()
    print("")

    teacher=True
    
    whole_length=tf_train_dataset.__len__()
    print("whole batch length : "+str(whole_length))

    # if epoch>=2: 
    #     # 그 이후로는 recon loss만을 사용한다.
    #     teacher=False
    #     print("this epoch teacher forcing is not used")

# # #    inp -> long sentences, tar -> summary
    for (batch, set) in enumerate(tqdm(tf_train_dataset)):
        
        train_step(set[0]['input_ids'], set[0]['decoder_input_ids'],teacher=teacher)
        # if batch % 1000 ==0:
        with train_summary_writer.as_default():
            tf.summary.scalar('enc_gen_loss', train_enc_gen_loss.result(), step=epoch)
            tf.summary.scalar('enc_gen_accuracy', train_enc_gen_accuracy.result(), step=epoch)
            tf.summary.scalar('enc_disc_loss', train_enc_disc_loss.result(), step=epoch)
            tf.summary.scalar('dec_gen_loss', train_dec_gen_loss.result(), step=epoch)
            tf.summary.scalar('dec_gen_accuracy', train_dec_gen_accuracy.result(), step=epoch)
            tf.summary.scalar('dec_disc_loss', train_dec_disc_loss.result(), step=epoch)
            tf.summary.scalar('dec_enc_cycled_loss', train_enc_cycled_loss.result(), step=epoch)
            tf.summary.scalar('dec_enc_cycled_accuracy', train_enc_cycled_accuracy.result(), step=epoch)
        # if batch % 100 ==0 :
        #     # wr2.writerow([float(train_enc_gen_loss.result()),float(train_enc_gen_accuracy.result()),float(train_enc_disc_loss.result()),
        #     # float(train_dec_gen_loss.result()),float(train_dec_gen_accuracy.result()),float(train_dec_disc_loss.result()),float(train_enc_cycled_loss.result()),
        #     # float(train_enc_cycled_accuracy.result())])
        #     print(f'Epoch {epoch + 1} Batch {batch} Encoder_Gen_Loss: {train_enc_gen_loss.result():.4f} | Accuracy: {train_enc_gen_accuracy.result():.4f}| Encoder_Disc_Loss: {train_enc_disc_loss.result():.4f} | Decoder_Gen_Loss: {train_dec_gen_loss.result():.4f} | Accuracy: {train_dec_gen_accuracy.result():.4f} | Decoder_Disc_Loss: {train_dec_disc_loss.result():.4f}')
        #     print(f'| Encoder_Cycle_Loss: {train_enc_cycled_loss.result():.4f} | Accuracy: {train_enc_cycled_accuracy.result():.4f}')
    
    ckpt_save_path = ckpt_manager.save()
    print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

    print(f'Epoch {epoch + 1} Batch {batch} Encoder_Gen_Loss: {train_enc_gen_loss.result():.4f} | Accuracy: {train_enc_gen_accuracy.result():.4f}| Encoder_Disc_Loss: {train_enc_disc_loss.result():.4f} | Decoder_Gen_Loss: {train_dec_gen_loss.result():.4f} | Accuracy: {train_dec_gen_accuracy.result():.4f} | Decoder_Disc_Loss: {train_dec_disc_loss.result():.4f}')
    print(f'| Encoder_Cycle_Loss: {train_enc_cycled_loss.result():.4f} | Accuracy: {train_enc_cycled_accuracy.result():.4f}')
    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
