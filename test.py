from datasets import load_dataset
import time
import tensorflow as tf
import os
#tf.config.set_visible_devices([], 'GPU')

dataset = load_dataset("cnn_dailymail", '3.0.0') # cnn dailymail로 했지만 다른 데이터도 같은 데이터 형식(dictionary, "long" : ~, "short" : ~)
# print(dataset["train"][100]['highlights'])
# print(dataset["train"][100]['article'])

print()

from transformers import BartTokenizer,T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small",model_max_length=1024)

MAX_VOCAB = len(tokenizer.get_vocab())+1
print('VOCAB_SIZE :',  MAX_VOCAB)
BATCH_SIZE=2
LONG_MAX=1024
SHORT_MAX=100

def tokenize_function(examples):
    return {'input_ids' : tokenizer(examples["article"],max_length=LONG_MAX, padding='max_length', truncation=True)['input_ids'],'decoder_input_ids' : tokenizer(examples["highlights"], max_length=SHORT_MAX, padding='max_length', truncation=True)['input_ids']}

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10000))

print(small_train_dataset)

from transformers import DefaultDataCollator

data_collator = DefaultDataCollator(return_tensors="tf")

tf_train_dataset = small_train_dataset.to_tf_dataset(
    columns=['input_ids', 'decoder_input_ids'],
    label_cols=['decoder_input_ids'],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=BATCH_SIZE,
) # train dataset을 batch 사이즈별로 제공해줌.

print(tf_train_dataset)

tf_validation_dataset = small_eval_dataset.to_tf_dataset(
    columns=['input_ids', 'decoder_input_ids'],
    label_cols=['decoder_input_ids'],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=BATCH_SIZE,
)


# print(tf_train_dataset)

#from transformers import TFAutoModel,BartModel,TFBartForConditionalGeneration
from transformers import TFT5ForConditionalGeneration
bart_model = TFT5ForConditionalGeneration.from_pretrained("t5-small")
inputs=tokenizer(dataset["train"][100]['article'], truncation=True,return_tensors="tf")
outputs=tokenizer(dataset["train"][100]['highlights'], truncation=True,return_tensors="tf")
# print(inputs)
# print("article :"+ str(dataset["train"][100]['article']))
# print("length : "+ str(len(dataset["train"][100]['article'])))
# print("summary :"+ str(dataset["train"][100]['highlights']))
# print("length :"+ str(len(dataset["train"][100]['highlights'])))
# made=tokenizer.decode(tf.squeeze(tf.argmax(bart_model({'input_ids' : inputs["input_ids"], 'decoder_input_ids' : outputs["input_ids"]}).logits,axis=2)))
# print("made summary : "+ made)
# print("length : "+str(len(made)))
decoder_bart_model = TFT5ForConditionalGeneration.from_pretrained("t5-small")
# made_2=tokenizer.decode(tf.squeeze(tf.argmax(decoder_bart_model({'input_ids' : outputs["input_ids"], 'decoder_input_ids' : inputs["input_ids"]}).logits,axis=2)))
# print("decoded artice : " + made_2)
# print("length :"+str(len(made_2)))



loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def summary_loss(logits, pred): # CUSTOM LOSS.
    mask = tf.math.logical_not(tf.math.equal(logits, 0))
    loss_ = loss_object(logits, pred)

    mask = tf.cast(mask, dtype=loss_.dtype) #mask가 되어있던 자리는 학습을 하지 않는다
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def reconstruction_loss(real,pred): #fake input과 실제input의 reconstruction loss
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype) #mask가 되어있던 자리는 학습을 하지 않는다
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def summary_accuracy_function(summary,pred):
  accuracies = tf.equal(summary, tf.argmax(pred, axis=2,output_type=tf.int64))

  mask = tf.math.logical_not(tf.math.equal(summary, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


def reconstruction_accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2,output_type=tf.int64))# int32가 아니면 type이 다르다고 오류남

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

alpha=0.4

class My_Decoder_BART(tf.keras.Model):
    def __init__(self, model,dim,vocab_size,rate=0.1):
        super().__init__()
        self.bart_model = model
        self.input_layer = tf.keras.layers.Dense(dim)
        # self.output_layer=tf.keras.layers.Dense(1)
        #self.final_layer = tf.keras.layers.Dense(vocab_size)
        #self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
    # Keras models prefer if you pass all your inputs in the first argument
        encoder_output,inp=inputs
        
        #bart_output=self.bart_model({'inputs_embeds' : embedding, 'decoder_input_ids' : input}).last_hidden_state #ㅑ last hidden state를 하지 않으면 모든 hidden state가 전부 반환됨.
        if(training):
            embedding=self.input_layer(encoder_output)
            #print(input)
            #bart_output=self.bart_model(decoder_input_ids=inp,inputs_embeds=embedding)
            bart_output=self.bart_model({'inputs_embeds' : embedding, 'decoder_input_ids' : inp}).logits # conditional generation은 logits이 last hidden state를 대체함.
        # input을 id가 아니라 임베딩 차원으로 넣어줄 수 있는 기능이 있다. 그런데 이렇게 하면 decoder를 id로 넣어줘도 되는건지 확실치않음.
        else:
            #bart_output=self.bart_model({'input_ids' : encoder_output, 'decoder_input_ids' : inp}).logits #decoder input id가 없으면 무조건 ids로만 가능하댄다.
            bart_output=self.bart_model.generate(encoder_output)#이는 teacher forcing이 적용되지 않은, real output이다(loss 계산 불가)
            
        #dropout=self.dropout(bart_output)
        #final_output = self.final_layer(dropout)  # (batch_size, tar_seq_len, target_vocab_size)
        return bart_output
        #return final_output

class My_Encoder_BART(tf.keras.Model):
    def __init__(self, model,vocab_size,rate=0.1):
        super().__init__()
        self.bart_model = model
        #self.final_layer = tf.keras.layers.Dense(vocab_size)
        #self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
    # Keras models prefer if you pass all your inputs in the first argument
        #bart_output=self.bart_model(inputs).last_hidden_state
        if training is True:
            bart_output=self.bart_model(inputs).logits # 왜인지 모르겠지만 얘는 되는데 decoder는 같은 코드가 실행이 안됨
        else:
            bart_output=self.bart_model.generate(inputs['input_ids'])
        #dropout=self.dropout(bart_output)
        #final_output = self.final_layer(dropout)  # (batch_size, tar_seq_len, target_vocab_size)
        return bart_output
        #return final_output
my_bart_encoder = My_Encoder_BART(
    model=bart_model,
    vocab_size=MAX_VOCAB,
    #tokenizers.pt.get_vocab_size().numpy(),
    rate=0.1)

BART_BASE_DIM=768
BART_LARGE_DIM=1024
T5_SMALL_DIM=512
my_bart_decoder = My_Decoder_BART(
    model=bart_model,
    vocab_size=MAX_VOCAB,
    dim=T5_SMALL_DIM,
    #tokenizers.pt.get_vocab_size().numpy(),
    rate=0.1)
#print(inputs["input_ids"])
#print(my_bart_encoder({'input_ids' : inputs["input_ids"], 'decoder_input_ids' : outputs["input_ids"]}).shape)
#print("test")
print(my_bart_decoder([tf.random.uniform((5, 57), dtype=tf.int32, minval=0, maxval=200),tf.random.uniform((5, 768), dtype=tf.int32, minval=0, maxval=200)],training=False).shape)

summary_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
reconstruction_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

checkpoint_path = "./MY_checkpoints/train"

ckpt = tf.train.Checkpoint(my_bart_decoder=my_bart_decoder,my_bart_encoder=my_bart_encoder,
                          summary_optimizer=summary_optimizer, reconstruction_optimizer=reconstruction_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')

EPOCHS=20

train_summary_loss = tf.keras.metrics.Mean(name='train_summary_loss')
train_reconstruction_loss=tf.keras.metrics.Mean(name='train_reconstruction_loss')
train_summary_accuracy = tf.keras.metrics.Mean(name='train_summary_accuracy')
train_reconstruction_accuracy=tf.keras.metrics.Mean(name='train_reconstruction_accuracy')

#print(my_bart_encoder.trainable_variables)
#print(my_bart_decoder.trainable_variables)
# @tf.function


def train_step(inp, summary):
    with tf.GradientTape(persistent=True) as tape:

        output,_= my_bart_encoder({'input_ids' : inp, 'decoder_input_ids' : summary})
        #print(output)
        loss = summary_loss(summary, output) 
        fake_input_predictions,_ = my_bart_decoder([output,inp],training=True)
        recon_loss = reconstruction_loss(inp,fake_input_predictions)
        total_loss=alpha*loss+(1-alpha)*recon_loss #total loss를 tape scope안에서 구해야 함.
  
    summary_gradients = tape.gradient(total_loss, my_bart_encoder.trainable_variables)
    reconstruction_gradients=tape.gradient(recon_loss,my_bart_decoder.trainable_variables)
  
    summary_optimizer.apply_gradients(zip(summary_gradients, my_bart_encoder.trainable_variables))
    reconstruction_optimizer.apply_gradients(zip(reconstruction_gradients,my_bart_decoder.trainable_variables))
  
    train_summary_loss(loss)
    train_reconstruction_loss(recon_loss)
    train_summary_accuracy(summary_accuracy_function(summary, output))
    train_reconstruction_accuracy(reconstruction_accuracy_function(inp, fake_input_predictions))

#train_step(inputs, outputs) # 예시.


#for (batch, set) in enumerate(tf_train_dataset):
#    print(batch)
#    train_step(set[0]['input_ids'], set[0]['decoder_input_ids'])
#    print(train_summary_loss.result())
#    print(train_reconstruction_loss.result())
#    if(batch==100):
#        break #예시.

from tqdm import tqdm
from tqdm import trange

# for epoch in trange(EPOCHS):
#     start = time.time()
#     train_summary_loss.reset_states()
#     train_summary_accuracy.reset_states()
#     train_reconstruction_loss.reset_states()
#     train_reconstruction_accuracy.reset_states()
#     print("")

# # # #    inp -> long sentences, tar -> summary
#     for (batch, set) in enumerate(tqdm(tf_train_dataset)):
#         #print("batch : "+str(batch))
#         train_step(set[0]['input_ids'], set[0]['decoder_input_ids'])
#         if batch % 1000 ==0 and batch!=0:
#             print(f'\rEpoch {epoch + 1} Batch {batch} Summary Loss {train_summary_loss.result():.4f} Accuracy {train_summary_accuracy.result():.4f} Reconstruct Loss {train_reconstruction_loss.result():.4f} Accuracy {train_reconstruction_accuracy.result():.4f}', end='')
    
#     ckpt_save_path = ckpt_manager.save()
#     print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

#     print(f'Epoch {epoch + 1} Summary Loss {train_summary_loss.result():.4f} Accuracy {train_summary_accuracy.result():.4f}')
#     print(f'Epoch {epoch + 1} Reconstruct Loss {train_reconstruction_loss.result():.4f} Accuracy {train_reconstruction_accuracy.result():.4f}')
#     print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

def plot_Expansion(decoder_model, plot):
    #logits = []
    empty=[] # forward 단계에선 쓰지 않는다.
    logits=decoder_model([plot,empty],training=False)
    #for p in plot:
    #    logits.append(decoder_model([p,empty],training=False))
    
    plots=[]
    #print(logits)
    #tokenize_plots=tf.argmax(logits,axis=2,output_type=tf.int64)
    tokenize_plots=logits # 얘는 generate으로 생성한 거임. 왜 encoder와 decoder가 실행 방식이 다른지는 모르겟음
    for token in tokenize_plots:
        plots.append(tokenizer.decode(token))

    return logits,plots

def make_Plot(encoder_model,higher_plot):
    #logits = []
    logits=encoder_model({'input_ids':higher_plot},training=False)
    #for p in higher_plot:
    #    logits.append(encoder_model({'input_ids' : tf.expand_dims(p,axis=0)}))
    
    plots=[]
    #tokenize_plots=[]
    #print(logits)
    #tokenize_plots=tf.argmax(logits,axis=2,output_type=tf.int64)
    tokenize_plots=logits
    for token in tokenize_plots:
        #print(logit)
        #print(tf.argmax(logit,axis=2))
        #print(token)
        plots.append(tokenizer.decode(token))
    
    return logits,tokenize_plots,plots

import csv
f= open('valid data result', 'w', newline='')
wr=csv.writer(f)
wr.writerow(['orig_article','orig_summary','gen_article','gen_summary'])
# #    inp -> long sentences, tar -> summary
for (batch, set) in enumerate(tqdm(tf_validation_dataset)):
    plot_logits,tokenize_summary,summary=make_Plot(my_bart_encoder,set[0]['input_ids'])
    exp_logits,expansion=plot_Expansion(my_bart_decoder,tokenize_summary)
    for i in range(BATCH_SIZE):
        wr.writerow([set[0]['input_ids'][i],set[0]['decoder_input_ids'][i],expansion[i],summary[i]])
    
    #if batch % 5 == 0:
    #    print(f'\rBatch {batch} Summary Loss ' + str(summary_loss(set[0]['decoder_input_ids'],plot_logits)) +  " Recon Loss " + str(reconstruction_loss(set[0]['input_ids'],exp_logits)) + 'Summary Accuracy : ' + str(summary_accuracy_function(set[0]['decoder_input_ids'],plot_logits)) + 'Recons Accuracy ' + str(reconstruction_accuracy_function(set[0]['input_ids'],exp_logits)), end="") 



