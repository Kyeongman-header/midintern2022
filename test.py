from datasets import load_dataset
import time
import tensorflow as tf
import os
#tf.debugging.set_log_device_placement(True)
#tf.config.set_visible_devices([], 'GPU') # CPU로 학습하기.

def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error Creating directory. ' + directory)


dataset = load_dataset("cnn_dailymail", '3.0.0') # cnn dailymail로 했지만 다른 데이터도 같은 데이터 형식(dictionary, "long" : ~, "short" : ~)
# print(dataset["train"][100]['highlights'])
# print(dataset["train"][100]['article'])



from transformers import BartTokenizer,T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small",model_max_length=1024)

MAX_VOCAB = len(tokenizer.get_vocab())
print('VOCAB_SIZE :',  MAX_VOCAB)
BATCH_SIZE=4
LONG_MAX=800
SHORT_MAX=100

def tokenize_function(examples):
    return {'input_ids' : tokenizer(examples["article"],max_length=LONG_MAX, padding='max_length', truncation=True)['input_ids'],'decoder_input_ids' : tokenizer(examples["highlights"], max_length=SHORT_MAX, padding='max_length', truncation=True)['input_ids']}

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

#print(small_train_dataset)

from transformers import DefaultDataCollator

data_collator = DefaultDataCollator(return_tensors="tf")

tf_train_dataset = small_train_dataset.to_tf_dataset(
    columns=['input_ids', 'decoder_input_ids'],
    label_cols=['decoder_input_ids'],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=BATCH_SIZE,
) # train dataset을 batch 사이즈별로 제공해줌.

#print(tf_train_dataset)

tf_validation_dataset = small_eval_dataset.to_tf_dataset(
    columns=['input_ids', 'decoder_input_ids'],
    label_cols=['decoder_input_ids'],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=BATCH_SIZE,
)


# print(tf_train_dataset)

from transformers import TFAutoModel,TFBartModel,TFBartForConditionalGeneration
from transformers import TFT5ForConditionalGeneration
bart_model = TFBartModel.from_pretrained("facebook/bart-base")
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
            bart_output=self.bart_model({'inputs_embeds' : embedding, 'decoder_input_ids' : inp,'training':True}).logits # conditional generation은 logits이 last hidden state를 대체함.
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
        self.final_layer = tf.keras.layers.Dense(vocab_size)
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, teacher=False):
    # Keras models prefer if you pass all your inputs in the first argument
        #bart_output=self.bart_model(inputs).last_hidden_state
        if teacher is True:
            #print(inputs)
            bart_output=self.bart_model({'input_ids':inputs['input_ids'],'decoder_input_ids':inputs['decoder_input_ids']}).last_hidden_state # 왜인지 모르겠지만 얘는 되는데 decoder는 같은 코드가 실행이 안됨
            #print("call shape " + str(bart_output.shape))
        else:
            #bart_output=self.bart_model.generate(inputs['input_ids'])
            bart_output=self.bart_model({'input_ids':inputs['input_ids']}).last_hidden_state

        dropout=self.dropout(bart_output)
        
        final_output = self.final_layer(dropout)  # (batch_size, tar_seq_len, target_vocab_size)
        
        #return bart_output
        return final_output


mirrored_strategy = tf.distribute.MirroredStrategy()
gpus = tf.config.experimental.list_logical_devices('GPU')

print(gpus[0].name)
print(gpus[1].name)
#with tf.device(gpus[0].name):
my_bart_encoder = My_Encoder_BART(
    model=bart_model,
    vocab_size=MAX_VOCAB,
    rate=0.1)

BART_BASE_DIM=768
BART_LARGE_DIM=1024
T5_SMALL_DIM=512
#with tf.device(gpus[1].name):
my_bart_decoder = My_Decoder_BART(
    model=decoder_bart_model,
    vocab_size=MAX_VOCAB,
    dim=T5_SMALL_DIM,
    #tokenizers.pt.get_vocab_size().numpy(),
    rate=0.1)
#print(inputs["input_ids"])
#print(my_bart_encoder({'input_ids' : inputs["input_ids"], 'decoder_input_ids' : outputs["input_ids"]}).shape)
#print("test")
#print(my_bart_decoder([tf.random.uniform((5, 57), dtype=tf.int32, minval=0, maxval=200),tf.random.uniform((5, 768), dtype=tf.int32, minval=0, maxval=200)],training=False).shape)

LEARNING_RATE=0.0001 # 0.0001 보다 크면, 학습이 제대로 되지 않는다!!

summary_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
reconstruction_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

alpha=0.85

checkpoint_path = "./MY_checkpoints/train_"+str(alpha)

ckpt = tf.train.Checkpoint(my_bart_decoder=my_bart_decoder,my_bart_encoder=my_bart_encoder,
                          summary_optimizer=summary_optimizer, reconstruction_optimizer=reconstruction_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=None)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')
else :
    print('This is New train!!(no checkpoint)')

#ckpt_save_path = ckpt_manager.save()
#print(f'Saving checkpoint for epoch at {ckpt_save_path}')

EPOCHS=20

train_summary_loss = tf.keras.metrics.Mean(name='train_summary_loss')
train_reconstruction_loss=tf.keras.metrics.Mean(name='train_reconstruction_loss')
train_summary_accuracy = tf.keras.metrics.Mean(name='train_summary_accuracy')
train_reconstruction_accuracy=tf.keras.metrics.Mean(name='train_reconstruction_accuracy')


# 학습에 필요한 특수 토큰(summarize: 이거랑 <pad>를 앞에 붙여줘야 함)
summarize_token=tokenizer('summarize: ')['input_ids'][:-1]
summarize_tokens=[]
for i in range(BATCH_SIZE):
        summarize_tokens.append(summarize_token)

summarize_tokens=tf.convert_to_tensor(summarize_tokens,dtype=tf.int64)
#print(summarize_tokens.shape)


pad_token=tokenizer('<pad>')['input_ids'][:-1]
pad_tokens=[]
for i in range(BATCH_SIZE):
        pad_tokens.append(pad_token)

pad_tokens=tf.convert_to_tensor(pad_tokens,dtype=tf.int64)       
#print(pad_tokens.shape)

@tf.function
def train_step(inp, summary,alpha,teacher):
    decoder_input_inp=inp[:,:-1]
    decoder_input_summary=summary[:,:-1] # huggingface 예시를 보면 이렇게 할 필요가 없는것 같아 보였지만 낚시였다.
    # teacher는  <pad>로 시작하고 </s> 전까지 생성하도록 했다
    # 그러면 loss를 구할 때는 <pad> 없이, </s>까지 생성해내도록 한다.
    # 이 방식으로 하지 않으면 generate의 결과가 완전 엉뚱해진다!!!
    TEACHER=teacher
    
    with tf.GradientTape(persistent=True) as tape:
        #print( tf.concat([summarize_tokens,inp],axis=-1))
        #print(summary)
        output= my_bart_encoder({'input_ids' : tf.concat([summarize_tokens,inp],axis=-1), 'decoder_input_ids' : tf.concat([pad_tokens,decoder_input_summary],axis=-1),'training':True},teacher=TEACHER)
        #output=my_bart_encoder({'input_ids' : tf.concat([summarize_tokens,inp],axis=-1),'training':True},teacher=False)
        if TEACHER :
            loss = summary_loss(summary, output)
        fake_input_predictions = my_bart_decoder([output,tf.concat([pad_tokens,decoder_input_inp],axis=-1)],training=True)
        recon_loss = reconstruction_loss(inp,fake_input_predictions)
        
        if TEACHER :
            total_loss=alpha*loss+(1-alpha)*recon_loss #total loss를 tape scope안에서 구해야 함.
        else :
            total_loss=recon_loss
    
    summary_gradients = tape.gradient(total_loss, my_bart_encoder.trainable_variables)
    reconstruction_gradients=tape.gradient(recon_loss,my_bart_decoder.trainable_variables)
  
    summary_optimizer.apply_gradients(zip(summary_gradients, my_bart_encoder.trainable_variables))
    reconstruction_optimizer.apply_gradients(zip(reconstruction_gradients,my_bart_decoder.trainable_variables))
    
    if TEACHER:
        train_summary_loss(loss)
        train_summary_accuracy(summary_accuracy_function(summary, output))
    else:
        train_summary_loss(0)
        train_summary_accuracy(0)
    train_reconstruction_loss(recon_loss)
    train_reconstruction_accuracy(reconstruction_accuracy_function(inp, fake_input_predictions))

#train_step(inputs, outputs) #예시


#for (batch, set) in enumerate(tf_train_dataset):
#    print(batch)
#    train_step(set[0]['input_ids'], set[0]['decoder_input_ids'])
#    print(train_summary_loss.result())
#    print(train_reconstruction_loss.result())
#    if(batch==100):
#        break #예시. 



from tqdm import tqdm,trange
import csv
from rouge import Rouge
rouge=Rouge()
from nltk.translate.bleu_score import sentence_bleu

NORMAL='normal_'
CASCADE='cascade_'
strategy=CASCADE

createFolder(strategy+str(alpha))

f= open(strategy+str(alpha)+'/training_results_with_scores.csv', 'w', newline='')
wr=csv.writer(f)
wr.writerow(['orig_article','orig_summary','gen_article','gen_summary','art_rouge','art_bleu','sum_rouge','sum_bleu'])
f2= open(strategy+str(alpha)+'/training_loss_acc.csv', 'w', newline='')
wr2=csv.writer(f2)
wr2.writerow(['sum_loss','sum_acc','art_loss','art_acc'])
#EPOCHS=0
for epoch in trange(EPOCHS):
    start = time.time()
    train_summary_loss.reset_states()
    train_summary_accuracy.reset_states()
    train_reconstruction_loss.reset_states()
    train_reconstruction_accuracy.reset_states()
    print("")
    teacher=True
    if epoch>=2:
        teacher=False
# # #    inp -> long sentences, tar -> summary
    for (batch, set) in enumerate(tqdm(tf_train_dataset)):
        #print("batch : "+str(batch))

        train_step(set[0]['input_ids'], set[0]['decoder_input_ids'],alpha=alpha,teacher=teacher)
        if batch % 1000 ==0:
            generate_art=my_bart_decoder([set[0]['decoder_input_ids'],[]],training=False)
            generate_sum=my_bart_encoder({'input_ids':set[0]['input_ids']},teacher=False)
            #teacher=my_bart_encoder({'input_ids': tf.concat([summarize_tokens,set[0]['input_ids']],axis=-1),'decoder_input_ids' : set[0]['decoder_input_ids']},training=True)
            #print(logits)
            #print("generate : " + tokenizer.batch_decode(generate)[0])
            #print("")
            art_rouge = rouge.get_scores([tokenizer.decode(set[0]['input_ids'][0])],[tokenizer.decode(generate_art[0])])
            art_bleu = sentence_bleu([tokenizer.decode(set[0]['input_ids'][0])], tokenizer.decode(generate_art[0]))
            sum_rouge = rouge.get_scores([tokenizer.decode(set[0]['decoder_input_ids'][0])],[tokenizer.decode(tf.argmax(generate_sum[0],axis=-1))])
            sum_bleu = sentence_bleu([tokenizer.decode(set[0]['decoder_input_ids'][0])],tokenizer.decode(tf.argmax(generate_sum[0],axis=-1)))
            wr.writerow([tokenizer.decode(set[0]['input_ids'][0]), tokenizer.decode(set[0]['decoder_input_ids'][0]),tokenizer.decode(generate_art[0]),tokenizer.decode(tf.argmax(generate_sum[0],axis=-1)),art_rouge,art_bleu,sum_rouge,sum_bleu])


        if batch % 100 ==0 :
            wr2.writerow([float(train_summary_loss.result()),float(train_summary_accuracy.result()),float(train_reconstruction_loss.result()),float(train_reconstruction_accuracy.result())])
            print(f'Epoch {epoch + 1} Batch {batch} Summary Loss {train_summary_loss.result():.4f} Accuracy {train_summary_accuracy.result():.4f} Reconstruct Loss {train_reconstruction_loss.result():.4f} Accuracy {train_reconstruction_accuracy.result():.4f}')
    
    ckpt_save_path = ckpt_manager.save()
    print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

    print(f'Epoch {epoch + 1} Summary Loss {train_summary_loss.result():.4f} Accuracy {train_summary_accuracy.result():.4f}')
    print(f'Epoch {epoch + 1} Reconstruct Loss {train_reconstruction_loss.result():.4f} Accuracy {train_reconstruction_accuracy.result():.4f}')
    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

def plot_Expansion(decoder_model, plot,origin_plot):
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

def make_Plot(encoder_model,higher_plot,origin_summary):
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

f3= open(strategy+str(alpha)+'/valid_results_with_scores.csv', 'w', newline='')
f4= open(strategy+str(alpha)+'/valid_scores_only.csv','w',newline='')
wr3=csv.writer(f3)
wr3.writerow(['orig_article','orig_summary','gen_article','gen_summary','art_rouge','art_bleu','sum_rouge','sum_bleu'])
wr4=csv.writer(f4)
wr4.writerow(['art_rouge','art_bleu','sum_rouge','sum_bleu']) # bleu와 rouge score만 따로 저장하는 파일.
# #    inp -> long sentences, tar -> summary
for (batch, set) in enumerate(tqdm(tf_validation_dataset)):
    plot_logits,tokenize_summary,summary=make_Plot(my_bart_encoder,set[0]['input_ids'],set[0]['decoder_input_ids'])
    exp_logits,expansion=plot_Expansion(my_bart_decoder,tokenize_summary,set[0]['input_ids'])
    #print("generated summary : ")
    #print(summary)
    #print("generated article : ")
    #print(expansion)
    
    art_rouge = rouge.get_scores([tokenizer.decode(set[0]['input_ids'][0])],[expansion[0]])
    art_bleu = sentence_bleu([tokenizer.decode(set[0]['input_ids'][0])], expansion[0])
    sum_rouge = rouge.get_scores([tokenizer.decode(set[0]['decoder_input_ids'][0])],[summary[0]])
    sum_bleu = sentence_bleu([tokenizer.decode(set[0]['decoder_input_ids'][0])],summary[0])
    wr3.writerow([tokenizer.decode(set[0]['input_ids'][0]), tokenizer.decode(set[0]['decoder_input_ids'][0]),expansion[0],summary[0],art_rouge,art_bleu,sum_rouge,sum_bleu])
    wr4.writerow([art_rouge,art_bleu,sum_rouge,sum_bleu]) # 차원이 다르므로 loss는 불가하다.

        #wr4.writerow([float(summary_loss(set[0]['decoder_input_ids'],plot_logits)),float(summary_accuracy_function(set[0]['decoder_input_ids'],plot_logits)),float(reconstruction_loss(set[0]['input_ids'],exp_logits)),float(reconstruction_accuracy_function(set[0]['input_ids'],exp_logits))])
    
        #print(f'Batch {batch} Summary Loss ' + str(summary_loss(set[0]['decoder_input_ids'],plot_logits)) +  " Recon Loss " + str(reconstruction_loss(set[0]['input_ids'],exp_logits)) + 'Summary Accuracy : ' + str(summary_accuracy_function(set[0]['decoder_input_ids'],plot_logits)) + 'Recons Accuracy ' + str(reconstruction_accuracy_function(set[0]['input_ids'],exp_logits))) 
    


