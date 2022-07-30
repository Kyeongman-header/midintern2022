from datasets import load_dataset
import time
dataset = load_dataset("cnn_dailymail", '3.0.0') # cnn dailymail로 했지만 다른 데이터도 같은 데이터 형식(dictionary, "long" : ~, "short" : ~)
# print(dataset["train"][100]['highlights'])
# print(dataset["train"][100]['article'])

from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

MAX_VOCAB = len(tokenizer.get_vocab())+1
print('VOCAB_SIZE :',  MAX_VOCAB)
BATCH_SIZE=8


def tokenize_function(examples):
    return {'input_ids' : tokenizer(examples["article"], padding="max_length", truncation=True)['input_ids'],'decoder_input_ids' : tokenizer(examples["highlights"], padding="max_length", truncation=True)['input_ids']}

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(30))

from transformers import DefaultDataCollator

data_collator = DefaultDataCollator(return_tensors="tf")

tf_train_dataset = small_train_dataset.to_tf_dataset(
    columns=['input_ids', 'decoder_input_ids'],
    label_cols=['decoder_input_ids'],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=BATCH_SIZE,
) # train dataset을 batch 사이즈별로 제공해줌.

tf_validation_dataset = small_eval_dataset.to_tf_dataset(
    columns=['input_ids', 'decoder_input_ids'],
    label_cols=['decoder_input_ids'],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=BATCH_SIZE,
)


# print(tf_train_dataset)


import tensorflow as tf
from transformers import TFAutoModel,BartModel,TFBartForConditionalGeneration
bart_model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
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
decoder_bart_model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-base")
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
  accuracies = tf.equal(summary, tf.argmax(pred, axis=2,output_type=tf.int32))

  mask = tf.math.logical_not(tf.math.equal(summary, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


def reconstruction_accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2,output_type=tf.int32))# int32가 아니면 type이 다르다고 오류남

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
        self.final_layer = tf.keras.layers.Dense(vocab_size)
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
    # Keras models prefer if you pass all your inputs in the first argument
        encoder_output,input=inputs
        embedding=self.input_layer(encoder_output)
        
        #bart_output=self.bart_model({'inputs_embeds' : embedding, 'decoder_input_ids' : input}).last_hidden_state # last hidden state를 하지 않으면 모든 hidden state가 전부 반환됨.
        bart_output=self.bart_model({'inputs_embeds' : embedding, 'decoder_input_ids' : input}).logits # conditional generation은 logits이 last hidden state를 대체함.
        # input을 id가 아니라 임베딩 차원으로 넣어줄 수 있는 기능이 있다. 그런데 이렇게 하면 decoder를 id로 넣어줘도 되는건지 확실치않음.
        dropout=self.dropout(bart_output)
        final_output = self.final_layer(dropout)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output

class My_Encoder_BART(tf.keras.Model):
    def __init__(self, model,vocab_size,rate=0.1):
        super().__init__()
        self.bart_model = model
        self.final_layer = tf.keras.layers.Dense(vocab_size)
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
    # Keras models prefer if you pass all your inputs in the first argument
        
        #bart_output=self.bart_model(inputs).last_hidden_state
        bart_output=self.bart_model(inputs).logits
        dropout=self.dropout(bart_output)
        final_output = self.final_layer(dropout)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output
my_bart_encoder = My_Encoder_BART(
    model=bart_model,
    vocab_size=MAX_VOCAB,
    #tokenizers.pt.get_vocab_size().numpy(),
    rate=0.1)
my_bart_decoder = My_Decoder_BART(
    model=bart_model,
    vocab_size=MAX_VOCAB,
    dim=1024,
    #tokenizers.pt.get_vocab_size().numpy(),
    rate=0.1)
#print(inputs["input_ids"])
#print(my_bart_encoder({'input_ids' : inputs["input_ids"], 'decoder_input_ids' : outputs["input_ids"]}).shape)
#print(my_bart_decoder([tf.random.uniform((1, 57,MAX_VOCAB), dtype=tf.int32, minval=0, maxval=200),tf.random.uniform((1, 768), dtype=tf.int32, minval=0, maxval=200)]).shape)

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
        output= my_bart_encoder({'input_ids' : inp, 'decoder_input_ids' : summary})
        loss = summary_loss(summary, output) 
        fake_input_predictions = my_bart_decoder([output,inp])
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
#     if(batch==100):
#         break #예시.

# for epoch in range(EPOCHS):
#     start = time.time()
#     train_summary_loss.reset_states()
#     train_summary_accuracy.reset_states()
#     train_reconstruction_loss.reset_states()
#     train_reconstruction_accuracy.reset_states()

# # #    inp -> long sentences, tar -> summary
#     for (batch, set) in enumerate(tf_train_dataset):
#         print(batch)
#         train_step(set[0]['input_ids'], set[0]['decoder_input_ids'])
#         if batch % 50 == 0:
#             print(f'Epoch {epoch + 1} Batch {batch} Summary Loss {train_summary_loss.result():.4f} Accuracy {train_summary_accuracy.result():.4f}')
#             print(f'Epoch {epoch + 1} Batch {batch} Reconstruct Loss {train_reconstruction_loss.result():.4f} Accuracy {train_reconstruction_accuracy.result():.4f}')

#     if (epoch + 1) % 5 == 0:
#         ckpt_save_path = ckpt_manager.save()
#         print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

#     print(f'Epoch {epoch + 1} Summary Loss {train_summary_loss.result():.4f} Accuracy {train_summary_accuracy.result():.4f}')
#     print(f'Epoch {epoch + 1} Reconstruct Loss {train_reconstruction_loss.result():.4f} Accuracy {train_reconstruction_accuracy.result():.4f}')
#     print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

def plot_Expansion(decoder_model, plot):
    logits = []
    
    for p in plot:
        logits.append(decoder_model({'input_ids' : tf.expand_dims(p,axis=0)}))
    
    plots=[]
    
    for logit in logits:
        plots.append(tokenizer.decode(tf.argmax(logit,axis=2)))

    return logits,plots

def make_Plot(encoder_model,higher_plot):
    logits = []
    
    for p in higher_plot:
        logits.append(encoder_model({'input_ids' : tf.expand_dims(p,axis=0)}))
    
    plots=[]
    
    for logit in logits:
        plots.append(tokenizer.decode(tf.argmax(logit,axis=2)))
    
    return logits,plots


# #    inp -> long sentences, tar -> summary
for (batch, set) in enumerate(tf_validation_dataset):
    print(batch)
    plot_logits,summary=make_Plot(my_bart_encoder,set[0]['input_ids'])
    exp_logits,expansion=plot_Expansion(my_bart_decoder,summary)
    if batch % 10 == 0:
        print(f'Batch {batch} Summary Loss ' + str(summary_loss(set[0]['decoder_input_ids'],plot_logits)) +  "Recon Loss " + str(reconstruction_loss(set[0]['input_ids'],exp_logits)))
        print(f'Batch {batch} Summary Accuracy ' + str(summary_accuracy_function(set[0]['decoder_input_ids'],plot_logits)) + 'Accuracy ' + str(reconstruction_accuracy_function(set[0]['input_ids'],exp_logits)))



