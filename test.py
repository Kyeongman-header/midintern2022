from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", '3.0.0')
# print(dataset["train"][100]['highlights'])
# print(dataset["train"][100]['article'])

from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

MAX_VOCAB = len(tokenizer.get_vocab())+1
print('VOCAB_SIZE :',  MAX_VOCAB)
BATCH_SIZE=8


def tokenize_function(examples):
    return {'input_ids' : tokenizer(examples["article"], padding="max_length", truncation=True)['input_ids'],'decoder_input_ids' : tokenizer(examples["highlights"], padding="max_length", truncation=True)['input_ids']}

# tokenized_datasets = dataset.map(tokenize_function, batched=True)

# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# from transformers import DefaultDataCollator

# data_collator = DefaultDataCollator(return_tensors="tf")

# tf_train_dataset = small_train_dataset.to_tf_dataset(
#     columns=['input_1', 'decoder_input_ids'], # 밑에 저렇게 create model 해주면서 input_1로 input key 이름이 바뀜
#     label_cols=['decoder_input_ids'],
#     shuffle=True,
#     collate_fn=data_collator,
#     batch_size=BATCH_SIZE,
# )

# tf_validation_dataset = small_eval_dataset.to_tf_dataset(
#     columns=['input_1', 'decoder_input_ids'],
#     label_cols=['decoder_input_ids'],
#     shuffle=False,
#     collate_fn=data_collator,
#     batch_size=BATCH_SIZE,
# )

import tensorflow as tf
from transformers import TFAutoModel,BartModel
bart_model = TFAutoModel.from_pretrained("facebook/bart-large-cnn")
inputs=tokenizer(dataset["train"][100]['article'], truncation=True,return_tensors="tf")
outputs=tokenizer(dataset["train"][100]['highlights'], truncation=True,return_tensors="tf")
# print(inputs)
# print(bart_model({'input_ids' : inputs["input_ids"], 'decoder_input_ids' : outputs["input_ids"]}).last_hidden_state)

decoder_bart_model = TFAutoModel.from_pretrained("facebook/bart-base")
# print(decoder_bart_model({'input_ids' : inputs["input_ids"], 'decoder_input_ids' : outputs["input_ids"]}).last_hidden_state)

# # vocab_layer=tf.keras.layers.Dense(VOCAB_SIZE, input_shape=(BATCH_SIZE,), activation='sigmoid')
 


# def create_model(bart_model, dim,MAX_VOCAB):
#     input_ids = tf.keras.Input(shape=(dim,),dtype='int32')
#     #   attention_masks = tf.keras.Input(shape=(dim,),dtype='int32')

#     output = bart_model([input_ids])
#     output = output[0] # 이게 LAST HIDDEN STATE라는 뜻이래
#     # output = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1), name = "Pooling_Embs")(output)
#     output = tf.keras.layers.Dense(32,activation='relu')(output)
#     output = tf.keras.layers.Dropout(0.2)(output)

#     output = tf.keras.layers.Dense(MAX_VOCAB)(output) # sigmoid나 softmax로 ㅏ면 logits=TRUE로 하면 안된단다
#     model = tf.keras.models.Model(inputs = [input_ids],outputs = output)
#     model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=tf.metrics.SparseCategoricalAccuracy(),
#     )  
#     return model

# custom model creation

# model = create_model(bart_model, dim=BATCH_SIZE, MAX_VOCAB=MAX_VOCAB)
# model.summary()


# model = tf.keras.Model(
#             inputs=model,
#             outputs=[vocab_layer])

# print(model.get_config())


# model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=1)


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
  accuracies = tf.equal(real, tf.argmax(pred, axis=2,output_type=tf.int32))# 이걸 안해주니깐 type이 다르다고 오류남

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
        # output_embedding=tf.reshape(self.output_layer(input),shape=[1,-1])
        bart_output=self.bart_model({'inputs_embeds' : embedding, 'decoder_input_ids' : input}).last_hidden_state

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
        
        bart_output=self.bart_model(inputs).last_hidden_state
        #print(bart_output.shape)

        #여기 입력으로 들어오는 inp는 Embedded Summary이므로, Training 단계에서는 embedding이 필요하지 않다.
        # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        #dropout=self.dropout(input)
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
#print(my_bart_decoder([tf.random.uniform((1, 57,MAX_VOCAB), dtype=tf.int32, minval=0, maxval=200),tf.random.uniform((1, 57), dtype=tf.int32, minval=0, maxval=200)]).shape)

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

def train_step(inp, summary):

  with tf.GradientTape(persistent=True) as tape:
    output= my_bart_encoder({'input_ids' : inp["input_ids"], 'decoder_input_ids' : summary["input_ids"]})
    loss = summary_loss(summary['input_ids'], output) #encoder의 summary loss는 final dense layer를 거친 prediction이 들어간다.

    fake_input_predictions = my_bart_decoder([output,inp["input_ids"]]) # decoder에는 embedded 된 상태인 summary prediction이 들어가 있다. 고로 decoder에선 embedding을 해줄 필요가 없다.
    # 저 argmax는 실제로 문장이 완성된 녀석이다.
    recon_loss = reconstruction_loss(inp['input_ids'],fake_input_predictions)
    total_loss=alpha*loss+(1-alpha)*recon_loss # 이 total loss를 tape scope 안에서 해줬어야 했던 거네~~
    #loss=alpha*loss
    #recon_loss=(1-alpha)*recon_loss
  print(loss)
  print(recon_loss)
  print(total_loss)
  
  
  summary_gradients = tape.gradient(total_loss, my_bart_encoder.trainable_variables)
  reconstruction_gradients=tape.gradient(recon_loss,my_bart_decoder.trainable_variables)
  
  summary_optimizer.apply_gradients(zip(summary_gradients, my_bart_encoder.trainable_variables))
  reconstruction_optimizer.apply_gradients(zip(reconstruction_gradients,my_bart_decoder.trainable_variables))
  
  train_summary_loss(loss)
  train_reconstruction_loss(recon_loss)
  train_summary_accuracy(summary_accuracy_function(summary['input_ids'], output))
  train_reconstruction_accuracy(reconstruction_accuracy_function(inp['input_ids'], fake_input_predictions))







# print(inputs)
# print(outputs)
#train_step(inputs, outputs)

for epoch in range(EPOCHS):
    start = time.time()
    train_summary_loss.reset_states()
    train_summary_accuracy.reset_states()
    train_reconstruction_loss.reset_states()
    train_reconstruction_accuracy.reset_states()

#    inp -> long sentences, tar -> summary
    for (batch, (inp, summary)) in enumerate(train_batches):
        train_step(inp, summary)
        if batch % 50 == 0:
            print(f'Epoch {epoch + 1} Batch {batch} Summary Loss {train_summary_loss.result():.4f} Accuracy {train_summary_accuracy.result():.4f}')
            print(f'Epoch {epoch + 1} Batch {batch} Reconstruct Loss {train_reconstruction_loss.result():.4f} Accuracy {train_reconstruction_accuracy.result():.4f}')

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

    print(f'Epoch {epoch + 1} Summary Loss {train_summary_loss.result():.4f} Accuracy {train_summary_accuracy.result():.4f}')
    print(f'Epoch {epoch + 1} Reconstruct Loss {train_reconstruction_loss.result():.4f} Accuracy {train_reconstruction_accuracy.result():.4f}')
    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')







