import tensorflow as tf
import numpy as np
import consts

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)



class My_Decoder_tiny_T5(tf.keras.Model):
    def __init__(self, model,dim,vocab_size,inp_length,length,rate=0.1):
        super().__init__()
        self.vocab_size=vocab_size
        self.dim=dim
        self.length=length
        self.t5_model = model
        self.inp_length=inp_length
        self.Embedding=tf.keras.layers.CategoryEncoding(num_tokens=vocab_size, output_mode="one_hot")
        self.Dense=tf.keras.layers.Dense(dim)

    def call(self, inputs,starts,end, teacher,is_first=False):
        encoder_output,inp=inputs
    # Keras models prefer if you pass all your inputs in the first argument
        encoder_output=self.embedding(encoder_output,self.vocab_size,self.dim,self.inp_length,is_first)
        if(teacher):
            t5_output=self.t5_model({'inputs_embeds' : encoder_output, 'decoder_input_ids' : inp,'training':True}).logits # conditional generation은 logits이 last hidden state를 대체함.
        else: # inference
            output_array=starts #(3,1)

            for i in tf.range(self.length):


                predictions = self.t5_model({"inputs_embeds":encoder_output, "decoder_input_ids":output_array}, training=False).logits
                #(2, len, vocab_size)
                ids=tf.argmax(predictions, axis=-1,output_type=tf.int64) #(3,len)
                last_id=tf.reshape(ids[:,-1],shape=[-1,1]).numpy() #(3,1)
                output_array=np.append(output_array,last_id,axis=1) #(3,len+1)

            t5_output=predictions#이는 teacher forcing이 적용되지 않은, real output이다(recon loss 계산 불가)
            # 마지막 predictions는 (b,length,vocab_size)이다.#이는 teacher forcing이 적용되지 않은, real output이다(recon loss 계산 불가)
        
          # (batch_size, tar_seq_len, target_vocab_size)
        return t5_output
    def embedding(self,input,input_vocab_size,d_model,length,is_first=False):
        embedding=input
        if is_first is True:
            embedding=tf.one_hot(embedding,depth=input_vocab_size,on_value=1.0, off_value=0.0,axis=-1)
        embedding=self.Dense(embedding) 
        embedding *= tf.math.sqrt(tf.cast(d_model, tf.float32))
        embedding += positional_encoding(length,d_model)[:, :length, :]
        
        return embedding

class My_Encoder_tiny_T5(tf.keras.Model):
    def __init__(self, model,vocab_size,dim,length,inp_length,rate=0.1):
        super().__init__()
        self.vocab_size=vocab_size
        self.dim=dim
        self.length=length
        self.inp_length=inp_length
        self.t5_model = model
        self.Embedding=tf.keras.layers.CategoryEncoding(num_tokens=vocab_size, output_mode="one_hot")
        self.Dense=tf.keras.layers.Dense(dim)

    def call(self, inputs,starts,end, teacher=False,is_first=False):
    # Keras models prefer if you pass all your inputs in the first argument
        inp,summary=inputs
        inp=self.embedding(inp,self.vocab_size,self.dim,self.inp_length,is_first=is_first)
        if(teacher):
            t5_output=self.t5_model({'inputs_embeds' : inp, 'decoder_input_ids' : summary,'training':True}).logits
        else: # inference, 혹은 no teacher forcing일때(이게 될까?!!)

            output_array=starts #(2,1)

            for i in tf.range(self.length):


                predictions = self.t5_model({"inputs_embeds":inp, "decoder_input_ids":output_array}, training=False).logits
                #(2, len, vocab_size)
                ids=tf.argmax(predictions, axis=-1,output_type=tf.int64) #(3,len)
                last_id=tf.reshape(ids[:,-1],shape=[-1,1]).numpy() #(3,1)
                output_array=np.append(output_array,last_id,axis=1) #(3,len+1)

            t5_output=predictions#이는 teacher forcing이 적용되지 않은, real output이다(recon loss 계산 불가)
            # 마지막 predictions는 (b,length,vocab_size)이다.
        
        return t5_output
    def embedding(self,input,input_vocab_size,d_model,length,is_first=False):
        embedding=input
        if is_first is True:
            embedding=tf.one_hot(embedding,depth=input_vocab_size,on_value=1.0, off_value=0.0,axis=-1)
        embedding=self.Dense(embedding) 
        embedding *= tf.math.sqrt(tf.cast(d_model, tf.float32))
        embedding += positional_encoding(length,d_model)[:, :length, :]
        
        return embedding

class My_Disc(tf.keras.Model):

    def __init__(self, vocab_size,length,dim,rate=0.1):
        super().__init__()
        self.vocab_size=vocab_size
        self.length=length
        self.dim=dim
        #self.bert_model=model
        self.Dense=tf.keras.layers.Dense(1)
        self.Dropout=tf.keras.layers.Dropout(rate)
        self.Embedding=tf.keras.layers.CategoryEncoding(num_tokens=vocab_size, output_mode="one_hot")
        self.LSTM=tf.keras.layers.LSTM(dim)
        self.Emb_Dense=tf.keras.layers.Dense(dim)

    def call(self,input,is_first=False):
        
        emb=self.embedding(input,self.vocab_size,self.dim,self.length,is_first)
        
        # emb=self.bert_model({'inputs_embeds' : emb, 'training':True}).last_hidden_state # (b,len,dim)
        emb=self.LSTM(emb) #(b,dim)
        dropout=self.Dropout(emb)
        
        final_output=self.Dense(dropout)
        return final_output #(b,1)

    def sample_gumbel(self,shape, eps=1e-20):
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    def gumbel_softmax(self,logits, temperature, hard=False):
        gumbel_softmax_sample = logits + self.sample_gumbel(tf.shape(logits))
        y = tf.nn.softmax(gumbel_softmax_sample / temperature)
        if hard:
            k = tf.shape(logits)[-1]
            y_hard = tf.cast(tf.equal(y, tf.math.reduce_max(y, 1, keepdims=True)),
                            y.dtype)
            y = tf.stop_gradient(y_hard - y) + y

        return y
    def embedding(self,input,input_vocab_size,d_model,length,is_first=False):
        embedding=input
        if is_first is True:
            #embedding=self.Embedding(embedding) # 이거 근데 one-hot encoding 해야 맞는거 아님?
            embedding=tf.one_hot(embedding,depth=input_vocab_size,on_value=1.0, off_value=0.0,axis=-1)
            # (b,len) -> (b,len,max_vocab)이 될거 아니야.
            # 그리고 의미상으로도 그게 맞고.
            # 그리고 is_first가 false인 녀석은 gumbel softmax로 똑같이 처리해주면 되니깐은.
        else :
            embedding=self.gumbel_softmax(embedding,0.5,hard=True) # gumbel softmax로 one-hot encoding해준 것과 비슷한 결과를 얻는다.
        
        embedding=self.Emb_Dense(embedding) 
        # embedding *= tf.math.sqrt(tf.cast(d_model, tf.float32))
        # embedding += positional_encoding(length,d_model)[:, :length, :] 
        # transformer에 입력으로 들어가는게 아니기 때문에, positional encoding은 사실 필요가 없다.
        
        return embedding

class SparseCategorical_Loss():
    def __init__(self,LAMBDA=1):
        super().__init__()
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.LAMBDA=LAMBDA

    def summary_loss(self,logits, pred): # CUSTOM LOSS.
        mask = tf.math.logical_not(tf.math.equal(logits, 0))

        loss_ = self.loss_object(logits, pred)

        mask = tf.cast(mask, dtype=loss_.dtype) #mask가 되어있던 자리는 학습을 하지 않는다
        loss_ *= mask
        return self.LAMBDA* (tf.reduce_sum(loss_)/tf.reduce_sum(mask))

    def reconstruction_loss(self,real,pred): #fake input과 실제input의 reconstruction loss
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype) #mask가 되어있던 자리는 학습을 하지 않는다
        loss_ *= mask

        return self.LAMBDA*(tf.reduce_sum(loss_)/tf.reduce_sum(mask))

    def summary_accuracy_function(self,summary,pred):
        accuracies = tf.equal(summary, tf.argmax(pred, axis=2,output_type=summary.dtype))

        mask = tf.math.logical_not(tf.math.equal(summary, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


    def reconstruction_accuracy_function(self,real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=2,output_type=real.dtype))# int64가 아니면 type이 다르다고 오류남

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

class DiscriminatorAndGenerator_Loss():
    def __init__(self,ALPHA=0.1):
        super().__init__()
        self.loss_object=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        #self.sparse_loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.ALPHA=ALPHA

    def gen_loss(self, fake_output): # CUSTOM LOSS.
        return self.ALPHA * self.loss_object(tf.ones_like(fake_output), fake_output)

    def critic_loss(self,real_output,fake_output): #fake input과 실제input의 reconstruction loss
        real_loss = self.loss_object(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_object(tf.zeros_like(fake_output), fake_output)
        total_loss = 0.5*(real_loss + fake_loss)

        return self.ALPHA * total_loss
    
    # def calc_cycle_loss(self,real,cycled): # 본래의 이미지 생성 cycle gan은 그냥 단순한 두 이미지 사이의 비교이지만,
    #     # 우리의 경우는 sparse categorical loss를 비교해야 한다.
    #   그런데 사실 우리는 정확히 같은 작업을 AE LOSS에서 이미 해주고 있다!!
    # 따라서 불필요하다.
    #     mask = tf.math.logical_not(tf.math.equal(real, 0))

    #     loss_ = self.sparse_loss_object(real, cycled)

    #     mask = tf.cast(mask, dtype=loss_.dtype) #mask가 되어있던 자리는 학습을 하지 않는다
    #     loss_ *= mask
    #     loss1=tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    #     return self.LAMBDA * loss1
    
    # def identity_loss(self,real, same): # 이것도 cyclegan에서 사용하는 독특한 loss로, 스스로의 도메인을 생성하는 놈에 자기 자신을 넣으면 자기 자신이 나와야 한다.

    #     mask = tf.math.logical_not(tf.math.equal(real, 0))

    #     loss_ = self.sparse_loss_object(real, same)

    #     mask = tf.cast(mask, dtype=loss_.dtype) #mask가 되어있던 자리는 학습을 하지 않는다
    #     loss_ *= mask
    #     loss=tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    #     return self.LAMBDA * 0.5 * loss
    # def enc_disc_accuracy_function(self,real_output,fake_output):
    #     accuracies =tf.equal(fake_output, tf.ones_like(fake_output)) +tf.equal(real_output, tf.ones_like(real_output))
    #     accuracies = tf.cast(accuracies, dtype=tf.float32)

    #     return tf.reduce_sum(accuracies) 이건 accuracy를 측정하기 좀 애매하네...