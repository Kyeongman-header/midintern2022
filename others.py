import os
import tensorflow as tf
import consts


def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error Creating directory. ' + directory)

def padder(inp1,inp2,t5_tokenizer,BATCH_SIZE=consts.BATCH_SIZE):
    pad_token=t5_tokenizer('<pad>')['input_ids'][:-1]
    pad_tokens=[]
    for i in range(BATCH_SIZE):
        pad_tokens.append(pad_token)

    pad_tokens=tf.convert_to_tensor(pad_tokens,dtype=tf.int64)
    end=t5_tokenizer('</s>', return_tensors="tf")['input_ids'][0][0]

    pad_inp1=tf.concat([pad_tokens,inp1[:,:-1]],axis=-1)

    pad_inp2=tf.concat([pad_tokens,inp2[:,:-1]],axis=-1)

    return pad_tokens,end,pad_inp1,pad_inp2,inp1,inp2
    # 순서대로 start tokens, end token, decoder input으로서의 input, summary, loss 의 logit으로서의 inp, summary이다.


def ForTest(my_encoder,my_enc_disc,my_decoder,my_dec_disc,ae_loss,gan_loss,BATCH_SIZE,SHORT_MAX,LONG_MAX,t5_tokenizer):
    #테스트용 데이터.
    start_tokens,end_token,pad_sum,pad_inp,loss_sum,loss_inp=padder(tf.random.uniform((BATCH_SIZE, SHORT_MAX), dtype=tf.int64, minval=0, maxval=200),tf.random.uniform((BATCH_SIZE, LONG_MAX), dtype=tf.int64, minval=0, maxval=200),t5_tokenizer=t5_tokenizer)
    enc_output=my_encoder([pad_inp,pad_sum],starts=start_tokens,end=end_token,teacher=False,is_first=True)
    t2=my_enc_disc(enc_output,is_first=False) # fake output
    t3=my_enc_disc(loss_sum,is_first=True) # real output
    dec_output=my_decoder([enc_output,pad_inp],starts=start_tokens,end=end_token,teacher=False,is_first=False)
    t5=my_dec_disc(dec_output,is_first=False) #fake output
    t6=my_dec_disc(loss_inp,is_first=True) # real output

    cycle_enc_output=my_encoder([dec_output,pad_sum],starts=start_tokens,end=end_token,teacher=True,is_first=False)
    # 여기는 원래 dec_output을 이용한 게 아니라 real summary를 넣어야 하는게 맞다.
    # 허나 우리는 문학에 한해서 real summary가 존재하지 않는다. 우리는 엄연히 domain 변화라기 보다는 summary를 하려는 것이다.
    # 따라서 먼저 pretrain 시킬 때에는 summary pair로 만들어낸 cycle enc output을 만들어야 할 것이고
    # summary가 없는 문학에 대해서는 enc output이 만들어낸 cycle enc output을 이용하며, 또한 encoder는 반드시 teacher=False로 학습해야 한다. 

    # same_sum=my_encoder([pad_sum,pad_sum],starts=start_tokens,end=end_token,teacher=True,is_first=True) # same domain
    # same_inp=my_decoder([pad_inp,pad_inp],starts=start_tokens,end=end_token,teacher=True,is_first=True) # same domain
    # 여기는 애초에 embedding을 해줘야 해서리 차원이 달라지면 안됨...
    # 우리는 identity loss 를 사용할 수 없는 대신 ae loss를 사용하기 때문에
    # 어느정도 커버가 되지 않을까 싶다.
    print(ae_loss.summary_loss(loss_sum,enc_output))
    print(ae_loss.summary_accuracy_function(loss_sum,enc_output)) # 이 부분은 원 논문에는 없다.

    print(gan_loss.gen_loss(t2))
    print(gan_loss.critic_loss(t3,t2)) # 항상 real 데이터가 먼저임.

    print(ae_loss.reconstruction_loss(loss_inp,dec_output))
    print(ae_loss.reconstruction_accuracy_function(loss_inp,dec_output))
    print(gan_loss.gen_loss(t5))
    print(gan_loss.critic_loss(t6,t5))

    print(ae_loss.summary_loss(loss_sum,cycle_enc_output)) 
    print(ae_loss.summary_accuracy_function(loss_sum,cycle_enc_output))

def model_saver(encoder,decoder,enc_disc,dec_disc,enc_optimizer,dec_optimizer,enc_disc_optimizer,dec_disc_optimizer,file_name='cyclegan'):
    checkpoint_path = "./MY_checkpoints/train_"+file_name

    ckpt = tf.train.Checkpoint(encoder=encoder,decoder=decoder,enc_disc=enc_disc,dec_disc=dec_disc,
                            enc_optimizer=enc_optimizer,dec_optimizer=dec_optimizer,
                            enc_disc_optimizer=enc_disc_optimizer, dec_disc_optimizer=dec_disc_optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=None)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    else :
        print('This is New train!!(no checkpoint)')
    
    return ckpt_manager