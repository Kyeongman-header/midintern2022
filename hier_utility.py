
from rouge import Rouge
import tqdm
import tensorflow as tf
rouge=Rouge()

def prefix_ver1(val_sum):
    
    # print(val_sum)
    return "The most abstract summary is : " + val_sum + " and the plot is : "
    
def prefix_ver2(val_sum,mother_plot, i):
    
    return "The most abstract summary is : " + mother_plot + " and the plot is : " + val_sum + " and the plot's sequence is " + str(i) + " and the original text is : "
    

def generate_valid(model,valid_summary,wr,epoch,tokenizer,val_inp,prefix,mother_plots=[]):
    c=0

    
    wr.writerow(['this is ' + str(epoch) + 'epoch generation task.'])
    r_1_avg=0
    r_2_avg=0
    r_l_avg=0
    outputs=[]
    for val_sum in valid_summary:

        
        
        if len(mother_plots)!= 0:
            
            mother_plot=mother_plots[int(math.floor(c/5))]
        # 이거로 해야 함.

        # 여기 이 tokenizer 부분은 지워야함
            #mother_plot=tokenizer.decode(mother_plots[int(math.floor(c/5))],skip_special_tokens=True)
            val_sum=prefix(val_sum,mother_plot,c%5)
        # c가 0~4까지는 c/5는 0이다. 5~9까지 c/5는 1이다. 이렇게 mother_plot을 5개씩 매칭한다.
            #페이지는 0~4까지 범위를 가진다
        
        else :
            #val_sum=tokenizer.decode(val_sum,skip_special_tokens=True)
        # 잘못 만들었음... 여기 tokenizer 부분은 지워야 함
            val_sum=prefix(val_sum)
        
        print("val summary ===> " + val_sum) # prefix 부분이다.
        # ~~~!! 적절한 prefix를 붙여줘야 한다!!
        input_ids = tokenizer.encode(val_sum, return_tensors='tf')
        # max length가 1024지, 실제로 1024개를 생성하는 경우는 없을 거다.
        
        output=""
        while len(output)==0: # length가 매우 드물게 0인 경우가 있는데 이럴 경우 다시 생성한다.
            output = tokenizer.decode(model.generate(input_ids,max_length = 1000,do_sample=True,top_p=0.92,top_k=50,early_stopping=True)[0],skip_special_tokens=True) # no repeat은 dumb repeat을 방지할 수도 있다!
        
        
        original=tokenizer.decode(val_inp[c],skip_special_tokens=True)
        
        # ~~~!! prefix는 없애줘야 한다!
        val_sum=val_sum.replace(" ,",",")
        val_sum=val_sum.replace(" .",".")
        val_sum=val_sum.replace(" '","'")
        val_sum=val_sum.replace(' "','"')
        val_sum=val_sum.replace(" n't","n't")
        val_sum=val_sum.replace(u'\xa0',u'')
        origianl=original.replace(" ,",",")
        original=original.replace(" .",".")
        original=original.replace(" '",".")
        original=original.replace(' "','"')
        original=original.replace(" n't","n't")
        original=original.replace(u'\xa0',u'')
        output=output.replace(" ,",",")
        output=output.replace(" .",".")
        output=output.replace(" '","'")
        output=output.replace(' "','"')
        output=output.replace(" n't","n't")
        output=output.replace(u'\xa0',u'')

        
        #print("val_sum : "+val_sum)
        print("is val_sum exist : " )
        print(output.find(val_sum)>=0) # False가 나오면 뭔가 이상한 거다


        output=output.replace(val_sum,"")
        
        original=original.replace(val_sum,"") # prefix 부분은 없애준다.
        print("original : " + original)
        print("len of original : " + str(len(tokenizer(original).input_ids)))
        print()
        print("output : " + output)
        print("len of output : " + str(len(tokenizer(output).input_ids)))
        
        outputs.append(output) # prefix부분은 빠진 output.

        if len(output)==0:
            r_1=0
            r_2=0
            r_l=0
        else:
            r=rouge.get_scores(output,original,avg=True)
            #bleu=sentence_bleu([original],output)
            r_1=r['rouge-1']['r']
            r_2=r['rouge-2']['r']
            r_l=r['rouge-l']['r'] # recall이 맞을 것 같다.(reference 길이가 분모가 됨.)
        
        r_1_avg=r_1_avg+r_1
        r_2_avg=r_2_avg+r_2
        r_l_avg=r_l_avg+r_l
        c=c+1
        print(r_1)
        print(r_2)
        print(r_l)
        wr.writerow([original,val_sum,output,r_1,r_2,r_l])
    # 되는 것 확인.
    


    #encodings=tf.reshape(valid_summary,[1,-1]) # 지우고 아래 주석으로 해야함
    
    # 여기서부턴 한꺼번에 perplexity 계산.
    #encodings=tokenizer("\n\n".join(valid_summary),return_tensors="tf")
    val_inp=tokenizer.batch_decode(val_inp,skip_special_tokens=True)
    encodings=tokenizer("\n\n".join(val_inp),return_tensors="tf").input_ids
    print("encoding input shape : " )
    #print(encodings.input_ids.shape) # (1,287664가 됨!! 여러 글이 한 덩어리로 합쳐짐.)
    
    print(encodings.shape) # 지우고 위의 주석으로 해야함

    max_length = model.config.n_positions
    stride = 512
    #size=encodings.input_ids.shape[1] 
    size=encodings.shape[1] # 지우고 위의 주석으로 해야함
    nlls = []
    print("model's output max_length : " + str(max_length))
    #print("input id size : " + str(encodings.input_ids.shape[1])) 
    
    print("input id size : " + str(encodings.shape[1])) # 지우고 위의 주석으로 해야함
    
    print("stride : " + str(stride))
    prev_end_loc=0

    # input()

    for begin_loc in (range(0, size, stride)): # (0~287644)를 512씩 움직인다. 이 경우 
        end_loc = min(begin_loc + max_length, size) 
                # print("begin_loc : " + str(begin_loc))
                # print("end_loc : " + str(end_loc))


        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        if trg_len==0:
            continue
        #input_ids = encodings.input_ids[:, begin_loc:end_loc] # begin loc은 512씩 가는데 end_loc은 거기서부터 1024까지의 길이이기때문에
        
        input_ids = encodings[:, begin_loc:end_loc] # 지우고 위의 주석으로 해야함


                # 0~1024, 512~1536, 1024~2048 이런식으로 input은 겹치면서 나아간다
                # 고로 end에서 1024만큼 뺀 부분은 -100으로 target을 설정해 줌으로써 그부분의 loss는 계산에서 제외할 수 있대(어째서지...)
                # 그니까, 앞의 512개의 context도 전부 보면서, 뒤의 512개의 생성에 대한 perplexity를 계산하겠다는 거지. 
                # 그 와중에 앞의 512개의 perplexity를 또 계산하면 중복되니깐 그 부분은 -100으로 설정해서 없애겠다는거고.
                # 이 stride가 작을 수록 perplexity는 작아짐(좋아짐.) 왜냐하면 겹치면서 보는 context가 많아지니까?
                # print("input_ids : ") # 2차원임[[]]
                # print(input_ids.shape)
                # print("trg_len : " + str(trg_len))

                # target_ids[:, :-trg_len] = -100
        py_target=[]
        for i in range(input_ids.shape[1]):
            if i<input_ids.shape[1]-trg_len:
                py_target.append(-100)
            else:
                py_target.append(input_ids[0][i])


        target_ids=tf.stack([py_target])

                # print("target_ids : ")
                # print(target_ids.shape) # [[]]
                # input()

        loss = model(input_ids, labels=target_ids).loss
                # print("outputs : ")
                # print(outputs[0])
                # print(outputs.loss)
        neg_log_likelihood = loss * trg_len

                # print("neg_log_likelihood : ")
                # print(neg_log_likelihood)
        prev_end_loc = end_loc
        nlls.append(neg_log_likelihood)

    ppl = tf.exp(tf.reduce_sum(tf.stack(nlls)) / end_loc)

    print("ppl is : " + str(ppl))

    
    wr.writerow(['this is avg rouge of generation: ' + str(r_1_avg/c) + " =r1 , " + str(r_2_avg/c) + " =r2 , " + str(r_l_avg/c) + " =rl , "])
    wr.writerow(['this is avg perplexity of generation : ' + str(ppl)])   
    return r_1_avg/c , r_2_avg/c ,r_l_avg/c , ppl, outputs


import math
def splitting_output(outputs,tokenizer):
    five_split_outputs=[]
    for o in outputs:
        w=round(len(tokenizer(o).input_ids)/5)
        sl=len(o.split('.'))
        seq=[0]
        count=0
        l=0
        for split in o.split('.'):
            count=count+1
            if len(seq)>=6:
                # 5 덩어리에서 끊는다.
                continue
            tokens=tokenizer(split).input_ids
            l=l+len(tokens)
            if l>=w:
                seq.append(count)
                l=len(tokens)
        seq.append(count)
        
        #print("before")
        #print(seq)
        """ 
        if len(seq)>6:#6덩어리가 있는 상태.
            dist=seq[-1]-seq[-2]+1
            dist=math.ceil(dist/5)

            while seq[5]<sl:
                seq[1]+=dist
                seq[2]+=dist*2
                seq[3]+=dist*3
                seq[4]+=dist*4
                seq[5]+=dist*5
        """
        if len(seq)>=7: # 이게 그니까 한 문장의 길이랑 whole/5 의 길이가 비슷할 때 6덩어리가 나오는 것 같다.
            # 그래서 최소 2000 words 이상에서 작업하는 hier는 6덩어리가 나올 일이 아예 없었던거다.
            # 그러나 지금은 막 150단어짜리도 나오고 그러다보니깐 5로 나눈게 한 문장 길이랑 거의 비슷한 수준이다.
            # 그러다보니 끝에 따라지 6번째 덩어리가 나오게 된다.
            # 긴 output에 대해서는 이러한 현상이 없기 때문에, 간단히 seq[6], 즉 마지막을 seq[5]로 pull한다.
            seq[5]=seq[6]
        
        #print("after")
        #print(seq)
        
        #print("whole : ")
        #print(o)
        if len(seq)<6: # 도대체 어떤 이유인지 짐작하기 어렵지만, 만약 5덩어리보다 작게 만들어졌다면,
            # (아마도 생성된 결과가 몹시 짧을때?? 잘 모르겠다)
            # 그냥 문장 단위로 자르는 것을 포기하고 단어 단위로 잘라버린다.
            print("during splitting, the seq length is under 6. words split start.")
            cut=round(len(o.split(' '))/5)
            
            mt=o.split(' ')[0:cut]
            five_split_outputs.append(mt)
            print("each middel target token length : " + str(len(tokenizer(mt).input_ids)))
            print(mt)
            
            mt=o.split(' ')[cut: 2* cut]
            five_split_outputs.append(mt)
            print("each middel target token length : " + str(len(tokenizer(mt).input_ids)))
            print(mt)
            
            mt=o.split(' ')[2*cut:3*cut]
            five_split_outputs.append(mt)
            print("each middel target token length : " + str(len(tokenizer(mt).input_ids)))
            print(mt)

            mt=o.split(' ')[3*cut:4*cut]
            five_split_outputs.append(mt)
            print("each middel target token length : " + str(len(tokenizer(mt).input_ids)))
            print(mt)

            mt=o.split(' ')[4*cut:]
            five_split_outputs.append(mt)
            print("each middel target token length : " + str(len(tokenizer(mt).input_ids)))
            print(mt)

            return five_split_outputs

        for i in range(6): # 무조건 5개씩만 한다(어떤 연유로, 예를 들어 너무 짧은 output이었다거나, 해서 이상한 오류가 나도 5덩어리만 본다)
            if i>0:
                mt=('.').join(o.split('.')[seq[i-1]:seq[i]])
                mt=mt+"."
                print("each middel target token length : " + str(len(tokenizer(mt).input_ids)))
                print(mt)
                five_split_outputs.append(mt)
    return five_split_outputs
