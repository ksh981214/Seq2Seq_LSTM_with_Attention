import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk.translate.bleu_score as bleu

class config:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch = 50
    hidden_size = 512
    dim = 512
    lr=0.0001
    optimizer='Adam'

config = config()

def Preprocess(mode='train', src_word2ind=None, trg_word2ind=None):
    if mode == 'train':
        data_path="RNN dataset/eng-fra_train.txt"
        train = open(data_path).read()
        train = train.split('\n')
        data = train
        print("train dataset length is {}".format(len(train)))
    elif mode == 'test':
        data_path="RNN dataset/eng-fra_test.txt"
        test = open(data_path).read()
        test = test.split('\n')
        data = test
        print("test dataset length is {}".format(len(test)))
    
    src = []
    trg = []
    for i, line in enumerate(data):
        if line == '':
            break
        temp = line.split('\t')
        src.append(temp[0])
        trg.append(temp[1])

    if mode == 'train':
        i=2
        src_word2ind = {}
        src_word2ind['<EOS>']=0
        src_word2ind['<UNK>']=1
        for sen in src:
            for word in sen.split():
                if word not in src_word2ind.keys():
                    src_word2ind[word] = i
                    i+=1
        src_ind2word = {}
        for k,v in src_word2ind.items():
            src_ind2word[v]=k

        i=4
        trg_word2ind = {}
        trg_word2ind['<EOS>']=0
        trg_word2ind['<UNK>']=1  
        trg_word2ind['<SOS>']=2  
        trg_word2ind['<BNK>']=3  
        
        for sen in trg:
            for word in sen.split():
                if word not in trg_word2ind.keys():
                    trg_word2ind[word] = i
                    i+=1
        trg_ind2word = {}
        for k,v in trg_word2ind.items():
            trg_ind2word[v]=k

    src_idx = []
    for sen in src:
        new_sen=[]
        for word in sen.split():
            if word in src_word2ind.keys():
                new_sen.append(src_word2ind[word])
            else:
                new_sen.append(src_word2ind['<UNK>'])
        new_sen.append(src_word2ind['<EOS>'])
        src_idx.append(new_sen)

    trg_idx = []
    for sen in trg:
        new_sen=[trg_word2ind['<SOS>']]
        for word in sen.split():
            if word in trg_word2ind.keys():
                new_sen.append(trg_word2ind[word])
            else:
                new_sen.append(trg_word2ind['<UNK>'])
        new_sen.append(trg_word2ind['<EOS>'])
        trg_idx.append(new_sen)
    
    nosos_trg_idx = []
    for sen in trg:
        new_sen=[]
        for word in sen.split():
            if word in trg_word2ind.keys():
                new_sen.append(trg_word2ind[word])
            else:
                new_sen.append(trg_word2ind['<UNK>'])
        new_sen.append(trg_word2ind['<EOS>'])
        new_sen.append(trg_word2ind['<BNK>'])
        nosos_trg_idx.append(new_sen)
    

    if mode == 'train':
        print("src vocab: {}".format(len(src_word2ind)))
        print("trg vocab: {}".format(len(trg_word2ind)))
        return src_word2ind, src_ind2word, trg_word2ind, trg_ind2word, src_idx, trg_idx, nosos_trg_idx

    elif mode == 'test':
        print("test src length: {}".format(len(src_idx)))
        print("test trg length: {}".format(len(trg_idx)))
        return src_idx, trg_idx
    
class Seq2Seq_LSTM(nn.Module):
    def __init__(self, src_word2ind, trg_word2ind, dim=config.dim, hidden_size = config.hidden_size, attention=False):
        super().__init__()
        self.src_word2ind = src_word2ind
        self.trg_word2ind = trg_word2ind
        
        self.src_embedding = nn.Embedding(len(src_word2ind),dim) 
        self.trg_embedding = nn.Embedding(len(trg_word2ind),dim)

        self.C = len(trg_word2ind)

        self.h2 = int(hidden_size/4)

        self.Wf = nn.Linear(dim+self.h2, self.h2, bias=True)
        self.Wi = nn.Linear(dim+self.h2, self.h2, bias=True)
        self.Wo = nn.Linear(dim+self.h2, self.h2, bias=True)
        self.Wg = nn.Linear(dim+self.h2, self.h2, bias=True)
        
        self.attention = attention
        if self.attention:
            print("Attention Mode")
            self.Wy = nn.Linear(self.h2+self.h2, len(trg_word2ind), bias=True)
        else:    
            self.Wy = nn.Linear(self.h2, len(trg_word2ind), bias=True)

    def encoder(self, x):
        '''
        x: (sen_len, dim)
        xt: (dim)
        ht_1: (hidden_size/4)
        concat(xt, ht_1) --> (bs, dim + h2)
        ct_1: (hidden_size/4)
        '''
        sen_len = x.size()[0]
        if self.attention:
            enc_ht = []

        ht_1 = torch.zeros(self.h2).to(config.device)
        ct_1 = torch.zeros(self.h2).to(config.device)
        for i in range(sen_len):
            if self.attention:
                enc_ht.append(ht_1)
            xt = x[i]
            concat = torch.cat((xt, ht_1))
            ft = torch.sigmoid(self.Wf(concat)) # (dim+h2)x(dim+h2,h2) = h2
            it = torch.sigmoid(self.Wi(concat))
            ot = torch.sigmoid(self.Wo(concat))
            gt = torch.tanh(self.Wg(concat))
            ct = ft*ct_1 + it*gt                #new cell states, h/4
            ht = ot*torch.tanh(ct)              #h2
            ct_1 = ct
            ht_1 = ht                           #bs, 2

        if self.attention:
            return ct_1, ht_1, enc_ht 
        else:
            return ct_1, ht_1


    def decoder(self, x, encoder_ct_1, encoder_ht_1, mode='train', enc_ht = None):
        '''
        x: (sen_len, dim)
        xt: (dim)
        ht_1: (hidden_size/4)
        ct_1: (hidden_size/4)
        enc_ht: (sen_len, h2)
        '''
        sen_len = x.size()[0]
        if mode == 'train':
            prob = torch.zeros([sen_len, self.C])
            ht_1 = encoder_ht_1
            ct_1 = encoder_ct_1
            for i in range(sen_len):
                xt = x[i] 
                concat = torch.cat((xt, ht_1))
                ft = torch.sigmoid(self.Wf(concat)) 
                it = torch.sigmoid(self.Wi(concat))
                ot = torch.sigmoid(self.Wo(concat))
                gt = torch.tanh(self.Wg(concat))
                ct = ft*ct_1 + it*gt                    #new cell states, h2
                ht = ot*torch.tanh(ct)                  #h2

                if self.attention:
                    score = []
                    for h in enc_ht:
                        temp = torch.dot(ht, h.detach())
                        score.append(temp)              #sen_len, e^t
                    
                    score = torch.softmax(torch.tensor(score), dim=0).to(config.device)         #sen_len, attention distribution
                    attention_vector = torch.zeros(score.size()[0], self.h2).to(config.device)  #sen_len, h2
                    for idx,s in enumerate(score):
                        attention_vector[idx] = s* (enc_ht[idx].detach())                       #pointwise product,(sen_len) (sen_len, h2)
                    attention_vector = torch.sum(attention_vector, dim =0).to(config.device)    #h2
                    yt = self.Wy(torch.cat((ht, attention_vector)))
                else:
                    yt = self.Wy(ht)                    #(h2)x(h2,C) = C
                pt = torch.softmax(yt,dim=0)
                prob[i] = pt
                ct_1 = ct
                ht_1 = ht
            return prob                                 #bs, max_len, trg_vocab_size


        elif mode == 'test':
            word_idxs = [] 
            ans_word_idxs = []

            ht_1 = encoder_ht_1
            ct_1 = encoder_ct_1
            xt = x[0]                               #SOS
            for i in range(sen_len):

                concat = torch.cat((xt, ht_1))
                ft = torch.sigmoid(self.Wf(concat)) # (dim)x(dim,h2) =  h2
                it = torch.sigmoid(self.Wi(concat))
                ot = torch.sigmoid(self.Wo(concat))
                gt = torch.tanh(self.Wg(concat))
                ct = ft*ct_1 + it*gt                #new cell states, bs, h2
                ht = ot*torch.tanh(ct)              #bs, h2

                if self.attention:
                    score = []
                    for h in enc_ht:
                        temp = torch.dot(ht, h.detach())
                        score.append(temp)                                                      #sen_len, e^t
                    score = torch.softmax(torch.tensor(score), dim=0).to(config.device)         #sen_len, attention distribution
                    attention_vector = torch.zeros(score.size()[0], self.h2).to(config.device)  #sen_len, h2
                    for idx,s in enumerate(score):
                        attention_vector[idx] = s* (enc_ht[idx].detach())                       #pointwise product,(sen_len) (sen_len, h2)
                    attention_vector = torch.sum(attention_vector, dim =0).to(config.device)    #h2
                    yt = self.Wy(torch.cat((ht, attention_vector)))
                else:
                    yt = self.Wy(ht)                                                            #(h2)x(h2,C) = C

                pred_word_idx = torch.argmax(yt)
                if pred_word_idx == 0:
                    break
                word_idxs.append(pred_word_idx.item()) 

                xt = self.trg_embedding(pred_word_idx)

                ct_1 = ct
                ht_1 = ht
            return word_idxs

    def forward(self, embedded_src, embedded_trg,mode='train'):
        if self.attention:
            ct_1, ht_1, enc_ht = self.encoder(embedded_src)
            if mode == 'train':
                prob = self.decoder(embedded_trg, ct_1, ht_1, mode, enc_ht)
                return prob
            elif mode =='test':
                word_idxs = self.decoder(embedded_trg, ct_1, ht_1, mode, enc_ht)
                return word_idxs
        else:
            ct_1, ht_1 = self.encoder(embedded_src)
            if mode == 'train':
                prob = self.decoder(embedded_trg, ct_1, ht_1, mode)
                return prob
            elif mode =='test':
                word_idxs = self.decoder(embedded_trg, ct_1, ht_1, mode)
                return word_idxs
            
def train(model, train_src_idx, train_trg_idx, train_nosos_trg_idx, epochs):
    '''
        train_src_embedded: (6670, max_len, dim)
        train_trg_idx: (6670,max_len)
    '''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optim = torch.optim.Adam(model.parameters(),lr = config.lr)

    model.train()
    start = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        for i, sen in enumerate(train_src_idx):    
            optim.zero_grad()
            src = torch.tensor(sen).to(config.device)
            trg = torch.tensor(train_trg_idx[i]).to(config.device)
            preds = model(model.src_embedding(src), model.trg_embedding(trg))
            loss= F.cross_entropy(preds.to(config.device), torch.tensor(train_nosos_trg_idx[i]).to(config.device))                 

            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            if i%1000==0:
                print("{}th iter , loss:{}".format(i+1, epoch_loss/(i+1)))
        loss_avg = epoch_loss
        print("time = %dm, epoch %d, loss = %.3f, %ds per epoch" % ((time.time() - start) // 60, epoch + 1, loss_avg, (time.time() - start)/(epoch+1)))

        
def restore_sentence(idxs, ind2word):
    sen = []
    for idx in idxs:
        word = ind2word[idx]
        #print(word)
        if word =='<BNK>':
            break
        else:
            sen.append(word)
    return sen

def test(model):
    model.eval()
    test_src_idx, test_trg_idx = Preprocess('test', model.src_word2ind, model.trg_word2ind)
    #_, _,_, _, test_src_idx, test_trg_idx, _ = Preprocess('train')
    bleu_ans=[]
    bleu_pred=[]
    for i, test in enumerate(test_src_idx):
        src = torch.tensor(test).to(config.device).clone().detach()
        trg = torch.tensor(test_trg_idx[i]).to(config.device).clone().detach()

        preds= model(model.src_embedding(src).to(config.device), model.trg_embedding(trg).to(config.device), mode='test')
        src_sen = restore_sentence(test_src_idx[i], src_ind2word)
        trg_sen = restore_sentence(test_nosos_trg_idx[i], trg_ind2word)

        preds_sen = restore_sentence(preds, trg_ind2word)
        bleu_ans.append(trg_sen[:-1])
        bleu_pred.append(preds_sen)
        #print("영어 문장: {}".format(src_sen))
        #print("정답 문장: {}".format(trg_sen))
        #print("예측 문장: {}".format(preds_sen))
    
    print("bleu_score:{}".format(bleu.corpus_bleu(bleu_pred, bleu_ans)))

def main():
    src_word2ind, src_ind2word, trg_word2ind, trg_ind2word, train_src_idx, train_trg_idx, train_nosos_trg_idx = Preprocess('train')

    att_model = Seq2Seq_LSTM(src_word2ind, trg_word2ind, attention=True).to(config.device)
    
    #noatt_model = Seq2Seq_LSTM(src_word2ind, trg_word2ind, attention=False).to(config.device)
    
    train(att_model, train_src_idx, train_trg_idx, train_nosos_trg_idx, config.epoch)
    
    test(att_model)
    
    
main()