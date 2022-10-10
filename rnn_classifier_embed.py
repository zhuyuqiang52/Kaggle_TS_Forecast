import torch
import unicodedata
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence,pack_sequence,pack_padded_sequence,pad_packed_sequence
import numpy as np
import os
import string

from rnn_test import input_size_int, hidden_size_int

path_str = './data/names/'
file_list = os.listdir(path_str)
all_ascii_letters = string.ascii_letters+'.,;'
all_ascii_letters.find('f')
n_letter_int = len(all_ascii_letters)
for file in file_list:
    names = open(path_str + file).read().strip().split('\n')

def unicode_to_ascii(s)->string:
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_ascii_letters
    )

def ascii_onehot2word(onehot_array)->str:
    #global name_datset
    word_loc_array = np.argmax(onehot_array,axis=1)
    word_str = ''.join([all_ascii_letters[i] for i in word_loc_array])
    print(f'name 3: {word_str}')
    #loc_int = np.where(name_datset.name_list==word_str)
    #country_label = name_datset.country_label[loc_int]
    #print('countey label :%d'%country_label)
    return word_str

def words2onehot(word_srings):
    words_list = []
    for word_str in word_srings:
        #转换为ascii
        word_ascii_str = unicode_to_ascii(word_str)
        idx_list = []
        for letter in word_ascii_str:
            idx_list.append(all_ascii_letters.find(letter))
        word_tensor = F.one_hot(torch.LongTensor(idx_list),n_letter_int)
        #print(f'name 1: {word_str} \nname 2: {word_ascii_str}')
        #word3_str = ascii_onehot2word(word_tensor.detach().numpy())
        #if word3_str != word_ascii_str:
        #    print(' name incompatiable')
        word_tensor = word_tensor.view(-1,n_letter_int).type(torch.float32)
        words_list.append(word_tensor)
    return words_list

def words2idxs(word_srings):
    words_list = []
    seq_len_list = []
    for word_str in word_srings:
        #转换为ascii
        word_ascii_str = unicode_to_ascii(word_str)
        idx_list = []
        for letter in word_ascii_str:
            idx_list.append(all_ascii_letters.find(letter))
        word_idx_tensor = torch.LongTensor(idx_list)
        words_list.append(word_idx_tensor)
        seq_len_list.append(len(idx_list))
    return words_list,seq_len_list

class name_Dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.path_str = './data/names/'
        self.file_list = os.listdir(self.path_str)
        self.contry_list = [i for i in self.file_list]
        self.country_label = []
        self.name_list = []
        for file_idx in range(len(self.file_list)):
            file_str = self.file_list[file_idx]
            name_tmp_list = open(self.path_str+file_str).read().strip().split('\n')
            #slice of original list
            self.name_list += name_tmp_list
            self.country_label += [file_idx]*len(name_tmp_list)
        self.name_word_list = self.name_list[:]
        self.name_list,self.seq_len_list = words2idxs(self.name_list)
        self.country_label_tensor = torch.LongTensor(self.country_label) #should be LongTensor
        self.name_array = pad_sequence(self.name_list).numpy() #padding
    def __getitem__(self, index):
        return (self.name_array[:,index],self.seq_len_list[index]),self.country_label[index]

    def __len__(self):
        return len(self.country_label)

def dataset_split(test_per_float,obj_dataset):
    size_int = len(obj_dataset)
    test_size_int = int(test_per_float*size_int)
    train_size_int = size_int-test_size_int
    train_dataset,test_dataset = torch.utils.data.random_split(obj_dataset,[train_size_int,test_size_int])
    return train_dataset,test_dataset

def collate_fn(data_tuple):
    X_tuple,label_tuple = zip(*data_tuple)
    X_tuple,seq_len_tuple = zip(*X_tuple)
    seq_len_tensor = torch.Tensor(seq_len_tuple)
    X_tensor = torch.from_numpy(np.array(X_tuple)).T.long()
    label_tensor = torch.LongTensor(label_tuple)
    return (X_tensor,seq_len_tensor),label_tensor

class rnet(nn.Module):
    def __init__(self,input_size_int,hidden_size_int,num_layers_int,batch_size_int,output_size_int,embedding_size_int,bidirection_bool=False):
        super(rnet, self).__init__()
        self.input_size_int = input_size_int
        self.num_layers_int = num_layers_int
        self.batch_size_int = batch_size_int
        self.output_size_int = output_size_int
        self.bidirection_num_int = 2 if bidirection_bool else 1
        self.hidden_size_int = hidden_size_int
        self.rnn = nn.LSTM(embedding_size_int,self.hidden_size_int,self.num_layers_int,bidirectional=bidirection_bool,dropout=0.5)
        self.output_linear = nn.Sequential(
            nn.Linear(self.hidden_size_int*self.bidirection_num_int,self.output_size_int)
        )
        self.embed = nn.Embedding(input_size_int,embedding_size_int)
    def forward(self,x,seq_len_tensor):
        #hidden = torch.zeros(self.num_layers_int,self.batch_size_int,self.hidden_size_int)
        x = self.embed(x)
        x_padded = pack_padded_sequence(x,seq_len_tensor,enforce_sorted=False)
        #test
        h0 = torch.ones(self.num_layers_int*self.bidirection_num_int,self.batch_size_int,self.hidden_size_int)
        c0 = torch.ones(self.num_layers_int*self.bidirection_num_int,self.batch_size_int,self.hidden_size_int)
        output,(h_n,c_n) = self.rnn(x_padded,(h0,c0))
        if self.bidirection_num_int==2:
            hidden = torch.cat([h_n[-1],h_n[-2]],dim=1)
        else:
            hidden = h_n[0]
        output_tensor = self.output_linear(hidden)
        return output_tensor #batch_sizeXnum_class

#hyper-parameters
input_size_int = 55
hidden_size_int = 250
num_layers_int = 2
batch_size_int = 1000
lr_float = 5e-3
epoch_int = 100
bidirectional_bool = True
embedding_size_int = 128
name_datset = name_Dataset()
#split dataset
train_dataset,test_dataset = dataset_split(0.3,name_datset)
train_datloader = DataLoader(train_dataset, batch_size=batch_size_int, shuffle=True, collate_fn=collate_fn,drop_last=True)
test_datloader = DataLoader(test_dataset, batch_size=batch_size_int, shuffle=True, collate_fn=collate_fn,drop_last=True)
output_size_int = len(name_datset.contry_list)

def softmax(input_array):#Batch_size*class_num=
    softmax_array = np.exp(input_array)
    divisor_array = np.sum(softmax_array,axis=1).reshape(-1,1).repeat(softmax_array.shape[1],axis=1)
    softmax_array = softmax_array/divisor_array
    return softmax_array

def train_step(epoch_turn_int,train_dataloader,loss_func,model,optimizer):
    print(f'epoch:{epoch_turn_int + 1}/{epoch_int} ')
    for i,((input,seq_len_tensor),label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(input,seq_len_tensor)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()
        print(f'Batch:{i},loss:{loss.item():.2f}')

def test_step(test_dataloader,model):
    correct_float = 0
    size_int = len(test_dataloader.dataset)
    with torch.no_grad():
        for i, ((input,seq_len_tensor), label) in enumerate(test_dataloader):
            pred = model(input,seq_len_tensor)
            correct_float += (pred.argmax(1) == label).type(torch.float).sum().item()
    correct_float /= size_int
    print(f"Test Error:  Accuracy: {(100 * correct_float):>0.1f}% \n")

def main():
    loss_func = nn.CrossEntropyLoss()
    input_size_int = next(iter(train_datloader))[0][0].shape[1]
    model = rnet(input_size_int, hidden_size_int, num_layers_int, batch_size_int, output_size_int,embedding_size_int,bidirectional_bool)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_float)
    for i in range(epoch_int):
        train_step(i,train_datloader,loss_func,model,optimizer)
        test_step(test_datloader,model)

if __name__=='__main__':
    main()

