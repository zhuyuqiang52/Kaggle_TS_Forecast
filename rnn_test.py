import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torch.nn.functional as F
import numpy as np

class recursive_net(nn.Module):
    def __init__(self,batch_size_int,input_size_int,hidden_size_int):
        super().__init__()
        self.input_size_int = input_size_int
        self.hidden_size_int = hidden_size_int
        self.batch_size_int = batch_size_int
        self.rnn_cell = nn.RNNCell(self.input_size_int,self.hidden_size_int)

    def forward(self, x,hidden):
        hidden = self.rnn_cell(x,hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size_int,self.hidden_size_int)

class single_rnn(nn.Module):
    def __init__(self, batch_size_int, input_size_int, hidden_size_int):
        super().__init__()
        self.seq_len_int = seq_len_int
        self.input_size_int = input_size_int
        self.hidden_size_int = hidden_size_int
        self.batch_size_int = batch_size_int
        self.num_layer_int = num_layer_int
        self.rnn = nn.RNN(self.input_size_int, self.hidden_size_int,self.num_layer_int)

    def forward(self, x):
        hidden = torch.zeros(self.num_layer_int,self.batch_size_int, self.hidden_size_int)
        hidden,_ = self.rnn(x, hidden)
        return hidden.view(-1,self.hidden_size_int)

input_str = 'hellohellohellohel'
opt_str = 'ohlollolohlolohlol'
input_list = list(input_str)
opt_list = list(opt_str)

#dict for transformming char to int
char_dict = {'h':0,'e':1,'l':2,'o':3}
char_array = np.array(list(char_dict.keys()))
input_int_list = [char_dict[i] for i in input_list]
opt_int_list = [char_dict[i] for i in opt_list]
#hyper-parameters
lr_float = 1e-2
input_size_int = 4
hidden_size_int = 4
batch_size_int = 3
epoch_int = 1000
seq_len_int = len(input_str)//batch_size_int
num_layer_int = 1
# one-hot
input_onehot_tensor = F.one_hot(torch.tensor(input_int_list))
input_onehot_tensor = input_onehot_tensor.view(-1,batch_size_int,input_size_int)
input_onehot_tensor = input_onehot_tensor.type(torch.float32)
#label
label_tensor = torch.LongTensor(opt_int_list).view(-1,batch_size_int) #batch size is here
def main():
    loss_func = nn.CrossEntropyLoss()
    model = recursive_net(batch_size_int,input_size_int,hidden_size_int)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr_float)
    for i in range(epoch_int):
        optimizer.zero_grad()
        hidden_tensor = model.init_hidden()
        loss = 0
        for input,label in zip(input_onehot_tensor,label_tensor):
            hidden_tensor = model(input,hidden_tensor)
            loss +=loss_func(hidden_tensor,label)
            _,idx = hidden_tensor.max(dim=1)
            print(''.join(char_array[idx.numpy()]),end='')
        loss.backward()
        optimizer.step()
        print(f',Epoch[{i+1}/{epoch_int}],loss:{loss.item():.2f}')

def main_rnn():
    loss_func = nn.CrossEntropyLoss()
    model = single_rnn(batch_size_int,input_size_int, hidden_size_int)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_float)
    for i in range(epoch_int):
        optimizer.zero_grad()
        hidden_tensor = model(input_onehot_tensor)
        loss = loss_func(hidden_tensor,label_tensor.view(1,-1)[0])
        _, idx = hidden_tensor.max(dim=1)
        print(''.join(char_array[idx.numpy()]), end='')
        loss.backward()
        optimizer.step()
        print(f',Epoch[{i + 1}/{epoch_int}],loss:{loss.item():.2f}')

if __name__=='__main__':
    main_rnn()

