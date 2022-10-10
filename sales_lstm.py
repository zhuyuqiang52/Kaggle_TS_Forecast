import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data.dataset import Dataset


train_df = pd.read_pickle('./data/sale_forcast/features_train.pkl')
test_df = pd.read_pickle('./data/sale_forcast/features_test.pkl')

def transform_data(raw_data_df,seq_len_int):
    raw_data_df.reset_index(inplace=True)
    raw_data_df = raw_data_df.sort_values(['store_nbr','family','date'])
    days_int = int(len(raw_data_df)/(33*54))
    pure_feature_df = raw_data_df.iloc[:,4:]
    dat_num_onetype_int = days_int-seq_len_int+1
    dat_array = np.zeros(shape=(dat_num_onetype_int*33*54,seq_len_int,29))
    num_ind_int = 0
    for row_i in range(0,len(raw_data_df),days_int):
        store_family_array = pure_feature_df.iloc[row_i:row_i+days_int,:].values
        row_ind_int = 0
        while(row_ind_int<dat_num_onetype_int):
            dat_array[num_ind_int,:,:] = store_family_array[row_ind_int:row_ind_int+seq_len_int,:]
            num_ind_int+=1
            row_ind_int+=1
    return dat_array

def dataset_split(test_per_float,obj_dataset):
    size_int = len(obj_dataset)
    test_size_int = int(test_per_float*size_int)
    train_size_int = size_int-test_size_int
    train_dataset,test_dataset = torch.utils.data.random_split(obj_dataset,[train_size_int,test_size_int])
    return train_dataset,test_dataset

class sale_dataset(Dataset):
    def __init__(self,dat_df,seq_len_int):
        super().__init__()
        dat_array = transform_data(dat_df,seq_len_int)
        dat_tensor = torch.Tensor(dat_array)
        self.label_tensor = dat_tensor[:,:,0].type(torch.LongTensor)
        self.feature_tensor = dat_tensor[:,:,1:].type(torch.float32)
        self.dat_len_int = len(self.label_tensor)

    def __getitem__(self,idx_int):
        return self.label_tensor[idx_int],self.feature_tensor[idx_int,:]

    def __len__(self):
        return self.dat_len_int

class rnet(nn.Module):
    def __init__(self,input_size_int,hidden_size_int,num_layers_int,batch_size_int,output_size_int,bidirection_bool=False):
        super(rnet, self).__init__()
        self.input_size_int = input_size_int
        self.num_layers_int = num_layers_int
        self.batch_size_int = batch_size_int
        self.output_size_int = output_size_int
        self.bidirection_num_int = 2 if bidirection_bool else 1
        self.hidden_size_int = hidden_size_int
        self.rnn = nn.LSTM(input_size_int,self.hidden_size_int,self.num_layers_int,bidirectional=bidirection_bool)
        self.output_linear = nn.Sequential(
            nn.Linear(self.hidden_size_int*self.bidirection_num_int,self.output_size_int),
            nn.ReLU()
        )
    def forward(self,x):
        #hidden = torch.zeros(self.num_layers_int,self.batch_size_int,self.hidden_size_int)
        #test
        h0 = torch.ones(self.num_layers_int*self.bidirection_num_int,self.batch_size_int,self.hidden_size_int)
        c0 = torch.ones(self.num_layers_int*self.bidirection_num_int,self.batch_size_int,self.hidden_size_int)
        output,(h_n,c_n) = self.rnn(x,(h0,c0))
        output_tensor = self.output_linear(output)
        output_tensor = output_tensor.squeeze()
        return output_tensor #batch_sizeXnum_class

def rmsle(pred_tensor,label_tensor):
    log_error = torch.log1p(pred_tensor)-torch.log1p(label_tensor)
    rmsle_tenfloat = torch.sqrt(torch.mean(torch.pow(log_error,2)))
    return rmsle_tenfloat

#hyper-parameters
seq_len_int = 7
input_size_int = 28
hidden_size_int = 250
num_layers_int = 2
batch_size_int = 10000
lr_float = 1e-1
epoch_int = 100
bidirectional_bool = False
sale_dataset = sale_dataset(train_df,seq_len_int)
train_dataset,test_dataset = dataset_split(0.3,sale_dataset)
train_datloader = DataLoader(train_dataset, batch_size=batch_size_int, shuffle=True, drop_last=True)
test_datloader = DataLoader(test_dataset, batch_size=batch_size_int, shuffle=True,drop_last=True)
output_size_int = 1
def train_loop(model,dataloader,loss_func,optimizer,epoch_turn_int):
    print(f'epoch:{epoch_turn_int + 1}/{epoch_int} ')
    for i,(label_tensor,X_tensor) in enumerate(dataloader):
        X_tensor = X_tensor.permute(1,0,2)
        label_tensor = label_tensor.permute(1, 0)
        optimizer.zero_grad()
        pred_tensor = model(X_tensor)
        loss = loss_func(pred_tensor,label_tensor)
        loss.backward()
        optimizer.step()
        print(f'Batch:{i},loss:{loss.item():.2f}')

def test_loop(model,dataloader,loss_func):
    loss_list = []
    with torch.no_grad():
        for i,(label_tensor,X_tensor) in enumerate(dataloader):
            X_tensor = X_tensor.permute(1, 0, 2)
            label_tensor = label_tensor.permute(1, 0)
            pred_tensor = model(X_tensor)
            loss = loss_func(pred_tensor,label_tensor)
            loss_list.append(loss.item())
        print(f'test RMSLE: {np.mean(loss_list):.2f}')

def main():
    model = rnet(input_size_int, hidden_size_int, num_layers_int, batch_size_int, output_size_int,bidirectional_bool)
    optimizer = torch.optim.Adam(model.parameters(),lr = lr_float)
    loss_func = rmsle
    for i in range(epoch_int):
        train_loop(model,train_datloader,loss_func,optimizer,i)
        test_loop(model,test_datloader,loss_func)

if __name__=='__main__':
    main()