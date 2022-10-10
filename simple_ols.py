import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import numpy as np

class ols_nn(nn.Module):
    def __init__(self):
        super(ols_nn, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(5,1)
        )
    def forward(self,x):
        fit_val = self.linear_stack(x)
        return fit_val

def loss_func(y_pred_tensor,y_tensor)->torch.Tensor:
    return  torch.mean((y_pred_tensor-y_tensor)**2,dim=0)

sample_int = 100
#features
x0_arr = np.random.randint(0,10000,sample_int).reshape(-1,1)
x1_arr = np.random.randint(0,10000,sample_int).reshape(-1,1)
x2_arr = np.random.randint(0,10000,sample_int).reshape(-1,1)
x3_arr = np.random.randint(0,10000,sample_int).reshape(-1,1)
x4_arr = np.random.randint(0,10000,sample_int).reshape(-1,1)
x_arr = np.concatenate([x0_arr, x1_arr, x2_arr, x3_arr, x4_arr],axis=1)
x_tensor = torch.Tensor(x_arr)
y_tensor = torch.Tensor(x0_arr*17+np.random.standard_normal(sample_int).reshape(-1,1)+
                         x1_arr*9+np.random.standard_normal(sample_int).reshape(-1,1)+
                         x2_arr*8+np.random.standard_normal(sample_int).reshape(-1,1)+
                         x3_arr*12+np.random.standard_normal(sample_int).reshape(-1,1)+
                         x4_arr*102+np.random.standard_normal(sample_int).reshape(-1,1))

train_dataset = TensorDataset(x_tensor,y_tensor)
train_dataloader = DataLoader(train_dataset,batch_size=20,shuffle=True)
def train_loop(model,optimizer,train_dataloader):
    for idx,(X_tensor,y_tensor) in enumerate(train_dataloader):
        y_pred_tensor = model(X_tensor)
        loss_tensor = loss_func(y_pred_tensor,y_tensor)
        optimizer.zero_grad() #set model parameters as 0
        loss_tensor.backward() #按给定学习率和负梯度调整
        optimizer.step()
        print(f'loss:{loss_tensor.item()}')
    for param in model.parameters():
        print(f'model parameters:{param}')
def main():
    model = ols_nn()
    #hyper-params
    learning_rate = 1e-10
    batch_size = 64
    epochs_int = 10000
    #optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    for i in range(epochs_int):
        print(f'epoch:{i+1}')
        train_loop(model,optimizer,train_dataloader)

if __name__ == '__main__':
    main()
