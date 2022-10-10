import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torch.nn.functional as F

# one channel
channel_int = 1
train_dat = datasets.FashionMNIST(
    root='data',
    train=True,
    download=False,
    transform=ToTensor()
)

test_dat = datasets.FashionMNIST(
    root='data',
    train=False,
    download=False,
    transform=ToTensor()
)

train_datloader = DataLoader(train_dat,batch_size=64)
test_datloader = DataLoader(test_dat,batch_size=64)

class inception(nn.Module):
    def __init__(self,in_channel_int):
        super(inception, self).__init__()
        self.avg_pool_1x1 = nn.Sequential(
            nn.AvgPool2d(3,padding=1,stride=1),
            nn.Conv2d(in_channel_int,24,kernel_size=(1,1))
        )
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channel_int,16,kernel_size=(1,1))
        )
        self.conv_1x1_5x5 = nn.Sequential(
            nn.Conv2d(in_channel_int,16,kernel_size=(1,1)),
            nn.Conv2d(16,24,kernel_size=(5,5),padding=2)
        )
        self.conv_1x1_3x3_3x3 = nn.Sequential(
            nn.Conv2d(in_channel_int, 16, kernel_size=(1, 1)),
            nn.Conv2d(16, 24, kernel_size=(3, 3),padding=1),
            nn.Conv2d(24,24,kernel_size=(3,3),padding=1)
        )
    def forward(self, x):
        x_avg_pool_1x1 = self.avg_pool_1x1(x)
        x_conv_1x1 = self.conv_1x1(x)
        x_conv_1x1_5x5 = self.conv_1x1_5x5(x)
        x_conv_1x1_3x3_3x3 = self.conv_1x1_3x3_3x3(x)
        x = torch.cat([x_avg_pool_1x1,x_conv_1x1,x_conv_1x1_5x5,x_conv_1x1_3x3_3x3],dim=1)
        return x

class resid_net(nn.Module):
    def __init__(self,in_channel_int,out_channel_int):
        super(resid_net,self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel_int,out_channel_int,kernel_size=(3,3),padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channel_int, out_channel_int, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
    def forward(self,x):
        conv = self.conv_layer(x)
        return F.relu(conv+x)


class CNN_module(nn.Module):
    def __init__(self):
        super(CNN_module, self).__init__()
        self.sequential_cnn = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            inception(64),
            resid_net(88,88),
            nn.BatchNorm2d(88),
            nn.Conv2d(88,88,kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            inception(88),
            resid_net(88,88),
            nn.Conv2d(88,88,kernel_size=(3,3)),
            nn.Flatten(),
            nn.Linear(792,10)
        )

    def forward(self,x):
        logits = self.sequential_cnn(x)
        return logits

def train_loop(train_datloader,model,loss_func,optimizer):
    size = len(train_datloader.dataset)
    for index_int,(X,y) in enumerate(train_datloader):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_func(pred,y)
        loss.backward()
        optimizer.step()
        if index_int%100==0:
            loss,current = loss.item(),index_int*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader,model,loss_func):
    size = len(dataloader.dataset)
    test_loss = 0
    correct = 0
    for index_int,(X,y) in enumerate(dataloader):
        pred = model(X)
        test_loss += loss_func(pred,y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    model = CNN_module()
    # hyper-params
    learning_rate = 1e-1
    batch_size = 64
    epochs = 30
    # loss func
    loss_func = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for t in range(epochs):
        print(f'epochs {t + 1}\n---------------------')
        train_loop(train_datloader, model, loss_func, optimizer)
        test_loop(test_datloader, model, loss_func)
    print('Done')

if __name__ == '__main__':
    main()