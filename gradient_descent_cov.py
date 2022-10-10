import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

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

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__() #调用父类init方法
        self.linear_relu_stack=nn.Sequential(
            nn.Conv2d(1,10,kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10,20,kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(500,10),
            nn.Sigmoid()
        )

    def forward(self,x):
        logits = self.linear_relu_stack(x)
        return logits



def train_loop(dataloader,model,loss_func,optimizer):
    size = len(dataloader.dataset)
    for batch,(X,y) in enumerate(dataloader):
        #Compute predictions and loss
        pred = model(X)
        loss = loss_func(pred,y)
        #Back-propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch%100==0:
            loss,current = loss.item(),batch*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
model = NeuralNetwork()
def test_loop(dataloader,model,loss_func):
    size = len(dataloader.dataset)
    test_loss,correct = 0,0

    with torch.no_grad():
        for X,y in dataloader:
            pred = model(X)
            test_loss +=loss_func(pred,y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
    test_loss /=size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    #hyper-params
    learning_rate = 1e-1
    batch_size = 64
    epochs = 200
    #loss func
    loss_func = nn.CrossEntropyLoss()
    #optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

    for t in range(epochs):
        print(f'epochs {t+1}\n---------------------')
        train_loop(train_datloader,model,loss_func,optimizer)
        test_loop(test_datloader,model,loss_func)
    print('Done')

if __name__=='__main__':
    torch.set_num_threads(1)
    main()