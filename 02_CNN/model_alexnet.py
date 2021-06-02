import torch
import torchvision
import torchvision.transforms as transforms

# 変換器の作成
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # torch.Tensor へ変換
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 正規化する
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(trainloader)  # ローダを使って実際にデータを取得するためのイテレータを作成する．
images, labels = dataiter.next()  # イテレータからデータを取得する．

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.max_pool2d(F.relu(self.conv5(x)), (2, 2))
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().cuda()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # 訓練データを複数回(2周分)学習する

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # ローダからデータを取得する; データは [inputs, labels] の形で取得される．
        # イテレータを使用していないように見えますが for の内部で使用されています．

        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        # 勾配を0に初期化する(逆伝播に備える)．
        optimizer.zero_grad()

        # 順伝播 + 逆伝播 + 最適化(訓練)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 統計を表示する．
        running_loss += loss.item()
        if i % 2000 == 1999:    # 2000 ミニバッチ毎に表示する．
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

PATH = './alex_net.pth'
torch.save(net.state_dict(), PATH)


net = Net()
net.load_state_dict(torch.load(PATH))

# テストデータ全体に対してどの程度正しい予測をするのか確認
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# 各クラス毎の予測精度を確認
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))