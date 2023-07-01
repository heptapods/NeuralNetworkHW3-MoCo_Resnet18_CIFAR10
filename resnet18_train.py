import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import pickle
import json
import time

import torch.multiprocessing as multiprocessing
multiprocessing.set_start_method('spawn', force=True)


# 检查是否有可用的GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
print('device=',device)

# 定义数据转换
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 加载训练集
trainset = torchvision.datasets.CIFAR10(root='', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

# 加载测试集
testset = torchvision.datasets.CIFAR10(root='', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

# 定义类别标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 加载预训练的ResNet-18模型
model = models.resnet18()

# 将模型移动到GPU
model = model.to(device)

# 将最后一层替换为与CIFAR-10数据集类别数相匹配的全连接层
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# 将模型的全连接层移动到GPU
model.fc = model.fc.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.5)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

# 计算正确率
def calcualteAccuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            # 将数据移动到GPU
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


if __name__ =='__main__':
    result = {'test_acc':[], 'train_acc':[], 'train_loss':[], 'train_time':[]}
    test_acc = calcualteAccuracy(model, testloader)
    train_acc = calcualteAccuracy(model, trainloader)
    result['train_acc'].append(train_acc)
    result['test_acc'].append(test_acc)
    # 训练模型

    for epoch in range(20):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # 将数据移动到GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(epoch, i, loss.item())
            if i % 200 == 199:
                mean_loss = running_loss / 200
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, mean_loss))
                result['test_loss'].append(mean_loss)
                running_loss = 0.0

        print('Finished training epoch', epoch)
        lr_scheduler.step()
        train_time = time.time() - start_time
        result['train_time'].append(train_time)
        # 测试模型
        test_acc = calcualteAccuracy(model, testloader)
        train_acc = calcualteAccuracy(model, trainloader)
        print(f'Accuracy on train images: {train_acc}%')
        print(f'Accuracy on test images: {test_acc}%')
        result['train_acc'].append(train_acc)
        result['test_acc'].append(test_acc)
        if train_acc > 99.9:
            print('Finished training epoch', epoch, 'training accuracy > 0.999')
            break

    torch.save(model, 'resnet18.pth')
    with open('loss_and_time.json', 'w') as f:
        json.dump(result, f)

