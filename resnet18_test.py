import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device=',device)

# 定义转换函数，用于对图像进行预处理
basic_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

moco_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

# 加载CIFAR-10测试数据集
# 作为baseline的resnet18和使用moco预训练的resnet18使用了不同的预处理方式，在加载resnet18.pth时使用transform=basic_transform,
# 在加载使用moco预训练的resnet18时使用transform=moco_transform
testset = torchvision.datasets.CIFAR10(root='', train=False, download=True, transform=moco_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# 定义ResNet-18模型
model = torchvision.models.resnet18()

# 加载已保存的模型参数
save_model = torch.load('model_best.pth.tar')
if type(save_model) == torchvision.models.resnet.ResNet:
    model = save_model
elif type(save_model) == dict:
    save_state_dict = save_model['state_dict']
    # 我们检查state_dict中的键名是否包含了
    # "module."，这是由torch.nn.DataParallel在保存模型时添加的前缀。
    if list(save_state_dict.keys())[0].startswith('module.'):
        print('torch.nn.DataParallel')
        model = torch.nn.DataParallel(model)
        model.load_state_dict(save_state_dict)
    else:
        model.load_state_dict(save_state_dict)


# 将模型移动到GPU
model = model.to(device)
# 设置模型为评估模式
model.eval()

# 定义类别标签
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if __name__ == '__main__':
    # 在测试数据集上进行预测
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算准确率
    accuracy = 100 * correct / total
    print('Accuracy on the test dataset: {:.2f}%'.format(accuracy))
