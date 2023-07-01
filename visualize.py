import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

if __name__ =="__main__":
    loss_file = "loss_and_time.json"
    with open(loss_file, "r") as f:
        train_process = json.loads(f.read())
    train_loss = train_process["train_loss"]

    iter = np.arange(len(train_loss)) * 200
    plt.plot(iter, train_loss)
    plt.title("loss vs iteration")
    plt.xlabel('iteration')
    plt.ylabel('train loss')
    plt.show()

    train_acc = train_process["train_acc"]
    test_acc = train_process["test_acc"]
    iter = np.arange(len(train_acc))
    plt.plot(iter, train_acc)
    plt.plot(iter, test_acc)
    plt.legend(['train acc', 'test acc'])
    plt.title("accuracy vs epochs")
    plt.xlabel('epochs')
    plt.ylabel('accuracy(%)')
    plt.show()

    print(f'训练集上准确率 {train_acc[-1]:.2f}%')
    print(f'测试集上准确率 {test_acc[-1]:.2f}%')
    print(f'每个epoch平均用时 {np.mean(train_process["train_time"]):.2f}s')



