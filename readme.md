#项目名称
这是一个使用MoCo进行resnet18的预训练，并使用Linear Classification Protocol进行评估的模型。数据集使用的是CIFAR-10数据集。

##项目结构
```
.
├── checkpoint_0199.pth.tar
├── cifar-10-batches-py
│   ├── batches.meta
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   ├── readme.html
│   └── test_batch
├── cmd.txt
├── loss_and_time.json
├── moco-main
│   ├── LICENSE
│   ├── README.md
│   ├── detection
│   │   ├── README.md
│   │   ├── configs
│   │   │   ├── Base-RCNN-C4-BN.yaml
│   │   │   ├── coco_R_50_C4_2x.yaml
│   │   │   ├── coco_R_50_C4_2x_moco.yaml
│   │   │   ├── pascal_voc_R_50_C4_24k.yaml
│   │   │   └── pascal_voc_R_50_C4_24k_moco.yaml
│   │   ├── convert-pretrain-to-detectron2.py
│   │   └── train_net.py
│   ├── main_lincls.py
│   ├── main_moco.py
│   └── moco
│       ├── __init__.py
│       ├── builder.py
│       └── loader.py
├── moco-record
│   ├── moco-lincls.txt
│   └── moco-pretrain.txt
├── moco_visualize.py
├── model_best.pth.tar
├── readme.md
├── resnet18.pth
├── resnet18_test.py
├── resnet18_train.py
├── result.txt
└── visualize.py

```
`cifar-10-batches-py`文件下是`CIFAR-10`数据集，该数据集已经划分好训练集与测试集且`pytorch`有直接解析该数据集的函数，该文件太大了无法上传至GitHub，使用时需自行下载 `CIFAR-10`数据集并放在根目录下。

`resnet18.pth`是随机初始化训练的resnet18模型，作为基准（baseline）使用。

`checkpoint_0199.pth.tar`是经过MoCo预训练的resnet18模型（特征提取器），`model_best.pth.tar`是在预训练好的特征提取器（`checkpoint_0199.pth.tar`）上通过Linear Classification Protocol进一步训练全链接层得到的resnet18模型。

`loss_and_time.json`里保存了基准resnet18模型训练时的损失（loss）、准确率（accuracy）以及每一轮时长。

`moco-record`目录下的文件存了MoCo预训练时和Linear Classification Protocol训练时的损失（loss）、准确率（accuracy）以及每一轮时长。

###代码文件包括以下内容：
+   `resnet18_train.py` 提供了训练作为基准的resnet18的代码。

+   `moco-main` 目录下提供了用MoCo自监督学习训练模型的方法以及使用Linear Classification Protocol进一步训练线性分类器的方法。
参考 https://github.com/facebookresearch/moco.git ，此处调整了部分代码使其能够在CIFAR10数据集上进行训练。
    
+   `resnet18_test.py` 提供了测试`resnet18`模型的代码，会显示模型在`CIFAR10`测试集上的准确率。
    
+   `visualize.py`和`moco_visualize.py`分别提供了可视化基准resnet18训练过程和MoCo预训练、Linear Classification Protocol的训练过程的代码。

##使用方法
+   用户可以在`resnet18_train.py`中训练自己的基准模型，如有必要，用户可以进行如下调整。
    +   `my_device`变量设置为自己能够使用的加速设备，可以是`cuda`、`cpu`、`mps`。
    +   `root_dir`变量设置为自己的`CIFAR10`数据集路径。
    +   `batch_size`这里设置为32，用户可以根据自己的设备调整。
    +   优化器的学习率、权重衰减、训练总轮次都可以自行调整，默认当在测试集上准确率大于99.9%时停止。

+   用户可以通过以下格式的命令用MoCo对resnet18进行预训练，每一个epoch都会产生一个`checkpoint.pth.tar`文件
    ```
    python moco-main/main_moco.py -a resnet18 --lr 0.03 --batch-size 256 --dist-backend gloo --dist-url 'tcp://localhost:10010' --multiprocessing-distributed --world-size 1 --rank 0 cifar-10-batches-py
    ```
    
+   用户可以通过以下格式的命令对预训练好的resnet18模型进行线性分类训练，最好的结果会被保存在`model_best.pth.tar`中。
    ```angular2html
    python moco-main/main_lincls.py -a resnet18 --lr 30.0 --batch-size 256 --dist-backend gloo --pretrained checkpoint_0199.pth.tar --dist-url 'tcp://localhost:10001'  --world-size 1 --rank 0 --multiprocessing-distributed cifar-10-batches-py
    ```

+   用户可以通过以下在`resnet18_test.py`中调用自己训练好的模型。
##已训练的模型

训练好的模型包括`resnet18.pth`,`checkpoint_0199.pth.tar`以及`model_best.pth.tar`，用户需要自己训练，本人训练好的模型保存在百度云。

链接: https://pan.baidu.com/s/1Gtw2Y0fW7qMZXHq0P0qpCw 提取码: kknr