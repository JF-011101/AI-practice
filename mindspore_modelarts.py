from easydict import EasyDict
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import mindspore
import mindspore.dataset as ds
# vision.c_transforms模块是处理图像增强的高性能模块，用于数据增强图像数据改进训练模型。
import mindspore.dataset.vision.c_transforms as CV
# c_transforms模块提供常用操作，包括OneHotOp和TypeCast
import mindspore.dataset.transforms.c_transforms as C
from mindspore.common import dtype as mstype
from mindspore.common.initializer import Normal
from mindspore import nn, load_checkpoint
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor

# 设置MindSpore的执行模式和设备
from mindspore import Tensor, context
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

import argparse
# 创建解析
parser = argparse.ArgumentParser(description="train flower classi",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# 添加参数
parser.add_argument('--train_url', type=str, 
                    help='the path model saved')
parser.add_argument('--data_url', type=str, help='the training data')
# 解析参数
args, unkown = parser.parse_known_args()



cfg = EasyDict({
    'data_size': 516,
    'image_width': 32,  # 图片宽度
    'image_height': 32,  # 图片高度
    'batch_size': 32,
    'channel': 3,  # 图片通道数
    'num_class': 6,  # 分类类别
    'weight_decay': 0.01,
    'lr': 0.0001,  # 学习率
    'dropout_ratio': 0.5,
    'epoch_size': 120,  # 训练次数
    'sigma': 0.01,

    'save_checkpoint_steps': 1,  # 多少步保存一次模型
    'keep_checkpoint_max': 1,  # 最多保存多少个模型
    'output_directory': './',  # 保存模型路径
    'output_prefix': "flower_classification"  # 保存模型文件名字
})

data_path = args.data_url
model_path = args.train_url

# 定义图像识别网络
class LeNet(nn.Cell):
    # define the operator required
    def __init__(self, num_class=cfg.num_class, num_channel=cfg.channel):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    # use the preceding operators to construct networks
    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_dataset(path):
    # 定义数据集
    data_set = ds.ImageFolderDataset(path, class_indexing={'bee_balm': 0,
                                                           'blackberry_lily': 1,
                                                           'blanket_flower': 2,
                                                           'bougainvillea': 3,
                                                           'bromelia': 4,
                                                           'foxglove': 5})

    # 解码前将输入图像裁剪成任意大小和宽高比。
    transform_img = CV.RandomCropDecodeResize([cfg.image_width, cfg.image_height], scale=(0.08, 1.0),
                                              ratio=(0.75, 1.333))  # 改变尺寸
    # 转换输入图像；形状（H, W, C）为形状（C, H, W）。
    hwc2chw_op = CV.HWC2CHW()
    # 类型调整，转换为给定MindSpore数据类型的Tensor操作。
    type_cast_op = C.TypeCast(mstype.float32)
    # 将操作应用到此数据集。
    data_set = data_set.map(input_columns="image", num_parallel_workers=8, operations=transform_img)
    data_set = data_set.map(input_columns="image", operations=hwc2chw_op, num_parallel_workers=8)
    data_set = data_set.map(input_columns="image", operations=type_cast_op, num_parallel_workers=8)
    data_set = data_set.shuffle(buffer_size=cfg.data_size)
    return data_set


de_dataset = create_dataset(data_path)
(de_train, de_val) = de_dataset.split([0.8, 0.2])
de_train = de_train.batch(cfg.batch_size, drop_remainder=True)
de_val = de_val.batch(cfg.batch_size, drop_remainder=True)

print('训练数据集数量：', de_train.get_dataset_size() * cfg.batch_size)  # get_dataset_size()获取批处理的大小。
print('测试数据集数量：', de_val.get_dataset_size() * cfg.batch_size)
data_next = de_dataset.create_dict_iterator(output_numpy=True).__next__()
print('通道数/图像长/宽：', data_next['image'].shape)


net = LeNet()

# 计算网络结构的总参数量
total_params = 0
for param in net.trainable_params():
    total_params += np.prod(param.shape)
print('参数量:', total_params)

# 定义损失函数，计算softmax交叉熵。
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

# 定义优化器
net_opt = nn.Adam(params=net.trainable_params(), learning_rate=cfg.lr, weight_decay=0.0)

# 实例化模型对象
model = Model(net, loss_fn=net_loss, optimizer=net_opt, metrics={"acc"})

# 监控训练过程、保存模型
loss_cb = LossMonitor(per_print_times=de_train.get_dataset_size() * 10)
config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                             keep_checkpoint_max=cfg.keep_checkpoint_max)
ckpoint_cb = ModelCheckpoint(prefix=cfg.output_prefix, directory=args.train_url, config=config_ck)


print("============== Starting Training ==============")
model.train(cfg.epoch_size, de_train, callbacks=[loss_cb, ckpoint_cb], dataset_sink_mode=True)

# 评估模型，打印总体准确率
metric = model.eval(de_val)
print(metric)

