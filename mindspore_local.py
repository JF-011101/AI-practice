
from easydict import EasyDict
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import mindspore
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.common import dtype as mstype
from mindspore.common.initializer import Normal
from mindspore import nn, load_checkpoint
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
#easydict模块用于以属性的方式访问字典的值
#glob模块主要用于查找符合特定规则的文件路径名，类似使用windows下的文件搜索
#os模块主要用于处理文件和目录
#导入mindspore框架数据集
#vision.c_transforms模块是处理图像增强的高性能模块，用于数据增强图像数据改进训练模型。
#c_transforms模块提供常用操作，包括OneHotOp和TypeCast
#导入模块initializer.Normal用于初始化截断正态分布

# 设置MindSpore的执行模式和设备
from mindspore import Tensor, context

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

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

data_path = 'flower_photos/'
test_path = 'TestImages/'


# 定义网络
class LeNet(nn.Cell):
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

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))  # C1 S2
        x = self.max_pool2d(self.relu(self.conv2(x)))  # C3 S4
        x = self.flatten(x)
        x = self.relu(self.fc1(x))  # C5
        x = self.relu(self.fc2(x))  # F6
        x = self.fc3(x)  # output
        return x


def create_dataset(path):
    # 定义数据集
    data_set = ds.ImageFolderDataset(path, class_indexing={'bee_balm': 0,
                                                           'blackberry_lily': 1,
                                                           'blanket_flower': 2,
                                                           'bougainvillea': 3,
                                                           'bromelia': 4,
                                                           'foxglove': 5})

    # 裁剪和解码
    transform_img = CV.RandomCropDecodeResize([cfg.image_width, cfg.image_height], scale=(0.08, 1.0),
                                              ratio=(0.75, 1.333))
    # 转换输入图像；形状（H, W, C）为形状（C, H, W）。
    hwc2chw_op = CV.HWC2CHW()
    # 类型调整
    type_cast_op = C.TypeCast(mstype.float32)
    # 将操作应用到此数据集。
    data_set = data_set.map(input_columns="image", num_parallel_workers=8, operations=transform_img)
    data_set = data_set.map(input_columns="image", operations=hwc2chw_op, num_parallel_workers=8)
    data_set = data_set.map(input_columns="image", operations=type_cast_op, num_parallel_workers=8)
    # 打乱数据
    data_set = data_set.shuffle(buffer_size=cfg.data_size)
    return data_set


de_dataset = create_dataset(data_path)
# 划分数据集和验证集
(de_train, de_val) = de_dataset.split([0.8, 0.2])
# 分批
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

# 定义损失函数
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

# 定义优化器
net_opt = nn.Adam(params=net.trainable_params(), learning_rate=cfg.lr, weight_decay=0.0)

# 实例化模型
model = Model(net, loss_fn=net_loss, optimizer=net_opt, metrics={"acc"})

loss_cb = LossMonitor(per_print_times=de_train.get_dataset_size() * 10)

config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                             keep_checkpoint_max=cfg.keep_checkpoint_max)
# 保存模型参数
ckpoint_cb = ModelCheckpoint(prefix=cfg.output_prefix, directory=cfg.output_directory, config=config_ck)

print("============== Starting Training ==============")
model.train(cfg.epoch_size, de_train, callbacks=[loss_cb, ckpoint_cb], dataset_sink_mode=True)

# 评估模型，打印总体准确率
metric = model.eval(de_val)
print(metric)

CKPT = os.path.join(cfg.output_directory, cfg.output_prefix + '-'
                    + str(cfg.epoch_size) + '_'
                    + str(de_train.get_dataset_size()) + '.ckpt')

# CKPT = os.path.join('./flower_classification-120_12.ckpt')

# 加载参数，实例化
net = LeNet()
load_checkpoint(CKPT, net=net)
model = Model(net)

class_names = {0: 'bee_balm', 1: 'blackberry_lily', 2: 'blanket_flower',
               3: 'bougainvillea', 4: 'bromelia', 5: 'foxglove'}

# 测试集很小，直接依次读入、预测和展示图像
for i in range(6):
    img = Image.open(test_path + "test" + str(i + 1) + ".jpg")

    temp = img.resize((cfg.image_width, cfg.image_height))
    temp = np.array(temp)
    temp = temp.transpose(2, 0, 1)  # HWC格式转化成CHW格式
    temp = np.expand_dims(temp, 0)  # 增加一个batch维度
    img_tensor = Tensor(temp, dtype=mindspore.float32)  # 将图像转成向量

    # 预测
    predictions = model.predict(img_tensor)
    predictions = predictions.asnumpy()
    label = class_names[np.argmax(predictions)]

    plt.title("PRE:{}".format(label))
    # 展示图像信息
    plt.imshow(np.array(img))
    plt.show()
