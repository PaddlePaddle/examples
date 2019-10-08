#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
' BiGRU + CRF '
 
__author__ = 'huxiaoman77'

# 导入 PaddlePaddle 函数库.
import paddle
from paddle import fluid

# 导入内置的 CoNLL 数据集.
from paddle.dataset import conll05

# 获取数据集的内置字典信息.
word_dict, _, label_dict = conll05.get_dict()

WORD_DIM = 32           # 超参数: 词向量维度.
BATCH_SIZE = 10         # 训练时 BATCH 大小.
EPOCH_NUM = 20          # 迭代轮数数目.
HIDDEN_DIM = 512        # 模型隐层大小.
LEARNING_RATE = 1e-1    # 模型学习率大小.

# 设置输入 word 和目标 label 的变量.
word = fluid.layers.data(name='word_data', shape=[1], dtype='int64', lod_level=1)
target = fluid.layers.data(name='target', shape=[1], dtype='int64', lod_level=1)

# 将词用 embedding 表示并通过线性层.
embedding = fluid.layers.embedding(size=[len(word_dict), WORD_DIM], input=word,
                                  param_attr=fluid.ParamAttr(name="emb", trainable=False))
hidden_0 = fluid.layers.fc(input=embedding, size=HIDDEN_DIM, act="tanh")

# 用 RNNs 得到输入的提取特征并做变换.
hidden_1 = fluid.layers.dynamic_lstm(
    input=hidden_0, size=HIDDEN_DIM,
    gate_activation='sigmoid',
    candidate_activation='relu',
    cell_activation='sigmoid')
feature_out = fluid.layers.fc(input=hidden_1, size=len(label_dict), act='tanh')

# 调用内置 CRF 函数并做状态转换解码.
crf_cost = fluid.layers.linear_chain_crf(
    input=feature_out, label=target,
    param_attr=fluid.ParamAttr(name='crfw', learning_rate=LEARNING_RATE))
avg_cost = fluid.layers.mean(crf_cost)

# 调用 SGD 优化函数并优化平均损失函数.
fluid.optimizer.SGD(learning_rate=LEARNING_RATE).minimize(avg_cost)

# 声明 PaddlePaddle 的计算引擎.
place = fluid.CPUPlace()
exe = fluid.Executor(place)
main_program = fluid.default_main_program()
exe.run(fluid.default_startup_program())

# 由于是 DEMO 因此用测试集训练模型.
feeder = fluid.DataFeeder(feed_list=[word, target], place=place)
shuffle_loader = paddle.reader.shuffle(paddle.dataset.conll05.test(), buf_size=8192)
train_data = paddle.batch(shuffle_loader, batch_size=BATCH_SIZE)

# 按 FOR 循环迭代训练模型并打印损失.
batch_id = 0
for pass_id in range(EPOCH_NUM):
    for data in train_data():
        data = [[d[0], d[-1]] for d in data]
        cost = exe.run(main_program, feed=feeder.feed(data), fetch_list=[avg_cost])

        if batch_id % 10 == 0:
            print("avg_cost:\t" + str(cost[0][0]))
        batch_id = batch_id + 1

