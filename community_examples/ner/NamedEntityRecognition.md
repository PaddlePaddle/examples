## 手把手教你用 PaddlePaddle 做命名实体识别


****命名实体识别（Named Entity Recognition，NER）是 NLP 几个经典任务之一，通俗易懂的来说，他就是从一段文本中抽取出需求的关键词，如地名，人名等。

如上图所示，Google、IBM、Baidu 这些都是企业名、Chinese、U.S. 都是地名。就科学研究来说，命名实体是非常通用的技术，类似任务型对话中的槽位识别（Slot Filling）、基础语言学中的语义角色标注（Semantic Role Labelling）都变相地使用了命名实体识别的技术；而就工业应用而言，命名实体其实就是序列标注（Sequential Tagging），是除分类外最值得信赖和应用最广的技术，例如智能客服、网络文本分析，关键词提取等。

下面我们先带您了解一些 Gated RNN 和 CRF 的背景知识，然后再教您一步一步用 Paddle Paddle 实现一个命名实体任务。另外，我们采用经典的 CoNLL 数据集。

## Part-1：RNN 基础知识

循环神经网络（Recurrent Neural Networks，RNN）是有效建模有时序特征输入的方式。它的原理实际上非常简单，可以被以下简单的张量公式建模：

$$\vec{h}_t = f(\vec{x}_t, \vec{h}_{t-1})$$
$$\vec{y}_t = g(\vec{h}_t)$$

其中函数 f, g 是自定的，可以非线性，也可以就是简单的线性变换，比较常用的是：


$$\vec{h}_t = ReLU(W_{xh}\vec{x}_t + W_{hh}\vec{h}_{t-1})$$
$$\vec{y}_t = W_{hy}\vec{h}_t$$

虽然理论上 RNN 能建模无限长的序列，但因为很多数值计算（如梯度弥散、过拟合等）的原因致使 RNN 实际能收容的长度很小。等等类似的原因催生了门机制。

<br/>

大量实验证明，基于门机制（Gate Mechanism）可以一定程度上缓解 RNN 的梯度弥散、过拟合等问题。LSTM 是最广为应用的 Gated RNN，它的结构如下：

<br/>
<div align=center><img src=https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/32_blog_image_1.png></div>
<br/>

如上图所示，运算 tanh（取值 -1 ~ 1） 和 α（Sigmoid，取值 0 – 1）表示控制滤过信息的 “门”。网上关于这些门有很多解释，可以参考这篇[博文](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)。

除了 LSTM 外，GRU（Gated Recurrent Unit） 也是一种常用的 Gated RNN：

> + 由于结构相对简单，相比起 LSTM，GRU 的计算速度更快；
> + 由于参数较少，在小样本数据及上，GRU 的泛化效果更好；

事实上，一些类似机器阅读的任务要求高效计算，大家都会采用 GRU。甚至现在有很多工作开始为了效率而采用 Transformer 的结构。可以参考这篇[论文](https://arxiv.org/pdf/1804.09541.pdf)。

## Part-2：CRF 基础知识

> 给定输入 $X=(x_1,x_2,⋯,x_n)$，一般 RNN 模型输出标注序列 $Y=(y_1,y_2,⋯,y_n)$ 的办法就是简单的贪心，在每个词上做 argmax，忽略了类别之间的时序依存关系。

<br/>
<div align=center><img src=http://www.davidsbatista.net/assets/images/2017-11-13-Conditional_Random_Fields.png></div>
<br/>

线性链条件随机场（Linear Chain Conditional Random Field），是基于马尔科夫性建模时序序列的有效方法。算法上可以利用损失 l(x)=-log⁡(exp⁡〖(x〗)) 的函数特点做前向计算；用维特比算法（实际上是动态规划，因此比贪心解码肯定好）做逆向解码。

形式上，给定发射特征（由 RNN 编码器获得）矩阵 E 和转移（CRF 参数矩阵，需要在计算图中被损失函数反向优化）矩阵 T，可计算给定输入输出的匹配得分：

$$ score(X,y)=\sum_{i,y_i} E_{i,y_i} + \sum_i T_{y_i,y_{i+1}} $$

其中 X 是输入词序列，y 是预测的 label 序列。然后使以下目标最大化：

$$ P(y_{gold} | X) = exp(s(X, y_{gold})) / \sum_{y \in Y_s} exp(s(X, y)) $$

以上就是 CRF 的核心原理。当然要实现一个 CRF，尤其是支持 batch 的 CRF，难度非常高，非常容易出 BUG 或低效的问题。之前笔者用 Pytorch 时就非常不便，一方面手动实现不是特别方便，另一方面用截取开源代码接口不好用。然而 PaddlePaddle 就很棒，它原生的提供了 CRF 的接口，同时支持损失函数计算和反向解码等功能。

## Part-3：建模思路

**我们数据简单来说就是一句话。目前比较流行建模序列标注的方法是 BIO 标注，其中 B 表示 Begin，即标签的起始；I 表示 In，即标签的内部；O 表示 other，即非标签词。如下面图所示，低端的 w_i,0≤i≤4 表示输入，顶端的输出表示 BIO 标注。

<br/>
<div align=center><img src=http://www.davidsbatista.net/assets/images/2018-10-21_LSTM_CRF_matrix.png></div>
<br/>

模型的结构也如上图所示，我们首先用 Bi-GRU（忽略图中的 LSTM） 循环编码以获取输入序列的特征，然后再用 CRF 优化解码序列，从而达到比单用 RNNs 更好的效果。

## Part-4：PaddlePaddle实现

终于到了动手的部分。本节将会一步一步教您如何用 PaddlePaddle 实现 BiGRU + CRF 做序列标注。由于是demo，我们力求简单，让您能够将精力放到最核心的地方！

```python
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

```

输出结果：

```bash
[==================================================]l05st/conll05st%2FwordDict.txt not found, downloading http://paddlemodels.bj.bcebos.com/conll05st%2FwordDict.txt
[==================================================]l05st/conll05st%2FverbDict.txt not found, downloading http://paddlemodels.bj.bcebos.com/conll05st%2FverbDict.txt
[==================================================]l05st/conll05st%2FtargetDict.txt not found, downloading http://paddlemodels.bj.bcebos.com/conll05st%2FtargetDict.txt



[==================================================]l05st/conll05st-tests.tar.gz not found, downloading http://paddlemodels.bj.bcebos.com/conll05st/conll05st-tests.tar.gz
avg_cost:	150.293
avg_cost:	102.468956
avg_cost:	55.6771
avg_cost:	45.3841
avg_cost:	55.393227
avg_cost:	47.93234
avg_cost:	43.970863
avg_cost:	46.385117
avg_cost:	41.363853
avg_cost:	46.78142
avg_cost:	57.744774
avg_cost:	33.612484
avg_cost:	30.556072
avg_cost:	36.819237
avg_cost:	37.627037
avg_cost:	38.523487
avg_cost:	38.453724
avg_cost:	47.136696
avg_cost:	39.19741
avg_cost:	36.097908
avg_cost:	38.043354
avg_cost:	25.416252
avg_cost:	22.22056
avg_cost:	45.511017
avg_cost:	36.31049
avg_cost:	33.19769
avg_cost:	37.441
avg_cost:	34.272476
avg_cost:	25.608454
avg_cost:	27.278118
avg_cost:	32.817966
avg_cost:	25.00168
avg_cost:	36.707935
avg_cost:	27.87775
avg_cost:	31.61958
avg_cost:	33.14825
avg_cost:	34.09555
avg_cost:	24.461145
avg_cost:	33.34344
avg_cost:	29.01653
avg_cost:	36.376263

```

