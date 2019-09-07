# API Examples书写规范

## API Examples的定位是什么？
- 针对每个API提供可完整跑通的最精简代码
- 用户对API文档理解不够透彻，或者觉得API文档中的代码示例不够全面，可以快速在API Examples中添加示例
- 缓解CI系统的压力，快速修复

## 输出规范
- 一个API一个.py文件
- .py文件的格式参考下面的示例
``` python
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# api: paddle.fluid.data
# env: local
# device: gpu
# text：feed-gpu-data

import paddle.fluid as fluid
import numpy as np

def gen_data():
    return {"x": np.ones((1, 32)).astype('float32')}

input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
input_val = exe.run(feed=gen_data(),
                    fetch_list=[input_x.name])
print(input_val)
# [array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]],
#         dtype=float32)]

```
- API的全路径名需要在注释中的api字段注明
- API的运行环境需要在env字段中注明，local/distributed
- API的运行设备需要在device字段中注明，gpu/cpu
- API示例需要能够运行，用户能够直接根据代码获得运算结果
- API示例可以起一个名称，作为官网文档的链接文字，代码中用text字段表示

## 管理规范
- API示例按照目录结构放置，命名规则为"API名_i.py"，其中i为示例序号。即单个API可以有多个示例，按序号编写。
- API examples会根据目录结构，example文件名在官网的API文档中通过链接的方式展现，定期更新做有效性检查。
- API examples对示例是否能够正常运行有要求，会有CI系统每天回归整体示例是否可以运行。
- 通过在官网进行示例展现，我们会统计后台点击数，客观评价API的需求，并对点击数较高的作者进行奖励。
