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
