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

# api: paddle.fluid.layers.sequence_pad
# env: local
# device: cpu
# text: sequence-pad

import paddle.fluid as fluid
import numpy
 
x = fluid.layers.data(name="question", shape=[1], dtype="int64", lod_level=1)
 
# define net here
embed = fluid.layers.embedding(input=x, size=[32, 2],
                               param_attr=fluid.ParamAttr(name='emb.w'))
 
pad_value = fluid.layers.assign(input=numpy.array([0], dtype=numpy.float32))
z, mask = fluid.layers.sequence_pad(x=embed, pad_value=pad_value)
 
place = fluid.CPUPlace()
exe = fluid.Executor(place)
feeder = fluid.DataFeeder(feed_list=[x], place=place)
exe.run(fluid.default_startup_program())
 
# prepare a batch of data
data = [([0, 1, 2, 3, 3],), ([0, 1, 2],)]
 
mask_out, z_out = exe.run(fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[mask, z],
                return_numpy=True)
 
print(mask_out)
print(z_out)

#[[5]
# [3]]
#[[[ 0.03990805 -0.10303718]
#  [ 0.08801201 -0.30412018]
#  [ 0.0706093  -0.18075395]
#  [-0.0283702   0.01683199]
#  [-0.0283702   0.01683199]]
 
# [[ 0.03990805 -0.10303718]
#  [ 0.08801201 -0.30412018]
#  [ 0.0706093  -0.18075395]
#  [ 0.          0.        ]
#  [ 0.          0.        ]]]
