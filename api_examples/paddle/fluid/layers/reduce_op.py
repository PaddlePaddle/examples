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

# api: paddle.fluid.layers.reduce_sum
# env: local
# device: cpu
# text: reduce_sum


import paddle.fluid as fluid
import numpy as np 

x = fluid.data(name='x', shape=[2, 4], dtype='float64')

reduce_sum_no_dim = fluid.layers.reduce_sum(x)
reduce_sum_empty_dim = fluid.layers.reduce_sum(x, dim = [])
reduce_sum_dim_one = fluid.layers.reduce_sum(x, dim = 0)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
feeder = fluid.DataFeeder(feed_list=[x], place=place)
exe.run(fluid.default_startup_program())

name_list = [reduce_sum_no_dim,reduce_sum_empty_dim,reduce_sum_dim_one]

#data = [[np.random.random((2,4)).astype("float64")]]
data = [[[[0.2, 0.3, 0.5, 0.9], 
        [0.1, 0.2, 0.6, 0.7]]]]

print(data)

sum_no, sum_empty, dim_one = exe.run(
                                 fluid.default_main_program(),
                                 feed=feeder.feed(data),
                                 fetch_list = name_list,
                                 return_numpy=True)


print(sum_no) #[3.5]
print(sum_empty) #[3.5]
print(dim_one) #[0.3, 0.5, 1.1, 1.6]
