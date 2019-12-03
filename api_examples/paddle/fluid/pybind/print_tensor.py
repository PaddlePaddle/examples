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

# api: paddle.fluid
# env: local
# device: cpu
# text: print-tensor

import paddle.fluid as fluid
import numpy as np
np.random.seed(9000)

fluid.default_main_program().random_seed = 9000
fluid.default_startup_program().random_seed = 9000
def gen_data():
    return {"x": np.random.random(size=(32, 32)).astype('float32'),
            "y": np.random.randint(2, size=(32, 1)).astype('int64')}
def mlp(input_x, input_y, hid_dim=128, label_dim=2):
    fc_1 = fluid.layers.fc(input=input_x, size=hid_dim)
    prediction = fluid.layers.fc(input=[fc_1], size=label_dim, act='softmax')
    cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
    sum_cost = fluid.layers.reduce_mean(cost)
    return sum_cost, fc_1, prediction

input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
cost, fc_1, pred = mlp(input_x, input_y)

print("Finished FF")

sgd = fluid.optimizer.Adam(learning_rate=0.01)
sgd.minimize(cost)

print("Finished optimize")
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
step = 10

with open("main_program.txt", 'w') as f:
    f.write(str(fluid.default_main_program()))

scope = fluid.global_scope()

for i in range(step):
    cost_val = exe.run(feed=gen_data(),
                       program=fluid.default_main_program(),
                       fetch_list=[cost.name])
    print("step: %d, fc_0.w_0: %s" % (i, scope.var("fc_0.w_0").get_tensor().__array__()))
