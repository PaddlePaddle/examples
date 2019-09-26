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

# api: paddle.fluid.framework.Program._prune_with_input()
# env: local
# device: cpu
# text：prune-with-input

import paddle.fluid as fluid
import paddle.fluid.optimizer as optimizer
import numpy as np

def sample_data():
   res = []
   for i in range(2):
       data = np.random.normal(size=(2,))
       label = np.random.randint(2, size=(1,))
       res.append((data, label))
   return res

x = fluid.layers.data(name='x', shape=[2], dtype='float32')
label = fluid.layers.data(name="label", shape=[1], dtype="int64")

# define net here
y = fluid.layers.fc(input=[x], size=2, act="softmax")
loss = fluid.layers.cross_entropy(input=y, label=label)
loss = fluid.layers.mean(x=loss)

sgd = fluid.optimizer.SGD(learning_rate=0.01)
sgd.minimize(loss)

with open("original_program", "w") as f:
    f.write(str(fluid.default_main_program()))

pruned_program = fluid.default_main_program()._prune_with_input(
                        feeded_var_names=[y.name, label.name],
                        targets = [loss])

with open("pruned_program", "w") as f:
    f.write(str(pruned_program))
