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
