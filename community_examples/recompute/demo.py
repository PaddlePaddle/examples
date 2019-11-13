import paddle.fluid as fluid
import numpy as np
np.random.seed(9000)

fluid.default_main_program().random_seed = 9000
fluid.default_startup_program().random_seed = 9000

def gen_data():
    return {"x": np.random.random(size=(32, 32)).astype('float32'),
            "y": np.random.randint(2, size=(32, 1)).astype('int64')}

def mlp(input_x, input_y, hid_dim=128, label_dim=2):
    print(input_x)
    fc_1 = fluid.layers.fc(input=input_x, size=hid_dim, act='relu')
    fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim, act='relu')
    fc_3 = fluid.layers.fc(input=fc_2, size=hid_dim, act='relu')
    prediction = fluid.layers.fc(input=[fc_3], size=label_dim, act='softmax')
    cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
    sum_cost = fluid.layers.reduce_mean(cost)
    return sum_cost, fc_1, prediction

input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
cost, fc_1, pred = mlp(input_x, input_y)

print("Finished FF")

sgd = fluid.optimizer.Adam(learning_rate=0.01)
sgd = fluid.optimizer.RecomputeOptimizer(sgd)
sgd._set_checkpoints([fc_1, pred])
sgd.minimize(cost)

print("Finished optimize")
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
step = 10

with open("main_program.txt", 'w') as f:
    f.write(str(fluid.default_main_program()))

for i in range(step):
    cost_val = exe.run(feed=gen_data(),
                       program=fluid.default_main_program(),
                       fetch_list=[cost.name])
    print("step=%d cost=%f" % (i, cost_val[0]))

fluid.io.save_inference_model("model", ["x", "y"], [cost], exe, 
                              export_for_deployment=False)

