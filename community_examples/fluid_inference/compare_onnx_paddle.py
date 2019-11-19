import base64
import getopt
#coding:utf-8
import sys
import os
import json
import numpy as np
import paddle.fluid as fluid
import numpy as np
import json
np.set_printoptions(threshold=np.inf)

val = 'vectors_3*224*224.txt'
vectors = []
file = open(val, 'r')
for line in file.readlines():
    datas = line.strip().split(" ")
    for d in datas:
        vectors.append(float(d))
file.close()
a = np.array(vectors)
print(a.shape)
tensor_img = a.reshape(3,224,224)
tensor_img = tensor_img.astype(np.float32)

tensor_img = np.expand_dims(tensor_img, axis=0)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
path = "./paddle_infer/" 
[inference_program, feed_target_names, fetch_targets] = (fluid.io.load_inference_model(dirname=path, executor=exe))
results = exe.run(inference_program,
                  feed={feed_target_names[0]: tensor_img},
                  fetch_list=fetch_targets)[0][0]
paddle_fea = results[:, 0, 0]

with open("onnx.json",'r') as load_f:
    onnx_dict = json.load(load_f)
error = []
onnx_fea = onnx_dict["res_str"]["raw_confidences"]
for index in range(len(onnx_fea)):
    _ofea = onnx_fea[index]
    _pfea = paddle_fea[index]
    error.append(abs(_ofea - _pfea))

print("max error:        {}".format(np.max(error)))
print("accumulate error: {}".format(np.sum(error)))

