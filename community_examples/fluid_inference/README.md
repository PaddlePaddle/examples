## 使用paddle进行预测以及onnx的使用

1、训练模型，使用save_inference_model函数保存训练好的模型以及参数，路径为./paddle_infer。

2、安装paddle2onnx，并转换模型，得到转换后的onnx文件。

```
pip install onnx          # 安装onnx
pip install paddle2onnx   # 安装paddle2onnx
paddle2onnx --fluid_model paddle_infer  --onnx_model onnx_infer  # paddle的预测模型转换为onnx
```

3、使用paddle库进行预测，同时和onnx转后的结果进行比较。

```
python paddle_infer.py         #使用load_inference_model 预测代码
```

```
python compare_onnx_paddle.py  #比较paddle转换前后的diff
```
