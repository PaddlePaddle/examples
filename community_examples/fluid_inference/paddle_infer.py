#!/usr/bin/env python
#coding:utf-8
import paddle.fluid as fluid
import numpy as np
import sys
import time
import base64
import json
import pyjsonrpc
import urllib2
import paddle.fluid as fluid
import numpy as np
import cv2
import urllib2
from PIL import Image
import imghdr
import StringIO
np.set_printoptions(threshold=np.inf)

def proc_img(img, centercrop=True):
    #img = open(name, 'rb').read()
    resize = 256
    #crop = 224
    if imghdr.what("", img) == "gif":
        buff_in = StringIO.StringIO(img)
        img = Image.open(buff_in)
        first_frame = Image.new("RGBA", img.size)
        first_frame.paste(img, (0, 0), img.convert('RGBA'))
        buff_out = StringIO.StringIO()
        first_frame.save(buff_out, "PNG")
        img = buff_out.getvalue()
        buff_in.close()
        buff_out.close()
    image_raw = np.asarray(bytearray(img), dtype="uint8")
    img = cv2.imdecode(image_raw, cv2.IMREAD_COLOR)   
    # center crop, hold aspect ratio
    if img.shape[0] >= img.shape[1]:
        img = cv2.resize(img, (resize, img.shape[0] * resize / img.shape[1]))
        if centercrop:
            start = (img.shape[0]-resize)/2
            img = img[start:start+resize, :, :]
    elif img.shape[0] < img.shape[1]:
        img = cv2.resize(img, (img.shape[1] * resize / img.shape[0], resize))
        if centercrop:
            start = (img.shape[1]-resize)/2
            img = img[:, start:start+resize, :]
    img = img[:, :, ::-1] # BGR to RGB
    img = img.astype('float32')
    img -= np.array([0.485, 0.456, 0.406]) * 255.0
    img /= np.array([0.229, 0.224, 0.225]) * 255.0
    img = img.transpose(2, 0, 1) # NCHW
    img = np.expand_dims(img, axis=0)
    return img


path = "./paddle_infer/"       # paddle save  inference model
file = "./imagenet/part-00000" # dataset for inference, base64 format

place = fluid.CPUPlace()
exe = fluid.Executor(place)
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)

fea_mat = []
with open(file, 'r') as f:
    #for line in sys.stdin:
    for line in f.xreadlines():
        try:
            temp = line.strip().split('\t')
            img_base641 = base64.b64decode(temp[1].replace('-','+').replace('_','/'))
            
            img = proc_img(img_base641)
            
            t = time.time()
            fea = exe.run(inference_program,
                          feed={feed_target_names[0]: img},
                          fetch_list=fetch_targets)[0][0]
            fea /= np.linalg.norm(fea)
            feature1 = np.array(fea, dtype='float16')
            fea_mat.append(feature1[:,0,0])
            #print(list(feature1))
            # print "%s\t%s\t%s\t%s" % (temp[0], temp[1], temp[2], list(feature1))
        except Exception, e:
            print >> sys.stderr, str(e), temp[0]

np.save("paddle_result.npy", fea_mat)
