import math
import sys
import os
import caffe
import re
from fast_rcnn.test import _get_blobs
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.config import cfg, cfg_from_file
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
from collections import Counter
from copy import deepcopy
import yaml

with open("./code/pickobject_config.yaml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

#caffe.set_mode_gpu()
#caffe.set_device(0)

cfg_from_file(config["model"]["cfg_file"])

if __name__ == '__main__':
    t0 = time.time()
    input_file = sys.argv[1]
    lr = float(sys.argv[2]) # initial learning rate
    targeted = bool(sys.argv[3]=='True')
    print(targeted)
    if targeted:
          new_class = int(sys.argv[4]) # desired class for targeted attack
    
    output_file_numpy = sys.argv[5]
    gpuID = int(sys.argv[6])

    caffe.set_mode_gpu()   
    caffe.set_device(gpuID)
    #caffe.set_solver_count(2)
    #caffe.set_solver_rank(gpuID)
    #caffe.set_multiprocess(True)

    ## model and prototxt
    weights = config["model"]["weights"]
    prototxt_train = config["model"]["prototxt"]

    ## specifiy min and max changes
    eps = 200

    net = caffe.Net(prototxt_train, caffe.TEST, weights=weights)

    MEANS = np.array([[102.9801, 115.9465, 122.7717]])

    data_path = config["data"]["data_path"]
    classes = ['__nothing__']
    with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
        for object in f.readlines():
            classes.append(object.split(',')[0].lower().strip())

    x = cv2.imread(input_file)
    print("Original image shape:",x.shape)

    blobs, im_scales = _get_blobs(x, None)
    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)

    gt_labels = np.zeros( (300), dtype=float )
    gt_labels += -1

    cls_temp = np.zeros((300,1), dtype=int)
    blobs['labels'] = gt_labels
    blobs['predicted_cls'] = cls_temp

    net.blobs['data'].reshape(*(blobs['data'].shape))

    if 'im_info' in net.blobs:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    if 'labels' in net.blobs:
        net.blobs['labels'].reshape( *(blobs['labels'].shape) )
    if 'predicted_cls' in net.blobs:
        net.blobs['predicted_cls'].reshape(*(blobs['predicted_cls'].shape))

    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}

    if 'im_info' in net.blobs:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    if 'labels' in net.blobs:
        forward_kwargs['labels'] = blobs['labels'].astype(np.float32, copy=False)
    if 'predicted_cls' in net.blobs:
        forward_kwargs['predicted_cls'] = blobs['predicted_cls'].astype(np.float32, copy=False)

    print("starting forward pass")
    d = net.forward(end='proposal',**forward_kwargs)
    num_rois = d['rois'].shape[0]
    gt_labels = np.zeros( (num_rois), dtype=float )
    gt_labels += -1

    cls_temp = np.zeros((num_rois,1), dtype=int)
    blobs['labels'] = gt_labels
    blobs['predicted_cls'] = cls_temp

    if 'labels' in net.blobs:
        net.blobs['labels'].reshape( *(blobs['labels'].shape) )
    if 'predicted_cls' in net.blobs:
        net.blobs['predicted_cls'].reshape(*(blobs['predicted_cls'].shape))

    if 'labels' in net.blobs:
        forward_kwargs['labels'] = blobs['labels'].astype(np.float32, copy=False)
    if 'predicted_cls' in net.blobs:
        forward_kwargs['predicted_cls'] = blobs['predicted_cls'].astype(np.float32, copy=False)
    
    net.forward(start='roi_pool5',**forward_kwargs)

    adversarial_x = x.astype('float32')
    print "Shape of the image:", adversarial_x.shape

    clip_min = adversarial_x - eps
    clip_max = adversarial_x + eps

    single_start_point = adversarial_x

    cls_score_all = net.blobs['cls_score'].data[:, 1:].argmax(axis=1) + 1
    original_cls = deepcopy(cls_score_all)

    a=net.blobs['cls_score'].data[:, 1:]
    print np.unravel_index(a.argmax(), a.shape)

    output_file_numpy = 'all' + '_' + output_file_numpy

    print cls_score_all
    print cls_score_all.shape

    cls_boxes = net.blobs['rois'].data[:,1:5] / im_scales[0]
    
    attack_try = 0
   
    remained_cls=list(set(range(1,1601)).difference(original_cls))
    target_idx = np.random.choice(remained_cls)
 
    while (True):
        
        caffe.set_mode_gpu()
        caffe.set_device(gpuID)

        if (( attack_try +1 ) % 120 == 0): # change it to 10 later
            adversarial_x = single_start_point
            lr = lr * (1.2)
        if (lr > 10000):
            break

        blobs, im_scales = _get_blobs(adversarial_x,None)
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)

        gt_labels = np.zeros((300), dtype=float )
        gt_labels += -1

        cls_temp = np.zeros((300,1), dtype=int)
        blobs['labels'] = gt_labels
        blobs['predicted_cls'] = cls_temp

        net.blobs['data'].reshape(*(blobs['data'].shape))

        if 'im_info' in net.blobs:
            net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
        if 'labels' in net.blobs:
            net.blobs['labels'].reshape( *(blobs['labels'].shape) )
        if 'predicted_cls' in net.blobs:
            net.blobs['predicted_cls'].reshape(*(blobs['predicted_cls'].shape))

        forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}

        if 'im_info' in net.blobs:
            forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
        if 'labels' in net.blobs:
            forward_kwargs['labels'] = blobs['labels'].astype(np.float32, copy=False)
        if 'predicted_cls' in net.blobs:
            forward_kwargs['predicted_cls'] = blobs['predicted_cls'].astype(np.float32, copy=False)
        
        net.forward(end='cls_prob',**forward_kwargs)
        num_rois = net.blobs['rois'].data.shape[0]
        scores = net.blobs['cls_score'].data
        cls_score_all = scores[:, 1:].argmax(axis=1) + 1
        print "number of rois", num_rois
       
        intersection_cls = set(cls_score_all).intersection(original_cls)
        print 'intersection_cls_len= ', len(intersection_cls) , 'intersection_cls= ', intersection_cls, 'target_idx= ', target_idx
              		
        if len(intersection_cls)==0 and not targeted:
              np.save(output_file_numpy, adversarial_x)
	      boxes = net.blobs['rois'].data[:,1:5]/im_scales[0]
              probs = net.blobs['cls_prob'].data
              np.save('original_cls_'+output_file_numpy, original_cls)
              np.save('probs_'+output_file_numpy, probs)
              break

        gt_labels = np.zeros((num_rois), dtype=float)
        gt_labels += -1

        cls_temp = np.zeros((num_rois, 1), dtype=int)

        blobs['labels'] = gt_labels
        blobs['predicted_cls'] = cls_temp


        if targeted:        
             blobs['labels'][:] = new_class
             blobs['predicted_cls'][:] = new_class
        else:
             blobs['labels'][:] = target_idx
             blobs['predicted_cls'][:] = target_idx

        net.blobs['data'].reshape(*(blobs['data'].shape))

        if 'im_info' in net.blobs:
            net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
        if 'labels' in net.blobs:
            net.blobs['labels'].reshape( *(blobs['labels'].shape) )
        if 'predicted_cls' in net.blobs:
            net.blobs['predicted_cls'].reshape(*(blobs['predicted_cls'].shape))

        forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}

        if 'im_info' in net.blobs:
            forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
        if 'labels' in net.blobs:
            forward_kwargs['labels'] = blobs['labels'].astype(np.float32, copy=False)
        if 'predicted_cls' in net.blobs:
            forward_kwargs['predicted_cls'] = blobs['predicted_cls'].astype(np.float32, copy=False)

        print("starting forward pass")
        net.forward(start='roi_pool5',**forward_kwargs)
        caffe.set_mode_cpu()

        print("starting backward pass")
        grads = net.backward(diffs=['data'])
        print("backward pass done")
        grad_data = grads['data']
        grad = grad_data * lr
        grad = np.squeeze(grad)
        grad = np.transpose(grad, (1, 2, 0))
        grad = cv2.resize(grad,(adversarial_x.shape[1],adversarial_x.shape[0]),interpolation=cv2.INTER_LINEAR)
        if targeted:
             adversarial_x = np.clip(adversarial_x - grad, clip_min, clip_max)
        else:
             adversarial_x = np.clip(adversarial_x - grad, clip_min, clip_max)
         
        adversarial_x = np.clip(adversarial_x, 0.0, 255.0)
	print 'loss====== ', net.blobs['loss_cls'].data
        print 'expected loss==', np.mean(-1*np.log(net.blobs['cls_prob'].data[:, target_idx]))
        attack_try = attack_try + 1

    t1 = time.time()
    print("Time:",(t1-t0))
    f = open('results_all.txt', 'a')
    f.write(output_file_numpy + ' ' + str(attack_try) + ' ' + str(lr) + '\n')
    f.close()

    #np.save('lr'+output_file_numpy, lr)
