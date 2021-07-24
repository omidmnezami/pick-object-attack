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

#caffe.set_mode_gpu()
#caffe.set_device(0)


cfg_from_file('experiments/cfgs/faster_rcnn_end2end_resnet.yml')

#print cfg['TEST']['RPN_POST_NMS_TOP_N'], 'before'
#cfg['TEST']['SCALES'] = (125,)
#cfg['TEST']['RPN_POST_NMS_TOP_N'] = 30
#print cfg['TEST']['RPN_POST_NMS_TOP_N'], 'after'
#sys.exit()

if __name__ == '__main__':
    t0 = time.time()
    input_file = sys.argv[1]
    #target_idx = int(sys.argv[2]) # target class
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
    weights = 'demo/resnet101_faster_rcnn_final.caffemodel'
    prototxt_test = 'demo/test.prototxt'
    prototxt_train = 'demo/test_gradient.prototxt'

    ## specifiy min and max changes
    eps = 200

    net = caffe.Net(prototxt_train, caffe.TEST, weights=weights)

    MEANS = np.array([[102.9801, 115.9465, 122.7717]])

    data_path = '../bottom-up-attention/data/genome/1600-400-20'
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
    a=net.blobs['cls_score'].data[:, 1:]
    print np.unravel_index(a.argmax(), a.shape)
    target_idx= np.unravel_index(a.argmax(), a.shape)[1]+1

    output_file_numpy = str(target_idx) + '_' + output_file_numpy
    #sys.exit()

    #target_idx = Counter(cls_score_all).most_common()[0][0]
    #print target_idx, Counter(cls_score_all).most_common()
    
    mul_attack_box_indx = np.where(cls_score_all == target_idx)[0]

    print cls_score_all
    print cls_score_all.shape
    print np.where(cls_score_all==target_idx)
    print len(np.where(cls_score_all == target_idx)[0])
    print classes[target_idx], "the label of the targetted index"

    cls_boxes = net.blobs['rois'].data[:,1:5] / im_scales[0]
    m = np.zeros(adversarial_x.shape)

    for ind in np.where(cls_score_all==target_idx)[0]:
        mul_attack_box = cls_boxes[ind]
        print mul_attack_box
        m[int(math.floor(mul_attack_box[1])):int(math.floor(mul_attack_box[3])),
        int(math.floor(mul_attack_box[0])):int(math.floor(mul_attack_box[2])), :] = 1
    
    attack_try = 0
    
    #caffe.set_device(1)
    #net = caffe.Net(prototxt_train, caffe.TEST, weights=weights)

    #caffe.set_device(2)
    #net = caffe.Net(prototxt_train, caffe.TEST, weights=weights)
 
    while (True):
        
        caffe.set_mode_gpu()
        caffe.set_device(gpuID)

        #print "set device 1=============="
        #caffe.set_device(1)
        #print "after set device 1 ======="
        #net = caffe.Net(prototxt_train, caffe.TEST, weights=weights)

        if (( attack_try +1 ) % 60 == 0): # change it to 10 later
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
        mul_attack_box_indx = np.where(cls_score_all == target_idx)[0]
        print "number of bounding boxes for attack",len(mul_attack_box_indx)
        print "number of rois", num_rois

        # Think about what extra condition to add here if it's a targeted attack e.g. 'ocean'(index = 65) for the current image
        if len(mul_attack_box_indx)==0 and targeted:
              num_boxes = np.where(cls_score_all == new_class)[0]
              print "number of boxes with target label",len(num_boxes)
              probs = net.blobs['cls_prob'].data[num_boxes,new_class]
              print("probabilities for target class:",probs)
              if len(num_boxes) > 0:#and np.max(probs)>0.6:
                   np.save(output_file_numpy, adversarial_x)
                   np.save('probs_'+output_file_numpy, probs)
                   break
              else:
                   #mul_attack_box_indx = np.array([scores[:,new_class].argmax(axis=0)])
                   #box = net.blobs['rois'].data[mul_attack_box_indx]
                   #x1, y1, x2, y2 = box[0][1:5]
                   #print np.any(m[int(math.floor(x1)):int(math.floor(x2)),int(math.floor(y1)):int(math.floor(y2))])
		   #print box, 'box========='
                   #print "new bounding box for attack", mul_attack_box_indx, "new bounding box for attack"
		   boxes = net.blobs['rois'].data[:,1:5]/im_scales[0]
		   #boxes_mask = filter(lambda x:np.any(m[int(math.floor(x[0])):int(math.floor(x[2])),int(math.floor(x[1])):int(math.floor(x[3]))]), boxes)
		   #print '============', boxes_mask, 'boxes mask================'
                   indexes = [i for i,x in enumerate(boxes) if np.any(m[int(math.floor(x[1])):int(math.floor(x[3])),int(math.floor(x[0])):int(math.floor(x[2]))])]
		   print 'indexes', indexes, 'indexes'
            	   indexes = np.array(indexes)
	           mul_attack_box_indx = indexes[np.array([scores[indexes,new_class].argmax(axis=0)])]
		   print "new bounding box for attack", mul_attack_box_indx, "new bounding box for attack"
		   #x = net.blobs['rois'].data[mul_attack_box_indx[0], 1:5]/im_scales[0]
	           #print m[int(math.floor(x[1])):int(math.floor(x[3])),int(math.floor(x[0])):int(math.floor(x[2]))]
		   #print x, m.shape 
              		
        elif len(mul_attack_box_indx)==0 and not targeted:
              np.save(output_file_numpy, adversarial_x)
              break

        #print "set device 0==========="
        #caffe.set_device(0)
        #net = caffe.Net(prototxt_train, caffe.TEST, weights=weights)
        #caffe.set_mode_cpu()

        gt_labels = np.zeros((len(mul_attack_box_indx)), dtype=float)
        gt_labels += -1

        cls_temp = np.zeros((len(mul_attack_box_indx),1), dtype=int)
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

        for key in net.blobs.keys():
            if len(net.blobs[key].data.shape)>0 and net.blobs[key].data.shape[0] == num_rois:
                arr_np = net.blobs[key].data[mul_attack_box_indx,:]
                net.blobs[key].reshape(*(arr_np.shape))
                net.blobs[key].data[...] = arr_np
       
        #print "set device 0==========="
        #caffe.set_device(0) 
        print("starting forward pass")
        net.forward(start='roi_pool5',**forward_kwargs)
        caffe.set_mode_cpu()
        #print "set device 2==========="

        #caffe.set_device(gpuID+1)
        #caffe.set_solver_count(2)
        #caffe.set_solver_rank(gpuID+1)
        #caffe.set_multiprocess(True)

        print("starting backward pass")
        grads = net.backward(diffs=['data'])
        print("backward pass done")
        #caffe.set_mode_gpu()
        grad_data = grads['data']
        grad = grad_data * lr
        grad = np.squeeze(grad)
        grad = np.transpose(grad, (1, 2, 0))
        grad = cv2.resize(grad,(adversarial_x.shape[1],adversarial_x.shape[0]),interpolation=cv2.INTER_LINEAR)
        print np.nonzero(grad), 'image change before ========='
	grad *= m
        print np.nonzero(grad), 'image change after ========='
        if targeted:
             adversarial_x = np.clip(adversarial_x - grad, clip_min, clip_max)
        else:
             adversarial_x = np.clip(adversarial_x + grad, clip_min, clip_max)
         
        adversarial_x = np.clip(adversarial_x, 0.0, 255.0)
        print 'loss====== ', net.blobs['loss_cls'].data
        print 'expected loss==', np.mean(-1*np.log(net.blobs['cls_prob'].data[:, target_idx]))
        print 'learning rate==', lr
        attack_try = attack_try + 1

    t1 = time.time()
    print("Time:",(t1-t0))
    f = open('results.txt', 'a')
    f.write(output_file_numpy + ' ' + str(attack_try) + ' ' + str(lr) + '\n')
    f.close()

    #np.save('lr'+output_file_numpy, lr)
