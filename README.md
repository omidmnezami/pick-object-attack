<h3 align="center">
<p>Pick-Object-Attack
</h3>

Implementation of [Pick-Object-Attack: Type-Specific Adversarial Attack for Object Detection](https://arxiv.org/abs/2006.03184):

Many recent studies have shown that deep neural models are vulnerable to adversarial samples: images with imperceptible perturbations, for example, can fool image classifiers.  In this paper, we present the first type-specific approach to generating adversarial examples for object detection, which entails detecting bounding boxes around multiple objects present in the image and classifying them at the same time, making it a harder task than against image classification.  We specifically aim to attack the widely used Faster R-CNN by changing the predicted label for a particular object in an image: where prior work has targeted one specific object (a stop sign), we generalise to arbitrary objects, with the key challenge being the need to change the labels of all bounding boxes for all instances of that object type.  To do so, we propose a novel method, named Pick-Object-Attack.  Pick-Object-Attack successfully adds perturbations only to bounding boxes for the targeted object, preserving the labels of other detected objects in the image. In terms of perceptibility, the perturbations induced by the method are very small.  Furthermore, for the first time, we examine the effect of adversarial attacks on object detection in terms of a downstream task, image captioning;  we show that where a method that can modify all object types leads to very obvious changes in captions, the changes from our constrained attack are much less apparent.

<p align="center">
<img src="img/example_caption.png" width=800 high=600>
</p>

### Reference
If you use our source code, please cite our paper:
```
@article{MOHAMADNEZAMI2021103257,
title = {Pick-Object-Attack: Type-specific adversarial attack for object detection},
journal = {Computer Vision and Image Understanding},
volume = {211},
pages = {103257},
year = {2021},
issn = {1077-3142},
doi = {https://doi.org/10.1016/j.cviu.2021.103257},
url = {https://www.sciencedirect.com/science/article/pii/S1077314221001016},
author = {Omid {Mohamad Nezami} and Akshay Chaturvedi and Mark Dras and Utpal Garain},
keywords = {Adversarial attack, Faster R-CNN, Deep learning, Image captioning, Computer vision}
}
```

### Install

1. Clone our repository
```buildoutcfg
git clone https://github.com/omidmnezami/pick-object-attack.git
cd pick-object-attack/
```

2. Clone the bottom-up-attention repository and install the required libraries.
(You also need to download the pre-trained Faster-RCNN.)
```buildoutcfg
git clone https://github.com/peteanderson80/bottom-up-attention.git
```

3. ``resnet101_faster_rcnn_final.caffemodel`` and ``test_gradient.prototxt`` should be in ``bottom-up-attention/demo`` directory.
```buildoutcfg
# you need to do the same for resnet101_faster_rcnn_final.caffemodel after downloading
cp test_gradient.prototxt bottom-up-attention/demo
```
 
 
### Run
Scripts to run targeted/nontargeted attacks:
We run the forward pass on GPU and the backward pass on CPU.
If you have enough GPU memory, you can run both on GPU. To do so, you need to comment "caffe.set_mode_cpu()" in the corresponding pickobject code.
```buildoutcfg
# you can update the paths in code/pickobject_config.yaml if needed as your local paths
# run the targeted attack against the most confident object
python script_run_targeted.py 1

# run the targeted attack against the most frequent object
python script_run_targeted.py 0

# run the nontargeted attack against the most confident object
python script_run_nontargeted.py 1

# run the nontargeted attack against the most frequent object
python script_run_nontargeted.py 0

# run the nontargeted attack against all objects
python script_run_attack_all.py
```
