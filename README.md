

# 
<h3 align="center">
<p>Pick-Object-Attack
</h3>

Implementation of [Pick-Object-Attack: Type-Specific Adversarial Attack for Object Detection](https://arxiv.org/abs/2006.03184):

Many recent studies have shown that deep neural models are vulnerable to adversarial samples: imageswith imperceptible perturbations, for example, can fool image classifiers.  In this paper, we presentthe first type-specific approach to generating adversarial examples for object detection, which entailsdetecting bounding boxes around multiple objects present in the image and classifying them at thesame time, making it a harder task than against image classification.  We specifically aim to attackthe widely used Faster R-CNN by changing the predicted label for a particular object in an image:where prior work has targeted one specific object (a stop sign), we generalise to arbitrary objects, withthe key challenge being the need to change the labels ofallbounding boxes for all instances of thatobject type.  To do so, we propose a novel method, named Pick-Object-Attack.  Pick-Object-Attacksuccessfully adds perturbations only to bounding boxes for the targeted object, preserving the labels ofother detected objects in the image. In terms of perceptibility, the perturbations induced by the methodare very small.  Furthermore, for the first time, we examine the effect of adversarial attacks on objectdetection in terms of a downstream task, image captioning;  we show that where a method that canmodify all object types leads to very obvious changes in captions, the changes from our constrainedattack are much less apparent.
<p align="center">
<img src="img/example_caption.png" width=800 high=600>
</p>
