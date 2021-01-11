## Refactor of YOLOv5 by Ultralytics 

This repo is a refactor of the Yolov5 by Ultralytics repository. Model.py aims to implement a version in Pytorch Lightning, however I found the performance to be slightly worse at the end of training.
Debugging shows that it is a bug in the backward pass. Du to this, "training.py" is an object oriented implementation of the training in the spirit of PIL and "test_with_gt.y" is an object oriented 
version of testing which also adds a "tag" functionality to check the IOU for objects in the training set that are tagged as for example "truncated", "occluded", etc. This allows to check for failure cases. 

## Goal 
The goal was to make the Code of the normal Yolov5 more readable in order to implement extensions as explained above. The goal is to implement student - teacher knowledge distillation and/or teacher - teaching assistant - student distillation. 
