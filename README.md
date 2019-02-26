# yolov3-pytorch-annotation
**annotation** and **specification** for yolov3   
**attention**:Inputs should be labeld the same as COCO (yolo format):normalized && cxywh  

refer to:https://github.com/ultralytics/yolov3   

**detect**
download yolov3.weights trained on COCO：https://pjreddie.com/darknet/yolo/
run ‘’‘python detect.py’‘’
![Image text](https://github.com/ming71/yolov3-pytorch-annotation/blob/master/output/30.jpg) 
![Image text](https://github.com/ming71/yolov3-pytorch-annotation/blob/master/output/COCO_train2014_000000000025.jpg) 

**train**
You can test it on offerred image in /data/dataset
run ‘’‘python train.py’‘’
test on the same image to verify it feasibility.

 This is my loss curls with testing on single picture for 100 epochs,it does converge well.   
 Run train-vis.py to monitor loss.   
 
![Image text](https://github.com/ming71/yolov3-pytorch-annotation/blob/master/notebook/loss.png) 
