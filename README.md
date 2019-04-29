# yolov3-pytorch-annotation<br>
**annotation** and **specification** for yolov3 <br>  
**attention**:Inputs should be labeld the same as COCO (yolo format):normalized && cxywh  <br>

refer to:https://github.com/ultralytics/yolov3     <br>

## **detect**   <br> 
download yolov3.weights trained on COCO：https://pjreddie.com/darknet/yolo/ <br>   
run   ```python detect.py```   <br>
![Image text](https://github.com/ming71/yolov3-pytorch-annotation/blob/master/output/30.jpg)    
![Image text](https://github.com/ming71/yolov3-pytorch-annotation/blob/master/output/COCO_train2014_000000000025.jpg)   

## **train**    <br>
You can test it on offerred image in /data/dataset   <br>
run   ```python train.py```    <br>
test on the same image to verify it feasibility.   <br>

 This is my loss curls with testing on single picture for 100 epochs,it does converge well.    <br>
 Run `train-vis.py` to monitor loss.   </br>
 
![Image text](https://github.com/ming71/yolov3-pytorch-annotation/blob/master/notebook/loss.png) 
