import argparse

from models import *
from utils.datasets import *
from utils.utils import *


def test(
        cfg,
        data_cfg,
        weights,
        batch_size=16,
        img_size=416,
        iou_thres=0.5,
        conf_thres=0.3,
        nms_thres=0.45
):
    device = torch_utils.select_device()

    # Configure run
    data_cfg_dict = parse_data_cfg(data_cfg)
    nC = int(data_cfg_dict['classes'])  # number of classes (80 for COCO)
    test_path = data_cfg_dict['valid']

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    model.to(device).eval()

    # Get dataloader
    # dataloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path), batch_size=batch_size)  # pytorch
    #该类的说明见notebook和dataset.py
    #(可改，多尺度训练和数据增广入口在load_images_and_labels()里面)
    #类似detect的load_image()，这里构造的dataloader也是一个迭代器，需要通过遍历來访问其两个返回值：torch.from_numpy(img_all), labels_all
    dataloader = LoadImagesAndLabels(test_path, batch_size=batch_size, img_size=img_size)

    mean_mAP, mean_R, mean_P = 0.0, 0.0, 0.0    #mAP,平均召回率，平均准确率
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP')) #%11s是每个字符串均占11个字符长度，向右对其，不足补空格
    outputs, mAPs, mR, mP, TP, confidence, pred_class, target_class = [], [], [], [], [], [], [], []
    AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)
    
    #通过遍历访问构造的迭代器：imgs是图片像素张量，targets是该图所有gt的张量
    #下面设涉及维度变化，遍历中间结果放在notebook展示比较清楚，直接搜for batch_i, (imgs, targets) in enumerate(dataloader)
    for batch_i, (imgs, targets) in enumerate(dataloader):
        output = model(imgs.to(device))
        output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)

        # Compute average precision for each sample
        #将预测的输出和真实gt进行压缩用于迭代（re：zip返回的是元组，这个创建迭代器真的好用）
        for sample_i, (labels, detections) in enumerate(zip(targets, output)):
            correct = []
            #没有检测出物体，但是实际gt是有物体
            if detections is None:
                # If there are no detections but there are labels mask as zero AP
                if labels.size(0) != 0: #存在图片：labels.size(0)返回的是第0维度长度，即当前batch的图片数目
                    mAPs.append(0), mR.append(0), mP.append(0)#有图片默认就有gt，但是没有检测出物体，因此p,r均为0
                continue

            # Get detections sorted by decreasing confidence scores
            #原来box是按照类内conf降序，下面转为所有类conf降序
            detections = detections.cpu().numpy()
            detections = detections[np.argsort(-detections[:, 4])]

            # If no labels add number of detections as incorrect
            if labels.size(0) == 0: #负样本，没有标签
                # correct.extend([0 for _ in range(len(detections))])
                mAPs.append(0), mR.append(0), mP.append(0)
                continue
            else:
                target_cls = labels[:, 0]   #摘出gt的类标签放入一维tensor，长度为gt的box个数

                # Extract target boxes as (x1, y1, x2, y2)
                #pred结果已经处理过是正常值，但是gt标注还是416归一化的，这里给她缩放到正常图（416）
                #xywh转成xyxy因为下面的iou计算要用xyxy
                target_boxes = xywh2xyxy(labels[:, 1:5]) * img_size

                detected = []
                #下面对一张图的每个预测的box进行遍历
                for *pred_bbox, conf, obj_conf, obj_pred in detections:
                    #（re)pred_bbox存放预测box的坐标信息，*代表传递多参数
                    pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                    # Compute iou with target boxes(传入形式要为xyxy，他这个可以(re)，参见utils)
                    #这里返回的张量不好理解，见notebook和utils.py
                    iou = bbox_iou(pred_bbox, target_boxes)
                    # Extract index of largest overlap
                    best_i = np.argmax(iou)
                    # If overlap exceeds threshold and classification is correct mark as correct
                    #(atten)这个if很仔细：1.iou大于阈值 2.类别和gt能对应上（易漏，如果不对应，那就是很严重的误检了）
                    if iou[best_i] > iou_thres and obj_pred == labels[best_i, 0] and best_i not in detected:
                        correct.append(1)       #满足上述，说明这个box预测的正确，记为1,否则0
                        detected.append(best_i) #记下这个box是属于哪gt
                    else:
                        correct.append(0)

            # Compute Average Precision (AP) per class
            AP, AP_class, R, P = ap_per_class(tp=correct, conf=detections[:, 4], pred_cls=detections[:, 6],
                                              target_cls=target_cls)

            # Accumulate AP per class
            AP_accum_count += np.bincount(AP_class, minlength=nC)
            AP_accum += np.bincount(AP_class, minlength=nC, weights=AP)

            # Compute mean AP across all classes in this image, and append to image list
            mAPs.append(AP.mean())
            mR.append(R.mean())
            mP.append(P.mean())

            # Means of all images
            mean_mAP = np.mean(mAPs)
            mean_R = np.mean(mR)
            mean_P = np.mean(mP)

        # Print image mAP and running mean mAP
        print(('%11s%11s' + '%11.3g' * 3) % (len(mAPs), dataloader.nF, mean_P, mean_R, mean_mAP))

    # Print mAP per class
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP') + '\n\nmAP Per Class:')

    classes = load_classes(data_cfg_dict['names'])  # Extracts class labels from file
    for i, c in enumerate(classes):
        print('%15s: %-.4f' % (c, AP_accum[i] / (AP_accum_count[i] + 1E-16)))

    # Return mAP
    return mean_mAP, mean_R, mean_P


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='cfg/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    with torch.no_grad():
        mAP = test(
            opt.cfg,
            opt.data_cfg,
            opt.weights,
            opt.batch_size,
            opt.img_size,
            opt.iou_thres,
            opt.conf_thres,
            opt.nms_thres
        )
