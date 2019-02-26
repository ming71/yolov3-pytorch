import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils import torch_utils

# Set printoptions
torch.set_printoptions(linewidth=1320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, 'r')
    names = fp.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

#打印出模型结构
def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '') #去掉前缀module_list.，只显示网络名称
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))     #统计参数并打印


def class_weights():  # frequency of each class in coco train2014
    weights = 1 / torch.FloatTensor(
        [187437, 4955, 30920, 6033, 3838, 4332, 3160, 7051, 7677, 9167, 1316, 1372, 833, 6757, 7355, 3302, 3776, 4671,
         6769, 5706, 3908, 903, 3686, 3596, 6200, 7920, 8779, 4505, 4272, 1862, 4698, 1962, 4403, 6659, 2402, 2689,
         4012, 4175, 3411, 17048, 5637, 14553, 3923, 5539, 4289, 10084, 7018, 4314, 3099, 4638, 4939, 5543, 2038, 4004,
         5053, 4578, 27292, 4113, 5931, 2905, 11174, 2873, 4036, 3415, 1517, 4122, 1980, 4464, 1190, 2302, 156, 3933,
         1877, 17630, 4337, 4624, 1075, 3468, 135, 1380])
    weights /= weights.sum()
    return weights


def plot_one_box(x, img, color=None, label=None, line_thickness=None):  # Plots one bounding box on image img
    #x实际传递的是向量，列表形式的[x1, y1, x2, y2];label是一个字符串，内含类别和置信度信息
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1 # line thickness（可改）线条宽度默认为2,整数输入（）
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.03)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.03)
        torch.nn.init.constant_(m.bias.data, 0.0)


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y


def scale_coords(img_size, coords, img0_shape):
    # Rescale x1, y1, x2, y2 from 416 to image size
    gain = float(img_size) / max(img0_shape)  # gain  = old / new
    pad_x = (img_size - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, :4] /= gain
    coords[:, :4] = torch.round(torch.clamp(coords[:, :4], min=0))
    return coords

#ap计算部分notebook展示过程，搜索AP计算
def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # lists/pytorch to numpy
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf), np.array(pred_cls), np.array(target_cls)

    # Sort by objectness
    #将预测信息按照conf从大到小排序(没必要，已经排好了)
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    #注意这里是对类遍历，因为是计算各类的AP
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c       #i为bool, 下面的矩阵比较和运算要注意
        n_gt = sum(target_cls == c)  # Number of ground truth objects真实gt中有几个是c这个类
        n_p = sum(i)                 # Number of predicted objects预测的box中有几个是这个c类
        #r和p都是预测正确的个数作为分子，如果没有那两者都是0
        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
        Returns the IoU of two bounding boxes
        （atten）1.这里的iou计算设计了两种输入方式，默认左上和右下的点（还可接受（x,y,w,h）），通过参数x1y1x2y2=True设置
                2.下面的坐标提取box1[:, 0]，因为传入的box就不是一维列表，而是二维张量！其第0维度是代表每个box，第一维度才是参数，
                  所以计算返回的是一个n维张量，每一维是一个gt的box和所有预测box计算的iou
         (re)返回值这里很机智：return inter_area / (b1_area + b2_area - inter_area + 1e-16)
            因为万一出现了预测box和真实gt完全重合，那么iou出现分母为0而报错。这里加一个极小的量可以避免
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    #（atten）以下要注意，算面积记得减一个像素，不然iou差别还挺大的
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


#（atten）该函数用于返回gt在yolo层的期望输出值，用于计算损失等
# (atten)再次提醒！target(s)包含的是bs张图的所有gt，是一个列表，长度为bs,每个元素是一个张量，张量才是多元的（n*5）
#传入的前面三个参数是用于评价计数的
def build_targets(target, anchor_wh, nA, nC, nG):
    """
     returns nT, nCorrect, tx, ty, tw, th, tconf, tcls
     
     pred_boxes: torch.Size([1, 3, 13, 13, 4])
     pred_conf:  torch.Size([1, 3, 13, 13])
     pred_cls:   torch.Size([1, 3, 13, 13, 80])
     target:     真实gt，列表，内含张量（每个张量多维，表征多个目标，最后一维为5：cls+xywh）
     anchor_wh:  tensor([[ 3.62500,  2.81250],
                         [ 4.87500,  6.18750],
                         [11.65625, 10.18750]])    已经缩放到特征图尺寸了
     nA, nC, nG=3, 80, 13/26/52
    """
    nB = len(target)  # number of images in batch,len返回第0维长度，也就是bs
    nT = [len(x) for x in target]     # targets per image   遍历每张图的gt tensor，len返回其中gt的个数列表
    #因为每个grid cell都要预测3个box，所以预测信息尺寸（bs,3,13,13/26/52）
    tx = torch.zeros(nB, nA, nG, nG)  # batch size (4), number of anchors (3), number of grid points (13)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)     #torch.Size([1, 3, 13, 13])
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)  # nC = number of classes,torch.Size([1, 3, 13, 13, 80]) 
    
    #b遍历bs的每张图片
    for b in range(nB):
        nTb = nT[b]     #当前遍历图片的gt(targets)个数
        if nTb == 0:
            continue
        t = target[b]   #当前图片的gt tensor

        # Convert to position relative to box
        #x,y,w,h乘以特征图尺寸，如13
        #原因：ground truth经过了原图的归一化，所有参数被限制在(0,1)相当于1*1像素点，需要放大到13*13的特征图上去
        gx, gy, gw, gh = t[:, 1] * nG, t[:, 2] * nG, t[:, 3] * nG, t[:, 4] * nG

        #gx.long()向下取整，获取网格框索引，gi，gj即左上角偏移坐标!
        #torch.clamp将结果夹紧到如[0,12]-->gt不能超出特征图大小
        gi = torch.clamp(gx.long(), min=0, max=nG - 1)
        gj = torch.clamp(gy.long(), min=0, max=nG - 1)

        # iou of targets-anchors (using wh only)
        #计算两个box的IOU
        box1 = t[:, 3:5] * nG           #gt框
        box2 = anchor_wh.unsqueeze(1)   #anchor框
        inter_area = torch.min(box1, box2).prod(2)
        iou = inter_area / (gw * gh + box2.prod(2) - inter_area + 1e-16)
        #iou的shape为torch.Size([3, n]),n为gt的个数。也就是每个anchor预测的box都与gt的box计算iou，得到3行n列的向量

        # Select best iou_pred and anchor
        #(纵向对比：选出合适尺度)找出每列（dim=0）最大值，也就是每个gt预测box在三个anchor尺度向下iou最好的那个选其尺度，返回值和索引a
        #shape:(1,n)n为gt标注物体数
        #(atten)这个a很有用，因为三个anchor理论上有三套不同的xywh，我们只预测一套，所有选与gt的iou最大的anchor为基础，得到期望的txtytwth，a就是确定使用哪个anchor的
        iou_best, a = iou.max(0)  # best anchor [0-2] for each target

        # Select best unique target-anchor combinations
        if nTb > 1: #目标物体不止一个
            # （横向排序：找出最大iou box）按照iou从高到低排序,返回索引
            iou_order = torch.argsort(-iou_best)  # best to worst

            # Unique anchor selection
            u = torch.cat((gi, gj, a), 0).view((3, -1))
            '''
             比如：gi, gj, a是tensor([4, 6, 4]) tensor([6, 7, 6]) tensor([1, 1, 1])
             输出u为：[[4 6 4]
                      [6 7 6]
                      [1 1 1]]
            '''
            #将u按照iou从高到低排序，然后返回其索引值，越靠前的列（gt），其box的iou重合越大
            _, first_unique = np.unique(u[:, iou_order], axis=1, return_index=True)  # first unique indices
            # _, first_unique = torch.unique(u[:, iou_order], dim=1, return_inverse=True)  # different than numpy?

            i = iou_order[first_unique]
            # best anchor must share significant commonality (iou) with target
            i = i[iou_best[i] > 0.10]
            if len(i) == 0:
                continue

            a, gj, gi, t = a[i], gj[i], gi[i], t[i]
            if len(t.shape) == 1:
                t = t.view(1, 5)
        else:   #图片中只有一个物体
            if iou_best < 0.10: #置信度太小就不要了
                continue
                
        #t是当前图片的gt tensor
        #取出gt的xywh，由于gt标注也是归一化过得，需要放大到13*13尺寸；tc是类标签编号
        tc, gx, gy, gw, gh = t[:, 0].long(), t[:, 1] * nG, t[:, 2] * nG, t[:, 3] * nG, t[:, 4] * nG

        # Coordinates
        #真实的gt减去其左上定位，得到gt的归一化相对坐标，该坐标只给预测iou最大的box的那个位置
        tx[b, a, gj, gi] = gx - gi.float()
        ty[b, a, gj, gi] = gy - gj.float()

        # Width and height (yolo method)
        #用预测iou最大的box的anchor尺度进行处理和取对数，输出的tw,th和tx,ty一样，是期望的输出值
        tw[b, a, gj, gi] = torch.log(gw / anchor_wh[a, 0])
        th[b, a, gj, gi] = torch.log(gh / anchor_wh[a, 1])

        # Width and height (power method)
        # tw[b, a, gj, gi] = torch.sqrt(gw / anchor_wh[a, 0]) / 2
        # th[b, a, gj, gi] = torch.sqrt(gh / anchor_wh[a, 1]) / 2

        # One-hot encoding of label
        #tc是类标签编号如0-79,在预测的最大iou那个类别上One-hot编码
        tcls[b, a, gj, gi, tc] = 1
        tconf[b, a, gj, gi] = 1

    return tx, ty, tw, th, tconf, tcls

#notebook有过程分析
def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    #prediction.shape例如torch.Size([1, 65, 85])，第0维扩充了维度
    #pred.shape例如torch.Size([65, 85]) 65为box数目，85为4+1+80
    
    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # Get score and class with highest confidence

        # cross-class NMS (experimental)
        cross_class_nms = False
        if cross_class_nms:
            a = pred.clone()
            _, indices = torch.sort(-a[:, 4], 0)  # sort best to worst
            a = a[indices]
            radius = 30  # area to search for cross-class ious
            for i in range(len(a)):
                if i >= len(a) - 1:
                    break

                close = (torch.abs(a[i, 0] - a[i + 1:, 0]) < radius) & (torch.abs(a[i, 1] - a[i + 1:, 1]) < radius)
                close = close.nonzero()

                if len(close) > 0:
                    close = close + i + 1
                    iou = bbox_iou(a[i:i + 1, :4], a[close.squeeze(), :4].reshape(-1, 4), x1y1x2y2=False)
                    bad = close[iou > nms_thres]

                    if len(bad) > 0:
                        mask = torch.ones(len(a)).type(torch.ByteTensor)
                        mask[bad] = 0
                        a = a[mask]
            pred = a

        # Experiment: Prior class size rejection
        # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # a = w * h  # area
        # ar = w / (h + 1e-16)  # aspect ratio
        
        # n = len(w)
        # log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] =
        #   multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])
        
        #返回类概率最大值及其索引，反映了每个box最可能属于哪一类
        class_prob, class_pred = torch.max(F.softmax(pred[:, 5:], 1), 1)

        v = ((pred[:, 4] > conf_thres) & (class_prob > .4))  # TODO examine arbitrary 0.4 thres here（可改，自行设定类筛选概率）
        v = v.nonzero().squeeze()   #返回bool矩阵为真（满足阈值条件）的索引
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]      #返回满足条件的box及其最大类概率及其所属的类
        class_prob = class_prob[v]
        class_pred = class_pred[v]

        # If none are remaining => process next image
        nP = pred.shape[0]  #留下的box数目
        if not nP:          #所有box都挂掉了，跳出循环，检测下一张
            continue

        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        #输入向量为（x,y,w,h）
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_prob, class_pred)
        detections = torch.cat((pred[:, :5], class_prob.float().unsqueeze(1), class_pred.float().unsqueeze(1)), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()    #获取检测出的类标签
        if prediction.is_cuda:  #如果跑的gpu，放到gpu加速
            unique_labels = unique_labels.cuda(prediction.device)
            
        #notebook见(atten关于为什么加维度在该部分有解释，搜索atten)
        nms_style = 'MERGE'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        for c in unique_labels:
            # Get the detections with class c
            dc = detections[detections[:, -1] == c]
            # Sort the detections by maximum object confidence
            _, conf_sort_index = torch.sort(dc[:, 4] * dc[:, 5], descending=True)
            dc = dc[conf_sort_index]

            # Non-maximum suppression
            det_max = []
            if nms_style == 'OR':  # default
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(det_max[-1], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

                # Image      Total          P          R        mAP
                #  4964       5000      0.629      0.594      0.586

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[:1], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc) > 0:
                    iou = bbox_iou(dc[:1], dc[0:])  # iou with other boxes
                    i = iou > nms_thres

                    weights = dc[i, 4:5] * dc[i, 5:6]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[iou < nms_thres]

                # Image      Total          P          R        mAP
                #  4964       5000      0.633      0.598      0.589  # normal

            if len(det_max) > 0:
                det_max = torch.cat(det_max)
                # Add max detections to outputs
                output[image_i] = det_max if output[image_i] is None else torch.cat((output[image_i], det_max))

    return output


def strip_optimizer_from_checkpoint(filename='weights/best.pt'):
    # Strip optimizer from *.pt files for lighter files (reduced by 2/3 size)
    import torch
    a = torch.load(filename, map_location='cpu')
    a['optimizer'] = []
    torch.save(a, filename.replace('.pt', '_lite.pt'))


def coco_class_count(path='../coco/labels/train2014/'):
    # histogram of occurrences per class
    import glob

    nC = 80  # number classes
    x = np.zeros(nC, dtype='int32')
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        x += np.bincount(labels[:, 0].astype('int32'), minlength=nC)
        print(i, len(files))


def plot_results():
    # Plot YOLO training results file 'results.txt'
    import glob
    import matplotlib.pyplot as plt
    import numpy as np
    # import os; os.system('rm -rf results.txt && wget https://storage.googleapis.com/ultralytics/results_v1_0.txt')

    plt.figure(figsize=(16, 8))
    s = ['X', 'Y', 'Width', 'Height', 'Confidence', 'Classification', 'Total Loss', 'mAP', 'Recall', 'Precision']
    files = sorted(glob.glob('results*.txt'))
    for f in files:
        results = np.loadtxt(f, usecols=[2, 3, 4, 5, 6, 7, 8, 11, 12, 13]).T  # column 13 is mAP
        n = results.shape[1]
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.plot(range(1, n), results[i, 1:], marker='.', label=f)
            plt.title(s[i])
            if i == 0:
                plt.legend()
