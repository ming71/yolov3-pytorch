import glob
import math
import os
import random

import cv2
import numpy as np
import torch

# from torch.utils.data import Dataset
from utils.utils import xyxy2xywh


class LoadImages:  # for inference
    def __init__(self, path, img_size=416):     #传入参数为输入图片文件夹，batch_size和img_size
        if os.path.isdir(path):                 #文件夹读取，返回其下包含所有文件路径的列表，见notebook
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(path):              #也可以直接添加图片路径
            self.files = [path]

        self.nF = len(self.files)       # number of image files
        self.height = img_size

        assert self.nF > 0, 'No images found in ' + path    #判断是否文件存在
    
    # RGB normalization values
    def __iter__(self):      #这里的迭代器不太好理解，大概就是，初始化为count=-1，只进行一次，创建迭代对象
        self.count = -1      #__iter__返回的self实例带着属性count=-1会进入__next__并且不再回头直到__next__抛出异常，下次进入重新创建迭代对象
        return self
    
    #(atten)类实例化后只要被迭代，每次迭代后return [img_path], img两个元素! 单个路径的列表形式和归一化并翻转后的像素矩阵（(c,w,h)和(rgb)）
    def __next__(self):             #__iter__传过来的带count=-1的实例self在这里一直循环到抛出异常（遍历完毕，索引越界）
        self.count += 1             #每次count+1，因此第一次运行的count实际是0
        if self.count == self.nF:   #跑完全部文件，抛出异常，结束本次迭代
            raise StopIteration
        img_path = self.files[self.count]   #count为索引从0开始

        # Read image
        img0 = cv2.imread(img_path) #(atten)注意为BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        #返回的img为按长边保持比例缩放到416后，放到图片中间，边缘填充灰色（因为color=(127.5, 127.5, 127.5))）的像素数组
        #其他三个是缩放比例ratio,宽高的总填充量
        img, _, _, _ = letterbox(img0, height=self.height)

        # Normalize RGB
        #色彩数值归一化，并且翻转数据，最终结构是(c,w,h)和(rgb)
        img = img[:, :, ::-1].transpose(2, 0, 1)    #(re)像素矩阵的翻转，比较麻烦，参见notebook注释
        img = np.ascontiguousarray(img, dtype=np.float32)
        # img -= self.rgb_mean
        # img /= self.rgb_std           #（可改）这两行代码还可以用于标准化处理        
        img /= 255.0                    # 归一化

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img, img0

    def __len__(self):      # 自定义len()方法，调用len()时会返回file数目
        return self.nF      # number of files


#调用摄像头
class LoadWebcam:  # for inference
    def __init__(self, img_size=416):
        self.cam = cv2.VideoCapture(0)
        self.height = img_size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == 27:  # esc to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Read image
        ret_val, img0 = self.cam.read()
        assert ret_val, 'Webcam Error'
        img_path = 'webcam_%g.jpg' % self.count
        img0 = cv2.flip(img0, 1)

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __len__(self):
        return 0

#（re）这个函数实现了很多数据增广的操作，值得借鉴
#部分中间变量见notebook
class LoadImagesAndLabels:  # for training
    def __init__(self, path, batch_size=1, img_size=608, multi_scale=False, augment=False):     #初始化类需要传入的参数
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [x.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
                            for x in self.img_files]

        self.nF = len(self.img_files)               # number of image files
        self.nB = math.ceil(self.nF / batch_size)   # number of batches
        self.batch_size = batch_size
        self.height = img_size
        self.multi_scale = multi_scale
        self.augment = augment

        assert self.nB > 0, 'No images found in %s' % path
    
    #迭代器初始化
    def __iter__(self):
        self.count = -1
        #如果开启了数据增广，产生随机序列，后面用于打乱图片；否则是一个有序的一维的array向量，长度为图片的数目
        self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
        return self
    #循环主体
    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration
            #这个是遍历起点终点

        ia = self.count * self.batch_size
        ib = min((self.count + 1) * self.batch_size, self.nF)
        #多尺度训练
        if self.multi_scale:
            # Multi-Scale YOLO Training
            height = random.choice(range(10, 20)) * 32  # 320 - 608 pixels
        else:
            # Fixed-Scale YOLO Training
            height = self.height

        img_all = []
        labels_all = []
        #range(ia, ib)遍历的是当前count计数的batch内的图片，
        #比如batch_size=3,第一个count=0它遍历为0-2;count=1时，遍历的是3-5
        #这样好处是遍历完图片的同时，返回的索引具有连贯性
        for index, files_index in enumerate(range(ia, ib)):
        #index是当前batch的索引，files_index是连续的索引，用于索引对应的图片
            img_path = self.img_files[self.shuffled_vector[files_index]]
            #（改）label的路径和图片在一起，这个最好改一下
            label_path = self.label_files[self.shuffled_vector[files_index]]

            img = cv2.imread(img_path)  # BGR
            if img is None:
                continue

            augment_hsv = True
             #（re）hsv色彩空间变换
            if self.augment and augment_hsv:
            # SV augmentation by 50%：饱和度和明度提高50%
                fraction = 0.50
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  #色彩空间变换
                #类似于BGR，HSV的shape=(w,h,c)，其中三通道的c[0,1,2]含有h,s,v信息
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)

                a = (random.random() * 2 - 1) * fraction + 1
                S *= a
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)

                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

            h, w, _ = img.shape
            #img是边界填充灰色后的height*height像素矩阵;缩放比ratio ;填充数目dw//2 dh//2
            img, ratio, padw, padh = letterbox(img, height=height)

            # Load labels
            #label文件是yolo格式：class ,x,y,w,h（）后四者经过了原图的归一化
            if os.path.isfile(label_path):
                #np.loadtxt()直接将txt文件内容按行存成矩阵
                #按照信息放成5列
                labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)

                # Normalized xywh to pixel xyxy format
                #（atten）这里将yolo格式的xywh转换为xyxy是为了便于下面的仿射变换（后面还会改回来的；这个坐标的变换比较乱）
                labels = labels0.copy()
                #由于图片进行了缩放，所以gt的box也要按照ritio缩放
                #（atten）这里注意了，为什么乘以w和h？因为他的gt坐标信息(xywh)是按照原图尺寸归一化过的（可以，这很yolo,和人家的一样了）
                #相当于使用的是转化好的yolo格式标签
                labels[:, 1] = ratio * w * (labels0[:, 1] - labels0[:, 3] / 2) + padw
                labels[:, 2] = ratio * h * (labels0[:, 2] - labels0[:, 4] / 2) + padh
                labels[:, 3] = ratio * w * (labels0[:, 1] + labels0[:, 3] / 2) + padw
                labels[:, 4] = ratio * h * (labels0[:, 2] + labels0[:, 4] / 2) + padh
            else:
                labels = np.array([])

            # Augment image and labels
            #仿射变换（注意gt的box也要改）：实际包含若干仿射变换内容，random_affine()
            #返回的M是综合的仿射变换矩阵
            if self.augment:
                img, labels, M = random_affine(img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))
            #(改)画图在这里被注销了
            plotFlag = False
            if plotFlag:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 10)) if index == 0 else None
                plt.subplot(4, 4, index + 1).imshow(img[:, :, ::-1])
                plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [2, 2, 4, 4, 2]].T, '.-')
                plt.axis('off')

            nL = len(labels)    #ground truth的数目
            if nL > 0:
                # convert xyxy to xywh
                #上面为了仿射变换改了坐标为xyxy，这里又改回xywh(atten，再次强调注意深拷贝copy)
                #这次ground truth是对缩填充放后的图片为准进行归一化（不再是原图）
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5].copy()) / height

            if self.augment:
                # random left-right flip
                #（rondom）随机左右翻转（当然，除图像外也要gt的box同样操作）
                lr_flip = True
                if lr_flip & (random.random() > 0.5):
                    img = np.fliplr(img)    #(atten)有函数直接翻转
                    if nL > 0:
                        labels[:, 1] = 1 - labels[:, 1]

                # random up-down flip
                #随机上下翻转
                ud_flip = False
                if ud_flip & (random.random() > 0.5):
                    img = np.flipud(img)
                    if nL > 0:
                        labels[:, 2] = 1 - labels[:, 2]
            #对一张图片遍历结束，列表存入图片信息矩阵和一个包含这张图所有gt的张量，notebook见
            img_all.append(img)
            labels_all.append(torch.from_numpy(labels))

        # Normalize
        img_all = np.stack(img_all)[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB and cv2 to pytorch
        img_all = np.ascontiguousarray(img_all, dtype=np.float32)
        #（改）可选，中心化
        # img_all -= self.rgb_mean
        # img_all /= self.rgb_std
        img_all /= 255.0

        return torch.from_numpy(img_all), labels_all

    def __len__(self):
        return self.nB  # number of batches


def letterbox(img, height=416, color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded square
    shape = img.shape[:2]                # shape = [height, width]!
    ratio = float(height) / max(shape)   # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  #计算按照长边保持比例缩放到height(416)的新shape列表：[new_height, new_width]
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding缩放后图片填充到416*416,水平垂直方向各两侧需要填充的量
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border开始缩放：将原图按比例以长边缩放到416
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square填充color为灰色(127.5, 127.5, 127.5)成416*416
    return img, ratio, dw, dh
    #返回值：
    #1.边界填充后的数组（不是img,而是cv2.copyMakeBorder(xxx),也就是填充后的图片像素矩阵）
    #2.缩放比ratio 
    #3.填充数目dw//2 dh//2
    #见notebook图片展示


#仿射变换包括：
#Translation	+/- 10% (vertical and horizontal)
#Rotation	+/- 5 degrees
#Shear	+/- 2 degrees (vertical and horizontal)
#Scale	+/- 10%
def random_affine(img, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(height, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 1:5].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            np.clip(xy, 0, height, out=xy)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 1:5] = xy[i]

        return imw, targets, M
    else:
        return imw


def convert_tif2bmp(p='../xview/val_images_bmp'):
    import glob
    import cv2
    files = sorted(glob.glob('%s/*.tif' % p))
    for i, f in enumerate(files):
        print('%g/%g' % (i + 1, len(files)))
        cv2.imwrite(f.replace('.tif', '.bmp'), cv2.imread(f))
        os.system('rm -rf ' + f)
