import argparse
import shutil
import time
from pathlib import Path
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *


def detect(
        cfg,                #cfg配置文件夹路径：coco.data && yolov3.cfg
        weights,            #权值路径
        images,             #图片/文件夹路径
        output='output',    # output folder
        img_size=416,
        conf_thres=0.3,
        nms_thres=0.45,
        save_txt=False,
        save_images=True,
        webcam=True
):
    device = torch_utils.select_device()    #返回运行设备(torch_utils的set_device)并打印
    if os.path.exists(output):
        shutil.rmtree(output)   # delete output folder
    os.makedirs(output)         # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)  #模型类实例化__init__()传入两个参数，cfg_path和size

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        if weights.endswith('yolov3.pt') and not os.path.exists(weights):   #键入的pt文件不存在
            if (platform == 'darwin') or (platform == 'linux'):
                os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights)#把pt权值文件下载到指定文件夹下
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    #(atten)这个很有必要，训练和测试时一定要把实例化的model指定train/eval
    model.to(device).eval()

    # Set Dataloader
    if webcam:      #摄像头优先
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size)#dataloader是迭代器，通过遍历來访问其两个返回值：[img_path]和img

    # Get classes and colors    #（re）根据类别数目随机生成对应数目的rgb颜色组（二维列表）
    classes = load_classes(parse_data_cfg('cfg/coco.data')['names'])#返回一个列表存储coco.names里面的80类名
    colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

    #除了索引i,还返回被迭代后执行__next__ return的两个值：[img_path], img（参见注释dataset.py有注释）
    for i, (path, img, im0) in enumerate(dataloader):
        if webcam:
            print('webcam frame %g: ' % (i + 1), end='')
        else:
            print('image %g/%g %s: ' % (i + 1, len(dataloader), path), end='')
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        t = time.time()
        img = torch.from_numpy(img).unsqueeze(0).to(device) #将ndarray转Tensor;加入了额外的维度第0维;放到gpu
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return
        #通过model(img)自动调用类的forward方法计算损失，检测模式返回（1,10647,85）的预测结果张量
        #取第三个维度（每个box的信息）的第5个元素（4+1+80,第五个为confidence）阈值比较,留下大于conf阈值的box，得到二维张量（[box_num,85]）
        #(atten)经过前向传播得到的xywh实际上已经是在416*416尺寸上的参数!
        pred = model(img)
        pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshold

        if len(pred) > 0:   # len选取张量的第一维度（box_num）确定存在confidence大于阈值的box再NMS
            
            # Run NMS on predictions
            '''
                （atten）这里的nms实际含有3步：
                1.最大类概率提取，将每个box的80个类别信息压缩为1个，取其最大概率的一个类，并且加上该类的标签编号，所以85->7
                2.thre筛选掉confidence小于阈值box
                3.NMS筛掉重叠度高于阈值nms_thre的框
            '''
            #pred.unsqueeze(0)给二维张量在第0维扩充一个维度
            #detections维度1为：(x1, y1, x2, y2, obj_conf, class_prob, class_pred)，按照标签&confidence排序
            detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]
            # Rescale boxes from 416 to true image size416到真实尺寸
            detections[:, :4] = scale_coords(img_size, detections[:, :4], im0.shape)

            # Print results to screen
            unique_classes = detections[:, -1].cpu().unique()
            for c in unique_classes:
                n = (detections[:, -1].cpu() == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for x1, y1, x2, y2, conf, cls_conf, cls in detections:
                #写txt  Write to file
                if save_txt:  
                    with open(save_path + '.txt', 'a') as file:
                        file.write('%g %g %g %g %g %g\n' %
                                   (x1, y1, x2, y2, cls, cls_conf * conf))

                # Add bbox to the image
                #（re）画box就完事了，实现注释见notebook:box绘制
                #作者这个调用的opencv字库，丑...自己找字体贴会好看很多，参见qwe大神的keras版本字体实现
                #当然，一次画一个box，但是上面对 detections进行了for遍历，循环画出一张图每个box
                label = '%s %.2f' % (classes[int(cls)], conf)   #索引值对应的类别，置信度（图片显示）
                plot_one_box([x1, y1, x2, y2], im0, label=label, color=colors[int(cls)])

        dt = time.time() - t
        print('Done. (%.3fs)' % dt, end=' ')

        if save_images:  # Save generated image with detections
            cv2.imwrite(save_path, im0)

        if webcam:  # Show live webcam
#            cv2.imshow(weights + ' - %.2f FPS' % (1 / dt), im0)#闪烁，将FPS打印即可
             print(' %.2f FPS' % (1 / dt))
             cv2.imshow('detection', im0)
             
    if save_images and (platform == 'darwin'):  # linux/macos
        os.system('open ' + output + ' ' + save_path)


if __name__ == '__main__':
    '''
    (re)
     argparse解析命令参数步骤：
        创建 ArgumentParser() 对象
        调用 add_argument() 方法添加参数
        使用 parse_args() 解析添加的参数
    '''
    #部分非设置参数在函数体中默认初始化，可以到上面改
    #confidence：需要的框较多，则调低阈值，需要的框较少，则调高阈值
    #nms:检测的iou阈值，大于阈值的重叠框被删除，只最终保留一个（改）重叠物体较多，则调高阈值，重叠物体较少，则调低阈值
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    #opt包含所有的参数名（主要关注添加的），可以对其下的参数名进行方法调用，如opt.image-folder取得图片输入路径
    #返回方式是类似键值对形式（但实际不是字典），包含参数名和传递的参数，如：image-folder='data/samples',batch-size=11等
    print(opt)

    torch.cuda.empty_cache()    #(re)清除没用的临时变量的，释放缓存，加快速度
    
    with torch.no_grad():
        detect(
            opt.cfg,
            opt.weights,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres
        )
