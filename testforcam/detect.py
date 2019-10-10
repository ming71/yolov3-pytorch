import argparse
from sys import platform

from models import *  
from utils.datasets import *
from utils.utils import *
from utils.draw import *

import test 

# features = []



def GetBAM(area_fp, weights,img,index,agnostic=False):
    output_bam = {}
    height, width, _ = img.shape

    scale = np.max(img.shape)/416    # 原图到长边416的缩放比例

    area_fp = area_fp.cpu().numpy()
    cx = area_fp[...,0]*scale       # xywh只是从特征图放大到416,还需进一步放大到原图
    cy = area_fp[...,1]*scale
    w  = area_fp[...,2]*scale
    h  = area_fp[...,3]*scale
    yv,xv = np.meshgrid(np.arange(area_fp.shape[1]),np.arange(area_fp.shape[0]))
    grid = np.stack((yv,xv), 2)
    coors = grid.reshape(-1,2)

    # weights = None
    if weights is None:
        img_weights = np.ones((1,height,width))
    else:
        img_weights = []
        for weight in weights:
            weight = weight.cpu().numpy()
            img_weights.append(cv2.resize(weight,(width,height)))   

    img_yv,img_xv = np.meshgrid(np.arange(width),np.arange(height))
    img_grid = np.stack((img_yv,img_xv), 2)
    img_coors = img_grid.reshape(-1,2)

    bam = np.zeros((height,width))
    if not agnostic:
        for cnt,cls_w in enumerate(img_weights):  
            for i,j in zip(coors[:,1],coors[:,0]):
                top    = int(np.maximum(0,     cy[i,j] - h[i,j]))
                bottom = int(np.minimum(height,cy[i,j] + h[i,j]))
                left   = int(np.maximum(0,     cx[i,j] - w[i,j]))
                right  = int(np.minimum(width, cx[i,j] + w[i,j]))
                bam[top:bottom,left:right] += cls_w[top:bottom,left:right]

                output_bam[index+str(cnt)] = bam
        
    else:
        max_weight_map = np.array(img_weights).transpose((1,2,0)).max(-1)
        for i,j in zip(coors[:,1],coors[:,0]):
            top    = int(np.maximum(0,     cy[i,j] - h[i,j]))
            bottom = int(np.minimum(height,cy[i,j] + h[i,j]))
            left   = int(np.maximum(0,     cx[i,j] - w[i,j]))
            right  = int(np.minimum(width, cx[i,j] + w[i,j]))
            bam[top:bottom,left:right] += max_weight_map[top:bottom,left:right]

            output_bam[index] = bam

    return output_bam






#  输入的feature_maps支持 2D tensor和list嵌套的2D tensor
def GetCAM(feature_maps,img,index,agnostic=False):
    output_cam = {}
    height , width = img.shape[:2]
    if torch.is_tensor(feature_maps):
        feature_maps = [feature_maps.cpu().numpy()]
    elif  isinstance(feature_maps,list) :
        feature_maps = [f.cpu().numpy() for f in feature_maps]
    else:
        print('Not supported type of feature_maps')

    if not agnostic:
        for cnt,cam in enumerate(feature_maps):
            cam = cv2.resize(cam,(width, height))
            output_cam[index+str(cnt)] = cam
    else:
        max_cls_map = np.array(feature_maps).transpose((1,2,0)).max(-1)
        cam = cv2.resize(max_cls_map,(width, height))
        output_cam[index] = cam

    return output_cam




def DrawMap(maps,img):
    num = len(maps)*len(maps[0])
    height , width = img.shape[:2]

    max_value = 0
    min_value = 0
    for map in maps: 
        for id,cls_map in map.items():
            min_value = np.min(cls_map) if  np.min(cls_map) < min_value else min_value
            max_value = np.max(cls_map) if  np.max(cls_map) > max_value else max_value

    for map in maps:
        for id,p in map.items():
            p = p - min_value     
            p_img = p / max_value
            p_img = np.uint8(255 * p_img)
            
            heatmap = cv2.applyColorMap(p_img, cv2.COLORMAP_JET)  
            result = heatmap * 0.5 + img * 0.5
            cv2.imwrite('/py/yolov3/cam/{}.jpg'.format(id), result)




# def yolo_hook(module,input,output):
#     if features==[] :
#         print('input shape: {} \n output[0] shape: {} \n output[1] shape: {} \n output[2] shape: {}'\
#             .format(input[0].shape, output[0].shape, output[1].shape, output[2].shape))
#         features.append(output[1])
#     else :
#         pass


def detect(save_txt=False, save_img=False):
    img_size =  opt.img_size  

    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    # os.makedirs(out)  # make new output folder

    model = Darknet(opt.cfg, img_size)  # 搭建模型（不连接计算图），只调用构造函数

    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)


    model.to(device).eval()

    vid_path, vid_writer = None, None
    save_img = True
    dataset = LoadImages(source, img_size=img_size, half=half)  # source是测试的文件夹路径，返回的dataset是一个迭代器
    
    classes = load_classes(parse_data_cfg(opt.data)['names'])                           # .data文件解析成dict并索引类别名的name文件地址
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]  # 配置颜色

    # registry hook
    # [i for i in model.named_modules()][-1][1].register_forward_hook(yolo_hook)

    # Run inference
    for path, img, im0s, vid_cap in dataset:    # im0s为原图(hwc)，img为缩放+padding之后的图(chw)

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:   # 查看数据维度是否为三维，等价于len(img.shape)
            img = img.unsqueeze(0)  # 加个第0维bs，但是detect实际没用

        pred, _ ,io_map = model(img)        # forward 
        for i, det in enumerate(non_max_suppression(pred, opt.conf_thres, opt.nms_thres)):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)

        ########################  施工区1: 单尺度   ##############################
        ac_maps = []
        for fi,fp in enumerate(io_map):
            for ai,base in enumerate(fp[0]):
                obj_feature  = base[...,4]

                cls_feature =  [base[...,c+5] for c in range(base[...,5:].shape[-1])]  
                scores = [f*obj_feature for f in cls_feature]

                # 格式:一个fp下的一个anchor返回一个dict,dict下的元素是若干个cam
                # GetCAM(obj_feature,tuple([img_h,img_w]),im0s,name='obj_{}_{}'.format(fi,ai))
                # GetBAM(base[...,:4],scores,tuple([img_h,img_w]),im0s,name='bam_{}_{}'.format(fi,ai))
                cam = GetCAM(scores, im0s, index=str(fi)+str(ai), agnostic=False)
                bam = GetBAM(base[...,:4], scores, im0s, index=str(fi)+str(ai), agnostic=False)

                ac_maps.append(bam)

        DrawMap(ac_maps,im0s)
        # display('/py/yolov3/cam')

        ########################  施工区2: 多尺度(改hook,使用不同的特征)   ##############################
        # anchor_index = [0,1,2]
        # fp_index = [0,1,2]

        # for fi in fp_index:
        #     for ai in anchor_index:
        #         base = io_map[fi]         # torch.Size([1, 3, 40, 52, 8]) # 8 = x y w h + c + classes 
        #         base = base[0,ai,...]     # 取出某个anchor的预测结果
        #         obj_feature  = base[...,4]

        #         cls_feature =  [base[...,c+5] for c in range(base[...,5:].shape[-1])]  
        #         scores = [f*obj_feature for f in cls_feature]

        #         img_h,img_w,_ = im0s.shape

                # DrawCAM(obj_feature,tuple([img_h,img_w]),im0s,name='obj_{}_{}'.format(fi,ai))




        ################################################################

        for i, det in enumerate(non_max_suppression(pred, opt.conf_thres, opt.nms_thres)):  # detections per image
            p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # s添加缩放后的图像尺度，如  320x416 
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():       # 取出最后一维类别并去重排序
                    n = (det[:, -1] == c).sum()     # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # s添加检测物体统计

                for *xyxy, conf, _, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])


            if view_img:
                cv2.imshow(p, im0)

            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)


    # results, maps = test.test(  opt.cfg,
    #                             opt.data,
    #                             batch_size=1,
    #                             img_size=opt.img_size,
    #                             model=model,
    #                             conf_thres= 0.1,  # 0.1 for speed
    #                             save_json=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-kmeans.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/voc.data', help='coco.data file path')
    # parser.add_argument('--weights', type=str, default='weights/test/backup4000.pt', help='path to weights file')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='cam/img/7.jpg', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='cam', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()

        
