import os
from collections import defaultdict

import torch.nn as nn

from utils.parse_config import *
from utils.utils import *

ONNX_EXPORT = False


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """ 
    hyperparams = module_defs.pop(0)                    #移除并返回列表首元素：net配置下超参数
    output_filters = [int(hyperparams['channels'])]     #图片通道数：3  （后面每层网络的输出通道都存放在这里）
    module_list = nn.ModuleList()                       #module_list用于存储每个block,每个block对应cfg文件中一个块，类似[convolutional]里面就对应一个卷积块
    for i, module_def in enumerate(module_defs):        #超参数net层被pop了，只剩网络层；返回模块索引i和模块信息module_def
        # 这里每个块用nn.sequential()创建为了一个module，一个module有多个层，是时序容器，`Modules` 会以他们传入的顺序被添加到容器中
        modules = nn.Sequential()
        #下面根据不同的层进行设计(一次遍历添加一个层)

        #卷积层（75个）
        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize']) #bn是一个开关，cfg中batch_normalize是1需要加bn层，为0不加（看过cfg了，全加）
            filters = int(module_def['filters'])    #卷积核数目
            kernel_size = int(module_def['size'])   #卷积核尺寸
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            #pad=0该层不用padding，pad=1需要填充，并计算值。
            #这个公式很巧妙，对于3*3的步长为2的降采样卷积核，可以使填充后计算得到1/2降采样效果，对于1*1卷积核，填充值计算为0不填充，所以作者实际把cfg的pad全置为1
            
            #开始添加卷积层：add_module(name,module) 将子模块加入当前的模块中，被添加的模块可以name来获取
            #nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],#通道列表的最后一个数（上一层网络的输出通道数）
                                                        out_channels=filters,          #输出通道数为该层卷积核
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),#卷积/池化步长
                                                        padding=pad,
                                                        bias=not bn))                   #加bn就不用偏置，可推导效果而言是一样的
            #bn和激活层也为层，此处嵌套在卷积层内因为其他层没有采用这些（linear层恒等映射不算）
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))#nn.BatchNorm2d见notebook注释
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1)) #negative_slope=0.01

        #池化层（实际上全卷积，没有池化层）
        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)

        #上采样层（2个）
        elif module_def['type'] == 'upsample':
            # upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')    
            # WARNING: deprecated(torch.nn的报错，作者自己在下面重新继承了下)
            upsample = Upsample(scale_factor=int(module_def['stride'])) #步长2，最近邻插值
            modules.add_module('upsample_%d' % i, upsample)

        #路由层
        elif module_def['type'] == 'route':#见notebook注释
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])#卷积核数目为两层的加和，参见注释
            modules.add_module('route_%d' % i, EmptyLayer())    #初始化为空,占位层

        #shortcut层（类似ResNet）
        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]#分析见notebook
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        #yolo层（三个特征层）
        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            #通过mask0-8决定每次使用哪三个anchor，传入YOLOLayer参数
            anchors = [float(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def['classes'])
            img_height = int(hyperparams['height']) #默认w=h，只提取一个就行
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height, anchor_idxs, cfg=hyperparams['cfg'])
            modules.add_module('yolo_%d' % i, yolo_layer)

        # Register module list and number of output filters
        module_list.append(modules)     #将模型放在这个list中
        output_filters.append(filters)  #将每一层的输出通道数记录（第一层为3输入图像通道），可以查看，代码实际没有调用

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""
    #为shortcut layer / route layer 提供占位, 具体功能不在此实现，在Darknet类的forward函数中有体现
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x

#torch.nn.upsample的上采样警报提醒，所以作者自定义了
class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode) #根据给定输出size或倍率上采样或下采样数据


class YOLOLayer(nn.Module):
    #分析以13*13特征图为例
    def __init__(self, anchors, nC, img_dim, anchor_idxs, cfg):
    #YOLOLayer实例化需要传入的参数：anchors, num_classes, img_height, anchor_idxs, cfg=hyperparams['cfg']
        super(YOLOLayer, self).__init__()

        anchors = [(a_w, a_h) for a_w, a_h in anchors]  # (pixels)#anchors不变，还是列表嵌套三个元组，但是将宽高提取赋值给了a_w和a_h
        nA = len(anchors)

        self.anchors = anchors
        self.nA = nA                # number of anchors (3)
        self.nC = nC                # number of classes (80)
        self.bbox_attrs = 5 + nC    #每个box的回归的属性值数目4+1+class
        self.img_dim = img_dim  # from hyperparams in cfg file, NOT from parser

        #降采样步长，同时也是对应原图每个grid cell的尺寸！后面会用
        if anchor_idxs[0] == (nA * 2):  # 6     #anchor_idxs[0]可能值为0,3,6
            stride = 32                         #检测13*13特征图，降采样步长32
        elif anchor_idxs[0] == nA:  # 3
            stride = 16                         #检测26*26特征图，降采样步长16
        else:
            stride = 8                          #检测52*52特征图，降采样步长8

        if cfg.endswith('yolov3-tiny.cfg'):
            stride *= 2                         #tiny-yolo3

        # Build anchor grids
        #这段比较复杂，中间结果见notebook
        nG = int(self.img_dim / stride)  # number grid points
        # grid_x、grid_y用于 定位 feature map的网格左上角坐标，对特征图编号
        self.grid_x = torch.arange(nG).repeat((nG, 1)).view((1, 1, nG, nG)).float() #shape:torch.Size([1, 1, 13, 13])
        self.grid_y = torch.arange(nG).repeat((nG, 1)).t().view((1, 1, nG, nG)).float()
        #将anchor也缩放到13*13（除32）尺寸上
        #（atten）实际上预测的xywh都是在13*13的特征图上完成的，所以anchor作为预测的模板要缩放。最后会*32还原原图上去
        self.anchor_wh = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])  # scale anchors
        self.anchor_w = self.anchor_wh[:, 0].view((1, nA, 1, 1))    #将anchow的w，h分离
        self.anchor_h = self.anchor_wh[:, 1].view((1, nA, 1, 1))
        self.weights = class_weights()  #统计了coco的80类gt出现的归一化比重作为权值

        self.loss_means = torch.ones(6) #6维行向量，用1初始化
        self.yolo_layer = anchor_idxs[0] / nA  # 2, 1, 0
        self.stride = stride
        self.nG = nG

        if ONNX_EXPORT:  # use fully populated and reshaped tensors
            self.anchor_w = self.anchor_w.repeat((1, 1, nG, nG)).view(1, -1, 1)
            self.anchor_h = self.anchor_h.repeat((1, 1, nG, nG)).view(1, -1, 1)
            self.grid_x = self.grid_x.repeat(1, nA, 1, 1).view(1, -1, 1)
            self.grid_y = self.grid_y.repeat(1, nA, 1, 1).view(1, -1, 1)
            self.grid_xy = torch.cat((self.grid_x, self.grid_y), 2)
            self.anchor_wh = torch.cat((self.anchor_w, self.anchor_h), 2) / nG

    def forward(self, p, targets=None, var=None):
    #p为传入的预测值，torch.Size([bs, 255, 13, 13])
        bs = 1 if ONNX_EXPORT else p.shape[0]  # batch size
        nG = self.nG if ONNX_EXPORT else p.shape[-1]  # number of grid points

        if p.is_cuda and not self.weights.is_cuda:  #将所有参数都放到cuda上
            self.grid_x, self.grid_y = self.grid_x.cuda(), self.grid_y.cuda()
            self.anchor_w, self.anchor_h = self.anchor_w.cuda(), self.anchor_h.cuda()
            self.weights, self.loss_means = self.weights.cuda(), self.loss_means.cuda()

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 80)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction
        #把255的列向量信息，加一个维度分开成3个box
        #p的shape为：torch.Size([1, 3, 13, 13, 85]) ，(bs, anchors, grid, grid, xywh+conf+class)
        #注意，当前YOLO层分配了3个anchor

        # Training
        if targets is not None:
            #坐标损失用均方差，分类损失用交叉熵
            MSELoss = nn.MSELoss()                      #实例化均方损失函数
            BCEWithLogitsLoss = nn.BCEWithLogitsLoss()  #交叉熵+sigmoid
            CrossEntropyLoss = nn.CrossEntropyLoss()    #多分类交叉熵

            # Get outputs
            #训练前向传入这里的x,y用sigmoid归一化（这里的xy是相对于13*13特征图的每个像素的，转化为真实xy还需加上该grid cell的位置并乘以32）
            x = torch.sigmoid(p[..., 0])  # Center x：torch.Size([1, 3, 13, 13]) ，1为bs
            y = torch.sigmoid(p[..., 1])  # Center y
            p_conf = p[..., 4]            # Conf： torch.Size([1, 3, 13, 13])
            p_cls = p[..., 5:]            # Class：torch.Size([1, 3, 13, 13, 80])

            # Width and height (yolo method)
            w = p[..., 2]                 # Width：torch.Size([1, 3, 13, 13])
            h = p[..., 3]                 # Height
            # width = torch.exp(w.data) * self.anchor_w
            # height = torch.exp(h.data) * self.anchor_h

            # Width and height (power method)
            # w = torch.sigmoid(p[..., 2])  # Width
            # h = torch.sigmoid(p[..., 3])  # Height
            # width = ((w.data * 2) ** 2) * self.anchor_w
            # height = ((h.data * 2) ** 2) * self.anchor_h

            tx, ty, tw, th, mask, tcls = build_targets(targets, self.anchor_wh, self.nA, self.nC, nG)

            tcls = tcls[mask]
            if x.is_cuda:
                tx, ty, tw, th, mask, tcls = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda(), mask.cuda(), tcls.cuda()

            # Compute losses
            nT = sum([len(x) for x in targets])  # number of targets
            nM = mask.sum().float()  # number of anchors (assigned to targets)
            nB = len(targets)  # batch size
            k = nM / nB
            if nM > 0:
                lx = k * MSELoss(x[mask], tx[mask])
                ly = k * MSELoss(y[mask], ty[mask])
                lw = k * MSELoss(w[mask], tw[mask])
                lh = k * MSELoss(h[mask], th[mask])

                lcls = (k / 4) * CrossEntropyLoss(p_cls[mask], torch.argmax(tcls, 1))
                # lcls = (k * 10) * BCEWithLogitsLoss(p_cls[mask], tcls.float())
            else:
                FT = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor
                lx, ly, lw, lh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0]), FT([0]), FT([0])

            lconf = (k * 64) * BCEWithLogitsLoss(p_conf, mask.float())

            # Sum loss components
            loss = lx + ly + lw + lh + lconf + lcls

            return loss, loss.item(), lx.item(), ly.item(), lw.item(), lh.item(), lconf.item(), lcls.item(), nT

        else:
            #不训练只是检测前向传播：
            if ONNX_EXPORT:
                # p = p.view(-1, 85)
                # xy = torch.sigmoid(p[:, 0:2]) + self.grid_xy[0]  # x, y
                # wh = torch.exp(p[:, 2:4]) * self.anchor_wh[0]  # width, height
                # p_conf = torch.sigmoid(p[:, 4:5])  # Conf
                # p_cls = F.softmax(p[:, 5:85], 1) * p_conf  # SSD-like conf
                # return torch.cat((xy / nG, wh, p_conf, p_cls), 1).t()

                p = p.view(1, -1, 85)
                xy = torch.sigmoid(p[..., 0:2]) + self.grid_xy  # x, y
                wh = torch.exp(p[..., 2:4]) * self.anchor_wh  # width, height
                p_conf = torch.sigmoid(p[..., 4:5])  # Conf
                p_cls = p[..., 5:85]
                # Broadcasting only supported on first dimension in CoreML. See onnx-coreml/_operators.py
                # p_cls = F.softmax(p_cls, 2) * p_conf  # SSD-like conf
                p_cls = torch.exp(p_cls).permute((2, 1, 0))
                p_cls = p_cls / p_cls.sum(0).unsqueeze(0) * p_conf.permute((2, 1, 0))  # F.softmax() equivalent
                p_cls = p_cls.permute(2, 1, 0)
                return torch.cat((xy / nG, wh, p_conf, p_cls), 2).squeeze().t()


            #（atten）经过前向传播得到的坐标参数，是416*416上的真实参数
            # p的shape为：torch.Size([1, 3, 13, 13, 85]) ，(bs, anchors, grid, grid, xywh+conf+class)
            # 前xywh处理:前面维度不管，只取最后维85信息的前四个，并且xy进行sigmoid归一化后加上左上定位坐标，wh取指数得到正的比例信息乘以anchor即可
            # xy得到的是在一个cell的相对比例信息，乘以降采样步长缩放到全图
            # wh为什么也要扩大32倍？因为实际上预测是在13*13的特征图进行的！所以预测出的坐标xywh都是基于13*13的，故均要缩放
            p[..., 0] = torch.sigmoid(p[..., 0]) + self.grid_x  # x
            p[..., 1] = torch.sigmoid(p[..., 1]) + self.grid_y  # y
            p[..., 2] = torch.exp(p[..., 2]) * self.anchor_w  # width
            p[..., 3] = torch.exp(p[..., 3]) * self.anchor_h  # height
            p[..., 4] = torch.sigmoid(p[..., 4])  # p_conf
            p[..., :4] *= self.stride

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return p.view(bs, -1, 5 + self.nC)


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg_path, img_size=416): #类传入两个参数初始化：cfg_path和img_size
        super(Darknet, self).__init__()         #继承父类的构造函数，继承自nn.Module可以直接不传递参数，如果是高阶类需要根据具体情况而定传递参数
        #注意：module_defs是一个嵌套的列表,外层列表内层嵌套的是字典
        self.module_defs = parse_model_cfg(cfg_path)    #parse_model_config()返回模型的参数信息，见notebook
        self.module_defs[0]['cfg'] = cfg_path
        self.module_defs[0]['height'] = img_size        #如第一个配置模块net的字典键值对height:416
        
        self.hyperparams, self.module_list = create_modules(self.module_defs)   #获取net网络配置参数和搭建模型结构（darknet-53）
        self.img_size = img_size
        self.loss_names = ['loss', 'x', 'y', 'w', 'h', 'conf', 'cls', 'nT']
        self.losses = []

    def forward(self, x, targets=None, var=0):
        self.losses = defaultdict(float)    #loss全局初始化为0.0
        is_training = targets is not None   #如果targets=None返回false，后面不训练
        layer_outputs = []                  #存储每层输出（包含yolo层）
        output = []                         #存储YOLO层的检测输出

        #见notebook(模型架构剖析)
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)   #直接计算前向输出
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')] #[-4],[-1,61],[-4],[-1,36]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)   #按第一维度（刨去batch维）融合
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]   #加和其上一层-1，当前层不是-1执行最后才append）和往上数第三层(-3)
            elif module_def['type'] == 'yolo':
                # Train phase: get loss
                if is_training:
                    #module[0]是YOLOLayer()，后面4个是向YOLO层传递进去的参数，根据自定义方式计算前向传播
                    x, *losses = module[0](x, targets, var)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)   #训练需要计算损失函数，检测直接前向计算结果就行
                output.append(x)    #存储yolo层前向传播结果
            layer_outputs.append(x) #所有层输出

        if is_training:
            self.losses['nT'] /= 3  # target category

        if ONNX_EXPORT:
            output = torch.cat(output, 1)  # merge the 3 layers 85 x (507, 2028, 8112) to 85 x 10647
            return output[5:85].t(), output[:4].t()  # ONNX scores, boxes

        return sum(output) if is_training else torch.cat(output, 1)


def load_darknet_weights(self, weights, cutoff=-1):
    #这里包括后面的self都是传入的模型
    # Parses and loads the weights stored in 'weights'
    # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
    weights_file = weights.split(os.sep)[-1]

    # Try to download weights if not available locally
    if not os.path.isfile(weights):
        try:
            os.system('wget https://pjreddie.com/media/files/' + weights_file + ' -O ' + weights)
        except IOError:
            print(weights + ' not found')

    # Establish cutoffs
    if weights_file == 'darknet53.conv.74':
        cutoff = 75
    elif weights_file == 'yolov3-tiny.conv.15':
        cutoff = 16

    # Open the weights file
    fp = open(weights, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    self.header_info = header

    self.seen = header[3]  # number of images seen during training
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w


"""
    @:param path    - path of the new weights file
    @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
"""


def save_weights(self, path, cutoff=-1):
    fp = open(path, 'wb')
    self.header_info[3] = self.seen  # number of images seen during training
    self.header_info.tofile(fp)

    # Iterate through layers
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            # If batch norm, load bn first
            if module_def['batch_normalize']:
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
            # Load conv bias
            else:
                conv_layer.bias.data.cpu().numpy().tofile(fp)
            # Load conv weights
            conv_layer.weight.data.cpu().numpy().tofile(fp)

    fp.close()
