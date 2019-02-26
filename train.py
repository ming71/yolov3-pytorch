import argparse
import time

import test  # Import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *


def train(
        cfg,
        data_cfg,
        img_size=416,
        resume=False,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        weights='weights',
        multi_scale=False,      #默认关闭多尺度训练
        freeze_backbone=True,
        var=0,
):
    device = torch_utils.select_device()

    if multi_scale:  # pass maximum multi_scale size
        img_size = 608
    else:
        #（re）在数据输入维度变化不大的情况下，加这一句可以提高计算速度，如果变化频繁反而拖累训练
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    latest = os.path.join(weights, 'latest.pt')
    best = os.path.join(weights, 'best.pt')

    # Configure run
    train_path = parse_data_cfg(data_cfg)['train']

    # Initialize model
    model = Darknet(cfg, img_size)

    # Get dataloader数据增强
    dataloader = LoadImagesAndLabels(train_path, batch_size, img_size, multi_scale=multi_scale, augment=True)

    lr0 = 0.001
    if resume:  #断点继续训练
        #checkpoint的信息包含：dict_keys(['epoch', 'best_loss', 'model', 'optimizer'])
        checkpoint = torch.load(latest, map_location='cpu')

        # Load weights to resume from
        model.load_state_dict(checkpoint['model'])  #断点数据初始化新建的模型

        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        model.to(device).train()

        # # Transfer learning (train only YOLO layers)
        # for i, (name, p) in enumerate(model.named_parameters()):
        #     if p.shape[0] != 650:  # not YOLO layer
        #         p.requires_grad = False

        # Set optimizer
        #采用的SGD详情见notebook，此函数可(re)
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            #加载优化器参数，至于内部结构见notobook
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']

        del checkpoint  # current, saved

    #从头训练
    else:
        start_epoch = 0
        best_loss = float('inf')    #正无穷

        # Initialize model with darknet53 weights (optional)
        load_darknet_weights(model, os.path.join(weights, 'darknet53.conv.74'))

        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        model.to(device).train()

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)

    # Set scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[54, 61], gamma=0.1)

    model_info(model)   #（re）该函数打印出网络结构,utils.py注释，&notebook(模型结构打印)
    t0 = time.time()
    
    #对所有样本进行每轮前向后向传播
    for epoch in range(epochs):
        epoch += start_epoch

        print(('%8s%12s' + '%10s' * 9) % (
            'Epoch', 'Batch', 'x', 'y', 'w', 'h', 'conf', 'cls', 'total', 'nTargets', 'time'))

        # Update scheduler (automatic)
        # scheduler.step()

        # Update scheduler (manual)  at 0, 54, 61 epochs to 1e-3, 1e-4, 1e-5
        if epoch > 50:  #50轮后降低学习速率
            lr = lr0 / 10
        else:
            lr = lr0
        for g in optimizer.param_groups:
            g['lr'] = lr    #（atten）设置学习率的方法，改optimizer.param_groups的key为'lr'的value

        # Freeze darknet53.conv.74 for first epoch
        #第一轮冻结预训练模型前面74层残差卷积（默认开启）
        if freeze_backbone:
            if epoch == 0:
                for i, (name, p) in enumerate(model.named_parameters()):
                    if int(name.split('.')[1]) < 75:  # if layer < 75
                        p.requires_grad = False #关闭求导就不更新参数
                #注意，layer编号的一个残差模块算一层，BN也附属于卷积层而不是单独算层
                #75层之前的冻结，也就是固定最后一个残差模块之前的层
            elif epoch == 1:
                for i, (name, p) in enumerate(model.named_parameters()):
                    if int(name.split('.')[1]) < 75:  # if layer < 75
                        p.requires_grad = True  #从第一轮开始设置回True，后面每轮进行正常的学习

        ui = -1
        rloss = defaultdict(float)  # running loss初始化0
        optimizer.zero_grad()       #清空优化器的梯度
        #dataloader包含两个信息：
        #全部图片的像素矩阵如torch.Size([30, 3, 416, 416])；图片gt张量（如30个tensor，每个tensor包含n个obj，每个obj5个元素：cxywh）
        for i, (imgs, targets) in enumerate(dataloader):
            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue

            # SGD burn-in
            #数据量不大的时候第一轮学习率设置
            if (epoch == 0) & (i <= 1000):
                lr = lr0 * (i / 1000) ** 4
                for g in optimizer.param_groups:
                    g['lr'] = lr

            # Compute loss, compute gradient, update parameters 前向传播隐式调用forward，传入forward的参数即可
            loss = model(imgs.to(device), targets, var=var)
            loss.backward()

            # accumulate gradient for x batches before optimizing
            if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            ui += 1
            for key, val in model.losses.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)

            s = ('%8s%12s' + '%10.3g' * 9) % (
                '%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, len(dataloader) - 1), rloss['x'],
                rloss['y'], rloss['w'], rloss['h'], rloss['conf'], rloss['cls'],
                rloss['loss'], model.losses['nT'], time.time() - t0)
            t0 = time.time()
            print(s)

        # Update best loss
        loss_per_target = rloss['loss'] / rloss['nT']
        if loss_per_target < best_loss:
            best_loss = loss_per_target

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_loss': best_loss,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, latest)

        # Save best checkpoint
        if best_loss == loss_per_target:
            os.system('cp ' + latest + ' ' + best)

        # Save backup weights every 5 epochs
        if (epoch > 0) & (epoch % 5 == 0):
            os.system('cp ' + latest + ' ' + os.path.join(weights, 'backup{}.pt'.format(epoch)))

        # Calculate mAP
        with torch.no_grad():
            mAP, R, P = test.test(cfg, data_cfg, weights=latest, batch_size=batch_size, img_size=img_size)

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 3 % (mAP, P, R) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #'--multi-scale'接受参数输入后，执行动作"store_true"改为 True
    #断点继续训练，同样'--resume'接受参数输入，执行动作"store_true"改为 True，在latest.pt基础上继续训练
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='cfg/coco.data', help='coco.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='pixels')
    parser.add_argument('--weights', type=str, default='weights', help='path to store weights')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--freeze', action='store_true', help='freeze darknet53.conv.74 layers for first epoch')
    parser.add_argument('--var', type=float, default=0, help='test variable')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    init_seeds()

    torch.cuda.empty_cache()
    
    train(
        opt.cfg,
        opt.data_cfg,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulated_batches=opt.accumulated_batches,
        weights=opt.weights,
        multi_scale=opt.multi_scale,
        freeze_backbone=opt.freeze,
        var=opt.var,
    )
