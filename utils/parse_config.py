#(re)
#解析yolov3.cfg配置，返回结果为所有的网络参数信息，见notebook的演示
def parse_model_cfg(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')     #获取所有行内容，存入列表，一行一个索引
    lines = [x for x in lines if x and not x.startswith('#')]   #提取开头不是‘#’的行，重新存入lines列表
    lines = [x.rstrip().lstrip() for x in lines]  #截掉每个切片的左右空格
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block[]代表是一个不同的模块
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs
# 定义了6种不同type
# 'net': 相当于超参数,网络全局配置的相关参数
# {'convolutional', 'net', 'route', 'shortcut', 'upsample', 'yolo'}




def parse_data_cfg(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options
    
# {'gpus': '0,1,2,3',
#  'num_workers': '10',
#  'classes': '80',
#  'train': '../coco/trainvalno5k.txt',
#  'valid': '../coco/5k.txt',
#  'names': 'data/coco.names',
#  'backup': 'backup/',
#  'eval': 'coco'}
