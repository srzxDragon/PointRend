# 基于Detectron2的PointRend模型训练（语义分割）
1. 下载并安装Detectron2，参考Detectron2[官方安装文档](https://detectron2.readthedocs.io/tutorials/install.html)
2. 进入PointRend工作目录，并修改相关文件
```bash
cd /path/to/detectron2/projects/PointRend
```
* 修改文件train_net.py
1. 定义自己数据的读取方式  
这里要在train_net.py里加一个函数，函数名可以随便取
以我的读取方式为例  
我的数据图片放在了/data/Dataset/VOC/JPEGImages文件夹下，这个文件夹下全部是以.jpg结尾的图片  
我的标签数据放在了/data/Dataset/VOC/Annotations文件下，这个文件夹下全部是以.png结尾的标签，因为任务是语义分割，所以标签是灰度图  
这个函数的参数是文件名存放的路径，image_name_file是一个文本文件，里面存放了图片的名称，存放方式如下,只存放了文件的基本名字，没有后缀。  
我定义了两个这样的文件，分别命名为train.txt和val.txt用作训练和测试(验证)，之后要注册这两个数据集。
你也可以定义你自己的文件读取方式函数，最后函数返回的是一个List[dict]  
其中dict字典要的key要包含file_name，height，width，image_id，sem_seg_file_name，分别是图片绝对路径，图片高度，图片宽度，图片id，相应的标签。
```
2007_000033
2007_000042
2007_000061
2007_000123
...
```
```python
from detectron2.data import DatasetCatalog

def VOC_function(image_name_file):
    JPEG_dir = "/data/Dataset/VOC/JPEGImages"
    Anno_dir = "/data/Dataset/VOC/Annotations"

    dataset_dicts = []
    with open(image_name_file, "r") as imgs_name:
        image_name_list = imgs_name.readlines()
        for idx, image_name in enumerate(image_name_list):
            record = {}
            image_path = os.path.join(JPEG_dir, image_name.strip() + ".jpg")
            height, width = cv2.imread(image_path).shape[:2]
            
            record["file_name"] = image_path
            record["height"] = height
            record["width"] = width
            record["image_id"] = idx


            anno_path = os.path.join(Anno_dir, image_name.strip() + ".png")
            record["sem_seg_file_name"] = anno_path

            dataset_dicts.append(record)

    return dataset_dicts
```

2. 注册自己的数据集  
我分别注册了我的训练集和验证集  
以训练集为例  
文件名存放位置为/data/zhengxiaolong/Dataset/VOC/train.txt  
注册后数据集名字为VOC_train  
注册时定义了一些metadata例如stuff_classes，是类别名  
```python
for d in ["train", "val"]:
    src_dir = "/data/zhengxiaolong/Dataset/VOC"
    image_name_file = os.path.join(src_dir, d + ".txt")
    DatasetCatalog.register("VOC_" + d, lambda image = image_name_file: SJTD_function(image_name_file))
    MetadataCatalog.get("VOC_" + d).set(stuff_classes = ['_background_', 'ground', 'tree', 'building', 'conductor', 'tower'])
    MetadataCatalog.get("VOC_" + d).set(evaluator_type = "sem_seg")
    MetadataCatalog.get("VOC_" + d).set(ignore_label = 255)

SJTD_metadata = MetadataCatalog.get("VOC_train")
SJTD_metadata = MetadataCatalog.get("VOC_val")
```
3. 修改配置  
我们以configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml为基础修该配置文件
```yaml
_BASE_: Base-PointRend-Semantic-FPN.yaml
MODEL:
  # WEIGHTS: detectron2://ImageNetPretrained/MSRA/R-101.pkl
  RESNETS:
    DEPTH: 50         # 这里我改成了resnet50
  SEM_SEG_HEAD:
    NUM_CLASSES: 6    # 类别数
  POINT_HEAD:
    NUM_CLASSES: 6    # 类别数
    TRAIN_NUM_POINTS: 2048
    SUBDIVISION_NUM_POINTS: 8192
DATASETS:
  TRAIN: ("cityscapes_fine_sem_seg_train",)
  TEST: ("cityscapes_fine_sem_seg_val",)
SOLVER:
  BASE_LR: 0.01
  STEPS: (40000, 55000)
  MAX_ITER: 65000
  IMS_PER_BATCH: 32
INPUT:
  MIN_SIZE_TRAIN: (512, 768, 1024, 1280, 1536, 1792, 2048)
  MIN_SIZE_TRAIN_SAMPLING: "choice"
  MIN_SIZE_TEST: 1024
  MAX_SIZE_TRAIN: 4096
  MAX_SIZE_TEST: 2048
  CROP:
    ENABLED: True
    TYPE: "absolute"
    SIZE: (512, 1024)
    SINGLE_CATEGORY_MAX_AREA: 0.75
  COLOR_AUG_SSD: True
DATALOADER:
  NUM_WORKERS: 10
TEST:
  EVAL_PERIOD: 6777   #间隔几个循环进行一个评估
```  
此外我们还可以在train_net.py中修改配置,修改setup()函数
```python
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.DATASETS.TRAIN = ("VOC_train",)   # 之前注册的训练集
    cfg.DATASETS.TEST = ("VOC_val",)      # 之前注册的测试集
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 6777 * 50      # 循环次数，我定义成训练集大小*50次
    cfg.SOLVER.STEPS = [20, 40]
    
    cfg.SOLVER.CHECKPOINT_PERIOD = 6777 * 2   # 在每几次循环结束后保存模型
    cfg.TEST.EVAL_PERIOD = 6777          # 在每几次个循环评估模型
    
    cfg.freeze()

    default_setup(cfg, args)
    return cfg
```
4. 训练时保存效果最好的模型(可选)  
如果想在训练时保存评估效果最好多的模型，可以在train_net中加如下代码
```py
from detectron2.engine import HookBase
import logging
class BestCheckpointer(HookBase):
    def __init__(self):
        super().__init__()


    def before_train(self):
        self.best_metric = 0.0
        self.logger = logging.getLogger("detectron2.trainer")
        self.logger.info("######## Running best check pointer")
    

    def after_step(self):
        # self.logger.info("========")
        # No way to use **kwargs

        ##ONly do this analys when trainer.iter is divisle by checkpoint_epochs
        curr_val = self.trainer.storage.latest().get('sem_seg/mIoU', 0)
        
        import math
        if type(curr_val) != int:
            curr_val = curr_val[0]
            if math.isnan(curr_val):
                curr_val = 0

        try:
            _ = self.trainer.storage.history('max_sem_seg/mIoU')
        except:
            self.trainer.storage.put_scalar('max_sem_seg/mIoU', curr_val)

        max_val = self.trainer.storage.history('max_sem_seg/mIoU')._data[-1][0]

        #print(curr_val, max_val)
        if curr_val > max_val:
            print("\n%s > %s要存!!\n"%(curr_val, max_val))
            self.trainer.storage.put_scalar('max_sem_seg/mIoU', curr_val)
            self.trainer.checkpointer.save("model_best_{}".format(max_val))
            #self.step(self.trainer.iter)


# 在Trainer中增加一个函数
class Trainer(DefaultTrainer):
    ....
    ....
    ....
    def build_hooks(self):
        ret = super().build_hooks()
        ret.append(BestCheckpointer())
        return ret
```
5. 训练
```bash
CUDA_VISIBLE_DEVICES=6 python train_net.py \
    --config-file configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml \
    --num-gpus 1
```
