"""YOLOX experiment for handwritten music symbols on staff-strip tiles.

Train:  python tools/train.py -f /path/to/yolox_handwritten.py -d 1 -b 32 --fp16 -o
Eval:   python tools/eval.py  -f /path/to/yolox_handwritten.py -c best_ckpt.pth -b 16 -d 1 --conf 0.001
Set MUSIC_DATA_DIR to the folder holding train2017/, val2017/ and annotations/.
Set MUSIC_MODEL=nano for the smaller model (default: tiny).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from classes import CLASSES, TILE_H, TILE_W  # noqa: E402

from yolox.exp import Exp as BaseExp  # noqa: E402


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        size = os.environ.get("MUSIC_MODEL", "tiny")
        self.depth = 0.33
        self.width = 0.25 if size == "nano" else 0.375
        self.num_classes = len(CLASSES)
        self.data_dir = os.environ.get("MUSIC_DATA_DIR", "data/coco")
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_test2017.json"
        self.input_size = (TILE_H, TILE_W)          # (height, width), multiples of 32
        self.test_size = (TILE_H, TILE_W)
        self.random_size = (8, 12)                  # multi-scale heights 256-384 keep the width ratio
        self.multiscale_range = 2
        # music is position-sensitive: no flips, almost no rotation, mild scaling
        self.flip_prob = 0.0
        self.hsv_prob = 0.0
        self.degrees = 2.0
        self.translate = 0.05
        self.shear = 1.0
        self.mosaic_prob = 0.5
        self.mosaic_scale = (0.8, 1.25)
        self.enable_mixup = False
        self.mixup_prob = 0.0
        self.max_epoch = int(os.environ.get("MUSIC_EPOCHS", 120))
        self.no_aug_epochs = 15
        self.warmup_epochs = 3
        self.eval_interval = 10
        self.print_interval = 50
        self.data_num_workers = 4
        self.test_conf = 0.25
        self.nmsthre = 0.5
        self.exp_name = f"music_{size}"
        if size == "nano":
            self.enable_mixup = False

    def get_model(self, sublinear=False):
        if self.width <= 0.25:  # nano uses depthwise convolutions
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            import torch.nn as nn
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act, depthwise=True)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act, depthwise=True)
            self.model = YOLOX(backbone, head)

            def init_yolo(M):
                for m in M.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eps = 1e-3
                        m.momentum = 0.03
            self.model.apply(init_yolo)
            self.model.head.initialize_biases(1e-2)
            self.model.train()
            return self.model
        return super().get_model()

    # a staff strip holds many more symbols than a COCO photo: raise the label caps (base uses 50/120)
    def get_dataset(self, cache=False, cache_type="ram"):
        from yolox.data import COCODataset, TrainTransform
        return COCODataset(data_dir=self.data_dir, json_file=self.train_ann, img_size=self.input_size,
                           preproc=TrainTransform(max_labels=400, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
                           cache=cache, cache_type=cache_type)

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=None):
        from yolox.data import TrainTransform
        import yolox.data
        orig = yolox.data.TrainTransform

        class Wide(TrainTransform):
            def __init__(self, max_labels=400, **kw):
                super().__init__(max_labels=max(max_labels, 400), **kw)
        yolox.data.TrainTransform = Wide
        try:
            return super().get_data_loader(batch_size, is_distributed, no_aug, cache_img)
        finally:
            yolox.data.TrainTransform = orig
