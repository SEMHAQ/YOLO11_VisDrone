import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v10/yolov10n.yaml') # YOLO11
    model.load('yolov10n.pt') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                batch=32,
                close_mosaic=10, # 最后多少个epoch关闭mosaic数据增强，设置0代表全程开启mosaic训练
                workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                # device='0,1', # 指定显卡和多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt,不懂就在百度云.txt找断点续训的视频
                # amp=False, # close amp | loss出现nan可以关闭amp
                # fraction=0.2,
                project='runs/train-v10',
                name='v10',
                )