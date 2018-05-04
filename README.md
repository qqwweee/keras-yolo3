# keras-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).

---

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo.py   OR   python yolo_video.py
```

---

## Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: image_file_path box1 box2 ... boxN;  
    Box format: x_min,y_min,x_max,y_max,class_id (no space).  
    For VOC dataset, try `python voc_annotation.py`

2. Make sure you have run `python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5`  
    A file model_data/yolo_weights.h5 will be generated when you run train.py for the first time.  
    The file is used to load pretrained weights.

3. Modify train.py and start training.  
    `python train.py`  
    You will get the trained model model_data/my_yolo.h5.
