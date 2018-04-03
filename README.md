# keras-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).

Training is not supported.

---

## Quick Start

- Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
- Convert the Darknet YOLO model to a Keras model.
- Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo.py
```
