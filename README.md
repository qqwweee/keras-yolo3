# keras-yolo3

[![Build Status](https://travis-ci.org/Borda/keras-yolo3.svg?branch=master)](https://travis-ci.org/Borda/keras-yolo3)
[![Build status](https://ci.appveyor.com/api/projects/status/24m00vife2wae7k0/branch/master?svg=true)](https://ci.appveyor.com/project/Borda/keras-yolo3/branch/master)
[![CircleCI](https://circleci.com/gh/Borda/keras-yolo3.svg?style=svg)](https://circleci.com/gh/Borda/keras-yolo3)
[![codecov](https://codecov.io/gh/Borda/keras-yolo3/branch/master/graph/badge.svg)](https://codecov.io/gh/Borda/keras-yolo3)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/e03dbbb0f0fd48baa70f637456f1fe36)](https://www.codacy.com/project/Borda/keras-yolo3/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Borda/keras-yolo3&amp;utm_campaign=Badge_Grade_Dashboard)
[![CodeFactor](https://www.codefactor.io/repository/github/borda/keras-yolo3/badge)](https://www.codefactor.io/repository/github/borda/keras-yolo3)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A [Keras](https://keras.io/) implementation of YOLOv3 ([Tensorflow backend](https://www.tensorflow.org/)) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).

---

## Quick Start

For more model and configuration please see  [YOLO website](http://pjreddie.com/darknet/yolo/) and [darknet](https://github.com/pjreddie/darknet/tree/master/cfg) repository.

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
    ```bash
    wget -O ./model_data/yolo3.weights  \
       https://pjreddie.com/media/files/yolov3.weights  \
       --progress=bar:force:noscroll
    ```
    alternatively you can download light version `yolov3-tiny.weights` 
2. Convert the Darknet YOLO model to a Keras model.
    ```bash
    python3 scripts/convert_weights.py \
        --config_path ./model_data/yolo.cfg \
        --weights_path ./model_data/yolo.weights \
        --output_path ./model_data/yolo.h5
    ```
3. Run YOLO detection.
    ```bash
    python3 scripts/predict.py \
       --path_weights ./model_data/yolo.h5 \
       --path_anchors ./model_data/yolo_anchors.csv \
       --path_classes ./model_data/coco_classes.txt \
       --path_output ./results \
       --path_image ./model_data/bike-car-dog.jpg \
       --path_video person.mp4
    ```
    For Full YOLOv3, just do in a similar way, just specify model path and anchor path with `--path_weights <model_file>` and `--path_anchors <anchor_file>`.
4. MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the Keras [multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

---

## Training

For training you can use [VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/), [COCO datset](cocodataset.org) or your own...

1. Generate your own annotation file and class names file.  
    * One row for one image;  
    * Row format: `image_file_path box1 box2 ... boxN`;  
    * Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    * For VOC dataset.  
    Run one of following scrips for dataset conversion 
    * `scripts/annotation_voc.py` 
    * `scripts/annotation_coco.py` 
    * `scripts/annotation_csv.py`  
    Here is an example:
    ```text
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```
2. Make sure you have run `python scripts/convert_weights.py <...>`.
    The file `model_data/yolo_weights.h5` is used to load pre-trained weights.
3. Modify train.py and start training.  `python train.py`.
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using `yolo_interactive.py`.
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

If you want to use original pre-trained weights for YOLOv3:  
  1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
  2. rename it as `darknet53.weights`  
  3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
  4. use `model_data/darknet53_weights.h5` in `train.py`

---

## Some issues to know

1. The test environment is Python 3.5.2 ; Keras 2.1.5 ; tensorflow 1.6.0
2. Default anchors are used. If you use your own anchors, probably some changes are needed.
3. The inference result is not totally the same as Darknet but the difference is small.
4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.
5. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.
6. The training strategy is for reference only. Adjust it according to your dataset and your goal. and add further strategy if needed.
7. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See this [post](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.
