# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import time
import logging
import colorsys

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras.utils import multi_gpu_model

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image, update_path
from yolo3.visual import draw_bounding_box

PREDICT_FIELDS = ('class', 'label', 'score', 'xmin', 'ymin', 'xmax', 'ymax')


class YOLO(object):

    _DEFAULT_PARAMS = {
        "model_path": 'model_data/tiny-yolo.h5',
        "anchors_path": 'model_data/tiny-yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._DEFAULT_PARAMS:
            return cls._DEFAULT_PARAMS[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, model_path='model_data/tiny-yolo.h5',
                 anchors_path='model_data/tiny-yolo_anchors.txt',
                 classes_path='model_data/coco_classes.txt',
                 score=0.3, iou=0.45,
                 model_image_size=(416, 416),
                 gpu_num=1, **kwargs):
        self.__dict__.update(kwargs)  # and update with user overrides
        self.model_path = update_path(model_path)
        self.anchors_path = update_path(anchors_path)
        self.classes_path = update_path(classes_path)
        self.score = score
        self.iou = iou
        self.model_image_size = model_image_size
        self.gpu_num = gpu_num
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        # classes_path = update_path(self.classes_path)
        logging.debug('loading classes from "%s"', self.classes_path)
        with open(self.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        # anchors_path = update_path(self.anchors_path)
        logging.debug('loading anchors from "%s"', self.anchors_path)
        with open(self.anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        # model_path = update_path(self.model_path)
        logging.debug('loading model from "%s"', self.model_path)
        assert self.model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = (num_anchors == 6)  # default setting
        try:
            self.yolo_model = load_model(self.model_path, compile=False)
        except Exception:
            if is_tiny_version:
                self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes)
            else:
                self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            out_shape = self.yolo_model.layers[-1].output_shape[-1]
            ration_anchors = num_anchors / len(self.yolo_model.output) * (num_classes + 5)
            assert out_shape == ration_anchors, \
                'Mismatch between model and given anchor and class sizes'

        logging.info('loaded model, anchors, and classes from %s', self.model_path)

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = time.time()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        logging.debug('image shape: %s', repr(image_data.shape))
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        end = time.time()

        logging.debug('Found %i boxes', len(out_boxes))

        thickness = (image.size[0] + image.size[1]) // 300

        predicts = []
        for i, c in reversed(list(enumerate(out_classes))):
            draw_bounding_box(image, self.class_names[c], out_boxes[i], out_scores[i],
                              self.colors[c], thickness)
            pred = dict(zip(
                PREDICT_FIELDS,
                (int(c), self.class_names[c], float(out_scores[i]), *[int(x) for x in out_boxes[i]])
            ))
            predicts.append(pred)

        logging.debug('elapsed time: %f sec.', (end - start))
        return image, predicts

    def close_session(self):
        self.sess.close()
