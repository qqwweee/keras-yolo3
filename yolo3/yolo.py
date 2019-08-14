# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import os
import time
import logging
import colorsys

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras.utils import multi_gpu_model

from yolo3.model import yolo_eval, yolo_body, yolo_body_tiny
from yolo3.utils import letterbox_image, update_path, get_anchors, get_class_names
from yolo3.visual import draw_bounding_box

PREDICT_FIELDS = ('class', 'label', 'score', 'xmin', 'ymin', 'xmax', 'ymax')


class YOLO(object):
    """YOLO detector with tiny alternative

    Example
    -------
    >>> # prepare EMPTY model since download and convert existing is a bit complicated
    >>> anchors = get_anchors(YOLO.get_defaults('anchors_path'))
    >>> classes = get_class_names(YOLO.get_defaults('classes_path'))
    >>> yolo_empty = yolo_body_tiny(Input(shape=(None, None, 3)), len(anchors) // 2, len(classes))
    >>> path_model = os.path.join(update_path('model_data'), 'yolo_empty.h5')
    >>> yolo_empty.save(path_model)
    >>> # use the empty one, so no reasonable detections are expected
    >>> from yolo3.utils import image_open
    >>> yolo = YOLO(weights_path=path_model,
    ...             anchors_path=YOLO.get_defaults('anchors_path'),
    ...             classes_path=YOLO.get_defaults('classes_path'),
    ...             model_image_size=YOLO.get_defaults('model_image_size'))
    >>> img = image_open(os.path.join(update_path('model_data'), 'bike-car-dog.jpg'))
    >>> yolo.detect_image(img)  # doctest: +ELLIPSIS
    (<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=520x518 at ...>, [...])
    """

    _DEFAULT_PARAMS = {
        "weights_path": os.path.join(update_path('model_data'), 'tiny-yolo.h5'),
        "anchors_path": os.path.join(update_path('model_data'), 'tiny-yolo_anchors.csv'),
        "classes_path": os.path.join(update_path('model_data'), 'coco_classes.txt'),
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, name):
        if name not in cls._DEFAULT_PARAMS:
            logging.warning('Unrecognized attribute name "%s"', name)
        return cls._DEFAULT_PARAMS.get(name)

    def __init__(self, weights_path, anchors_path, classes_path, model_image_size,
                 score=0.3, iou=0.45, gpu_num=1, **kwargs):
        """

        :param str weights_path: path to loaded model weights, e.g. 'model_data/tiny-yolo.h5'
        :param str anchors_path: path to loaded model anchors, e.g. 'model_data/tiny-yolo_anchors.csv'
        :param str classes_path: path to loaded trained classes, e.g. 'model_data/coco_classes.txt'
        :param float score: confidence score
        :param float iou:
        :param tuple(int,int) model_image_size: e.g. for tiny (416, 416)
        :param int gpu_num:
        :param kwargs:
        """
        self.__dict__.update(kwargs)  # and update with user overrides
        self.weights_path = update_path(weights_path)
        self.anchors_path = update_path(anchors_path)
        self.classes_path = update_path(classes_path)
        self.score = score
        self.iou = iou
        self.model_image_size = model_image_size
        self.gpu_num = gpu_num
        self.class_names = get_class_names(self.classes_path)
        self.anchors = get_anchors(self.anchors_path)
        self._open_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _open_session(self):
        if K.backend() == 'tensorflow':
            import tensorflow as tf
            from keras.backend.tensorflow_backend import set_session

            config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
            config.gpu_options.force_gpu_compatible = True
            # config.gpu_options.per_process_gpu_memory_fraction = 0.5
            # Don't pre-allocate memory; allocate as-needed
            config.gpu_options.allow_growth = True

            sess = tf.Session(config=config)
            set_session(sess)

        self.sess = K.get_session()

    def generate(self):
        # weights_path = update_path(self.weights_path)
        logging.debug('loading model from "%s"', self.weights_path)
        assert self.weights_path.endswith('.h5'), \
            'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = (num_anchors == 6)  # default setting
        try:
            self.yolo_model = load_model(self.weights_path, compile=False)
        except Exception:
            logging.exception('Loading weights from "%s"', self.weights_path)
            if is_tiny_version:
                self.yolo_model = yolo_body_tiny(Input(shape=(None, None, 3)),
                                                 num_anchors // 2, num_classes)
            else:
                self.yolo_model = yolo_body(Input(shape=(None, None, 3)),
                                            num_anchors // 3, num_classes)
            # make sure model, anchors and classes match
            self.yolo_model.load_weights(self.weights_path, by_name=True, skip_mismatch=True)
        else:
            out_shape = self.yolo_model.layers[-1].output_shape[-1]
            ration_anchors = num_anchors / len(self.yolo_model.output) * (num_classes + 5)
            assert out_shape == ration_anchors, \
                'Mismatch between model and given anchor %r and class %r sizes' \
                % (ration_anchors, out_shape)

        logging.info('loaded model, anchors, and classes from %s',
                     self.weights_path)

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        _fn_colorr = lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255))
        self.colors = list(map(_fn_colorr, self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        # Shuffle colors to decorrelate adjacent classes.
        np.random.shuffle(self.colors)
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output,
                                           self.anchors,
                                           len(self.class_names),
                                           self.input_image_shape,
                                           score_threshold=self.score,
                                           iou_threshold=self.iou)
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
        if image_data.max() > 1.5:
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
        logging.debug('Found %i boxes in %f sec.', len(out_boxes), (end - start))

        thickness = (image.size[0] + image.size[1]) // 300

        predicts = []
        for i, c in reversed(list(enumerate(out_classes))):
            draw_bounding_box(image, self.class_names[c], out_boxes[i],
                              out_scores[i], self.colors[c], thickness)
            pred = dict(zip(
                PREDICT_FIELDS,
                (int(c), self.class_names[c], float(out_scores[i]),
                 *[int(x) for x in out_boxes[i]])
            ))
            predicts.append(pred)
        return image, predicts

    def _close_session(self):
        self.sess.close()

    def __del__(self):
        self._close_session()
