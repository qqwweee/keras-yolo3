# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from keras.utils import multi_gpu_model
K.clear_session()


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.88,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 2,
    }
    # _defaults = {
    #     "model_path": 'model_data/yolo_zhou.h5',
    #     "anchors_path": 'model_data/yolo_anchors.txt',
    #     "classes_path": 'model_data/zhou_classes.txt',
    #     "score" : 0.5,
    #     "iou" : 0.45,
    #     "model_image_size" : (416, 416),
    #     "gpu_num" : 1,
    # }
    # _defaults = {
    #     "model_path": 'model_data/smart_court_weights_v1.h5',
    #     "anchors_path": 'model_data/yolo_anchors_court.txt',
    #     "classes_path": 'model_data/voc_classes_court.txt',
    #     "score": 0.3,
    #     "iou": 0.45,
    #     "model_image_size": (800, 800),
    #     "gpu_num": 0,
    # }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        import keras.backend.tensorflow_backend as KTF
        KTF.set_session(self.get_session())
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def get_session(self,gpu_fraction=0.3):
        '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

        num_threads = os.environ.get('OMP_NUM_THREADS')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

        if num_threads:
            return tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
        else:
            return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(
            '.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(
                self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(float(x) / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(
            self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(
                self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(
            self.yolo_model.output,
            self.anchors,
            len(self.class_names),
            self.input_image_shape,
            score_threshold=self.score,
            iou_threshold=self.iou)
        return boxes, scores, classes

    def letterbox_image(self, image, size):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size
        # print(iw, ih)
        w, h = size
        # print("size:",size)
        scale = min(float(w) / iw, float(h) / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        # print(nw,nh)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image

    def detect_image(self, image):
        # import pdb;pdb.set_trace()
        if self.model_image_size != (None, None):
            assert self.model_image_size[
                0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[
                1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = self.letterbox_image(
                image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = self.letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        label = 0;left = 0;right = 0; top = 0; bottom = 0
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i],
                               outline=self.colors[c])
            # print("color:",c,self.colors[c])
            draw.rectangle(
                [tuple(text_origin),
                 tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        return image, (label, (left, top), (right, bottom))

    def close_session(self):
        self.sess.close()
