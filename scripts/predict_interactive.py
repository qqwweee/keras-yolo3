"""
Detection in interactive mode

python predict_interactive.py \
        --path_weights ./model_data/yolo3-tiny.h5 \
        --path_anchors ./model_data/tiny-yolo_anchors.csv \
        --path_classes ./model_data/coco_classes.txt \
        --path_output ./results \
        --images

"""

import os
import sys
import logging

sys.path += [os.path.abspath('.'), os.path.abspath('..')]
from yolo3.yolo import YOLO
from yolo3.utils import update_path
from scripts.predict import predict_image, predict_video, arg_params_yolo


def parse_params():
    # class YOLO defines the default value, so suppress any default HERE
    parser = arg_params_yolo()
    parser.add_argument('--images', default=False, action='store_true',
                        help='Image detection mode.')
    # Command line positional arguments -- for video detection mode
    parser.add_argument('--videos', default=False, action='store_true',
                        help='Image detection mode.')
    arg_params = vars(parser.parse_args())
    for k in (k for k in arg_params if 'path' in k):
        arg_params[k] = update_path(arg_params[k])
        assert os.path.exists(arg_params[k]), 'missing (%s): %s' % (k, arg_params[k])
    logging.debug('PARAMETERS: \n %s', repr(arg_params))
    return arg_params


def loop_detect_image(yolo, path_output=None):
    while True:
        img_path = input('Input image filename:')
        if img_path.lower() == 'exit':
            return
        predict_image(yolo, img_path, path_output)


def loop_detect_video(yolo, path_output=None):
    while True:
        vid_path = input('Input image filename:')
        if vid_path.lower() == 'exit':
            return
        predict_video(yolo, vid_path, path_output)


def _main(path_weights, path_anchors, model_image_size, path_classes, gpu_num,
          path_output=None, images=False, videos=False):
    assert images or videos, 'nothing to do...'

    yolo = YOLO(path_weights, path_anchors, path_classes, model_image_size,
                gpu_num=gpu_num)

    if images:
        # Image detection mode, disregard any remaining command line arguments
        logging.info('Image detection mode')
        loop_detect_image(yolo, path_output)

    if videos:
        logging.info('Video detection mode')
        loop_detect_video(yolo, path_output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # class YOLO defines the default value, so suppress any default HERE
    arg_params = parse_params()
    _main(**arg_params)
