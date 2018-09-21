"""
Detection in iteractive mode
"""

import os
import sys
import logging
import time

from PIL import Image
import numpy as np

sys.path += [os.path.abspath('.'), os.path.abspath('..')]
from yolo3.yolo import YOLO
from scripts.predict import predict_image, arg_params_yolo


def parse_params():
    # class YOLO defines the default value, so suppress any default HERE
    parser = arg_params_yolo()
    parser.add_argument('--image', default=False, action='store_true',
                        help='Image detection mode.')
    # Command line positional arguments -- for video detection mode
    parser.add_argument('--video', nargs='?', type=str, required=False,
                        default='./path2your_video', help='Video input path.')
    return parser.parse_args()


def detect_img(yolo):
    while True:
        img_path = input('Input image filename:')
        predict_image(yolo, img_path)


def detect_video(yolo, video_path, output_path=''):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError('Could not open webcam or video')
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != '' else False
    if isOutput:
        logging.error('!!! TYPE: %s, %s, %s, %s', type(output_path), type(video_FourCC),
                      type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = 'FPS: ??'
    while vid.isOpened():
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        start_time = time.time()
        image, pred_items = yolo.detect_image(image)
        result = np.asarray(image)
        exec_time = time.time() - start_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = 'FPS: ' + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # class YOLO defines the default value, so suppress any default HERE
    arg_params = parse_params()

    yolo = YOLO(**vars(arg_params))

    if arg_params.image:
        # Image detection mode, disregard any remaining command line arguments
        logging.info('Image detection mode')
        if hasattr(arg_params, 'video'):
            logging.warning('Ignoring remaining command line arguments: %s , %s',
                            arg_params.video, arg_params.path_output)
        detect_img(yolo)
    elif hasattr(arg_params, 'video'):
        detect_video(yolo, arg_params.video, arg_params.path_output)
    else:
        logging.info('Must specify at least video_input_path.  See usage with --help.')

    yolo.close_session()
