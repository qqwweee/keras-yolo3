import sys
import cv2
import argparse
from yolo import YOLO, yolo_parser, detect_video


FLAGS = None

if __name__ == '__main__':
    parser = yolo_parser()

    # Add the video_path and optional output video_path to the default cmdline options supported by class YOLO
    parser.add_argument("video_path", help="input video path")
    parser.add_argument(
      '--output',
      type=str,
      default="",
      help='[optional] output video path'
    )

    FLAGS = parser.parse_args()

    detect_video(YOLO(**vars(FLAGS)), FLAGS.video_path, FLAGS.output)
