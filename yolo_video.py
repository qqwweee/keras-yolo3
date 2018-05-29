
from yolo import YOLO
from yolo import detect_video

import sys

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: $ python {0} [video_path]", sys.argv[0])
        exit()

    video_path = sys.argv[1]
    detect_video(YOLO(), video_path)
