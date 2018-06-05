import sys

if len(sys.argv) < 2:
    print("Usage: $ python {0} [video_path, [, output_path]]", sys.argv[0])
    exit()

from yolo import YOLO
from yolo import detect_video

if __name__ == '__main__':
    video_path = sys.argv[1]
    detect_video(YOLO(), video_path)
