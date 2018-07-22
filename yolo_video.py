import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '-i', '--image', default=False, action="store_true",
        help='image detetion mode, will ignore all positional arguments'
    )

    parser.add_argument(
        "video_input", nargs='?', type=str,
        help = "Video input path"
    )

    parser.add_argument(
        "video_output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        print("Image detection mode, ignoring other command line arguments.")
        detect_img(YOLO(**vars(FLAGS)))
    elif "video_input" not in FLAGS:
        print(parser.print_help())
        print("Must specify at least vide_input_path")
    else:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.video_input, FLAGS.video_output)
