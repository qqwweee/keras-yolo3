import sys
import argparse
import glob
import os
from yolo import YOLO, detect_video
from PIL import Image

class NotFoundError(Exception):
    pass

def get_unused_dir_num(pdir, pref=None):
    os.makedirs(pdir, exist_ok=True)
    dir_list = os.listdir(pdir)
    for i in range(1000):
        search_dir_name = "" if pref is None else (
            pref + "_" ) + '%03d' % i
        if search_dir_name not in dir_list:
            return os.path.join(pdir, search_dir_name)
    raise NotFoundError('Error')

def detect_img(yolo):

    image_glob = FLAGS.image_glob
    print(image_glob)
    print(FLAGS.model)
    result_name = os.path.basename(FLAGS.model)

    img_path_list = glob.glob(image_glob)

    output_dir = get_unused_dir_num(pdir="results/", pref=result_name)
    image_output_dir = os.path.join(output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    for img_path in img_path_list:
        img_basename = os.path.basename(img_path)
        try:
            image = Image.open(img_path)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image, result = yolo.detect_image(image)
            r_image.save(
                os.path.join(
                    image_output_dir,
                    img_basename + ".jpg",
                ))
            print(result)

    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str, default=YOLO.get_defaults("model_path"),
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
        "-i", "--image_glob", nargs='?', type=str, default="images/pics/*jpg",
        help="Image glob pattern"
    )

    FLAGS = parser.parse_args()

    detect_img(YOLO(**vars(FLAGS)))
