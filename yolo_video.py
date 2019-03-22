import sys
import argparse
import os
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

def get_image_files(dir):
    imgs = []
    for file in os.listdir(dir):
        file_lower = file.lower()
        if file_lower.endswith(".png") or file_lower.endswith(".jpg"):
            imgs.append(os.path.join(dir, file))
    return imgs

def detect_imgdir(yolo, dir, output_txt=False):
    img_files = get_image_files(dir)
    save_dir = os.path.join(dir,'out')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    for img in img_files:
        try:
            image = Image.open(img)
        except:
            print('Open Error! {}'.format(img))
            continue
        else:
            fullpath = os.path.join(save_dir, os.path.basename(img))
            detections = list()
            r_image = yolo.detect_image(image, single_image=False, output=detections)

            if not output_txt:
                r_image.save(fullpath,"JPEG")
                print('save {}'.format(fullpath))
            else:
                basename = os.path.basename(img)  # eg. 123.jpg
                txt_file = os.path.splitext(basename)[0]+'.txt'  # eg. 0001
                txt_fullpath = os.path.join(save_dir, txt_file)
                with open(txt_fullpath, 'w+') as the_file:
                    full_text = '\n'.join(detections)
                    the_file.write(full_text)

    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
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
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )

    parser.add_argument(
        '--imgdir', type=str, default='',
        help='Image dir detection mode, will ignore all positional arguments'
    )

    parser.add_argument(
        '--txt', default=False, action="store_true",
        help='Image dir detection will output txt files'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))

    elif os.path.isdir(FLAGS.imgdir):
        print("Image directory mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_imgdir(YOLO(**vars(FLAGS)), FLAGS.imgdir, FLAGS.txt)
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
