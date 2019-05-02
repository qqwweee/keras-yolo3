import sys
import collections
import os
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import glob
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.8f')

def detect_img(yolo, img):
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
    
    predictions = yolo.get_predictions(image)
    return predictions

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
        '--folder', type=str,
        help='path to folder containing the test images file, default ' + ""
    )

    parser.add_argument(
        '--output', type=str,
        help='path to output folder, default ' + ""
    )

    FLAGS = parser.parse_args()

    if FLAGS.folder:
        print("Evaluating Model")
	yolo = YOLO(**vars(FLAGS))
	formatted_predictions = []
	counter = 0
	size = glob.glob(FLAGS.folder + '*.jpg')
	for filename in glob.glob(FLAGS.folder + '*.jpg'): 
	    counter += 1
	    print(counter, "/", len(size))
            predictions = detect_img(yolo, filename)
            head, tail = os.path.split(filename)            
            for predicted_class, box, score in predictions:
		d = collections.OrderedDict()
                d["name"] = tail
                d["timestamp"] = 10000
                d["category"] = predicted_class
                d["bbox"] = [float(box[1]), float(box[0]), float(box[3]), float(box[2])]
                d["score"] = float(score)
		formatted_predictions.append(d)
	yolo.close_session()
        with open('ev_test.json', 'w') as outfile:
            json.dump(formatted_predictions, outfile, indent=2)
    else:
        print("Must specify at least a test folder path and output path.  See usage with --help.")
