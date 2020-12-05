import numpy as np
import argparse

"""
class id should start with 0, not 1.
"""

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation")
    parser.add_argument("classes")
    return parser.parse_args()

def main():
    args = get_arguments()
    annotation_txt_path = args.annotation
    classes_txt_path = args.classes

    with open(classes_txt_path, "r") as f:
        lines = f.read().split("\n")
        count = 0
        for line in lines:
            if len(line) <= 0:
                break

            count += 1

    class_id_list = [0 for _ in range(count)]
    with open(annotation_txt_path, "r") as f:
        lines = f.read().split("\n")
        for line in lines:
            contents = line.split(" ")
            annotation_info = contents[1:]
            for anno in annotation_info:
                anno_values = anno.split(",")
                class_id = int(anno_values[-1])
                class_id_list[class_id] += 1

        for id, result in enumerate(class_id_list):
            print("[{}]: {}".format(id, result))


if __name__ == '__main__':
    main()
