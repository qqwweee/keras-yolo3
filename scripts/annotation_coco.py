"""
Creating training file fro COCO dataset

>> python annotation_coco.py \
    --path_annot /home/jb/Data/COCO/annotations/instances_train2014.json \
    --path_images /home/jb/Data/COCO/train2014 \
    --path_output .
"""

import os
import logging
import argparse
import json
from collections import defaultdict

import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description='Annotation Converter (COCO).')
    parser.add_argument('--path_annot', type=str, required=True,
                        help='Path to annotation for COCO dataset.')
    parser.add_argument('--path_images', type=str, required=True,
                        help='Path to images of VOC dataset.')
    parser.add_argument('--path_output', type=str, required=False, default='.',
                        help='Path to output folder.')
    arg_params = vars(parser.parse_args())
    logging.debug('PARAMETERS: %s', repr(arg_params))
    return arg_params


def chenge_category(cat):
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11
    return cat


def get_bounding_box(info):
    x_min = int(info[0][0])
    y_min = int(info[0][1])
    x_max = x_min + int(info[0][2])
    y_max = y_min + int(info[0][3])
    box = '%d,%d,%d,%d,%d' % (x_min, y_min,
                              x_max, y_max, int(info[1]))
    return box


def _main(path_annot, path_images, path_output):
    path_annot = os.path.abspath(os.path.expanduser(path_annot))
    assert os.path.isfile(path_annot), 'missing: %s' % path_annot
    path_dataset = os.path.abspath(os.path.expanduser(path_images))
    assert os.path.isdir(path_dataset), 'missing: %s' % path_dataset
    path_output = os.path.abspath(os.path.expanduser(path_output))
    assert os.path.isdir(path_output), 'missing: %s' % path_output

    name_box_id = defaultdict(list)
    logging.info('loading annotations "%s"', path_annot)
    with open(path_annot, encoding='utf-8') as fp:
        data = json.load(fp)

    annotations = data['annotations']
    for ant in tqdm.tqdm(annotations):
        id = ant['image_id']
        name_img = 'COCO_%s_%012d.jpg' % (os.path.basename(path_images), id)
        path_img = os.path.join(path_images, name_img)
        if not os.path.isfile(path_img):
            logging.debug('missing image: %s', path_img)
            continue
        cat = chenge_category(ant['category_id'])
        name_box_id[path_img].append([ant['bbox'], cat])

    name_out_list = os.path.basename(path_annot).replace('.json', '.txt')
    path_out_list = os.path.join(path_output, 'COCO_' + name_out_list)
    logging.info('creating out list "%s"', path_out_list)
    with open(path_out_list, 'w') as fp:
        for key in tqdm.tqdm(name_box_id.keys()):
            box_infos = name_box_id[key]
            bboxes = []
            for info in box_infos:
                bboxes.append(get_bounding_box(info))
            fp.write('%s %s\n' % (key, ' '.join(bboxes)))

    logging.info('Done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_params = parse_arguments()
    _main(**arg_params)
