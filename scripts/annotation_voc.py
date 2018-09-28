"""
Creating training file fro VOC dataset

>> wget -O ~/Data/VOCtrainval_2007.tar  http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
>> wget -O ~/Data/VOCtest_2007.tar      http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
>> wget -O ~/Data/VOCtrainval_2012.tar  http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
>> wget -O ~/Data/VOCtest_2012.tar      http://pjreddie.com/media/files/VOC2012test.tar

>> tar xopf VOCtrainval_2007.tar --directory ~/Data
>> ...

>> python scripts/annotation_voc.py \
    --path_dataset ~/Data/VOCdevkit \
    --classes aeroplane bicycle bird boat bottle bus car cat chair \
            cow diningtable dog horse motorbike person pottedplant \
            sheep sofa train tvmonitor \
    --sets 2007,train 2007,val \
    --path_output .
"""

import os
import glob
import argparse
import logging
import json
import xml.etree.ElementTree as ET

import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description='Annotation Converter (VOC).')
    parser.add_argument('--path_dataset', type=str, required=True,
                        help='Path to VOC dataset.')
    parser.add_argument('--sets', type=str, required=False, nargs='+',
                        help='List of sets in dataset.',
                        default=('2007,train', '2007,val', '2007,test'))
    parser.add_argument('--path_output', type=str, required=False, default='.',
                        help='Path to output folder.')
    parser.add_argument('--classes', type=str, required=False, default=None, nargs='*',
                        help='Use only following classes.')
    arg_params = vars(parser.parse_args())
    logging.debug('PARAMETERS: %s', repr(arg_params))
    return arg_params


def load_all_classes(path_dataset_year):
    path_annots = glob.glob(os.path.join(path_dataset_year, 'Annotations', '*.xml'))
    classes = []
    for path_annot in path_annots:
        with open(path_annot, 'r') as in_file:
            tree = ET.parse(in_file)
        root = tree.getroot()
        classes += [obj.find('name').text for obj in root.iter('object')]
    classes_uq = list(set(classes))
    return classes_uq


def convert_annotation(path_dataset_year, image_id, classes):
    path_annots = os.path.join(path_dataset_year,
                               'Annotations', '%s.xml' % image_id)
    logging.debug('loading annotations: %s', path_annots)
    with open(path_annots, 'r') as in_file:
        tree = ET.parse(in_file)
    root = tree.getroot()

    records = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if (classes is not None and cls not in classes) or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text),
             int(xmlbox.find('ymin').text),
             int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        records.append(','.join([str(a) for a in b]) + ',' + str(cls_id))
    return records


def _main(path_dataset, path_output, sets, classes=None):
    path_dataset = os.path.abspath(os.path.expanduser(path_dataset))
    assert os.path.isdir(path_dataset), 'missing: %s' % path_dataset
    path_output = os.path.abspath(os.path.expanduser(path_output))
    assert os.path.isdir(path_output), 'missing: %s' % path_output

    if classes is None:
        classes = [load_all_classes(os.path.join(path_dataset,
                                                 'VOC%s' % year_image_set[0]))
                   for year_image_set in sets]
        classes = list(set(classes))

    path_json = os.path.join(path_output, 'voc_classes.txt')
    logging.info('export TXT classes: %s', path_json)
    with open(path_json, 'w') as fp:
        fp.write('\n'.join(classes))

    for year_image_set in sets:
        year, image_set = year_image_set.split(',')
        path_dataset_year = os.path.join(path_dataset, 'VOC%s' % year)
        path_imgs_ids = os.path.join(path_dataset_year, 'ImageSets', 'Main',
                                     '%s.txt' % image_set)
        if not os.path.isfile(path_imgs_ids):
            logging.warning('missing annot. file: %s', path_imgs_ids)
            continue
        logging.info('loading image IDs: %s', path_imgs_ids)
        with open(path_imgs_ids, 'r') as fp:
            image_ids = fp.read().strip().split()

        path_out_list = os.path.join(path_output, 'VOC_%s_%s.txt' % (year, image_set))
        logging.info('creating list file: %s', path_out_list)
        with open(path_out_list, 'w') as list_file:
            for image_id in tqdm.tqdm(image_ids):
                path_img = os.path.join(path_dataset_year,
                                        'JPEGImages', image_id + '.jpg')
                if not os.path.isfile(path_img):
                    logging.warning('missing image: %s', path_img)
                    continue
                recs = convert_annotation(path_dataset_year, image_id, classes)
                list_file.write(path_img + ' ' + ' '.join(recs) + '\n')

    logging.info('Done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_params = parse_arguments()
    _main(**arg_params)
