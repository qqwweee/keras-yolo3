"""
Creating training file from costume dataset

>> python annotation_csv.py \
    --path_dataset ~/Data/PeopleDetections \
    --path_output ../model_data
"""

import os
import glob
import argparse
import logging

import pandas as pd
import tqdm

IMAGE_EXTENSIONS = ['.png', '.jpg']
ANNOT_COLUMS = ['xmin', 'ymin', 'xmax', 'ymax', 'class']


def parse_arguments():
    parser = argparse.ArgumentParser(description='Annotation Converter (VOC).')
    parser.add_argument('--path_dataset', type=str, required=True,
                        help='Path to VOC dataset.')
    parser.add_argument('--path_output', type=str, required=False, default='.',
                        help='Path to output folder.')
    arg_params = vars(parser.parse_args())
    logging.debug('PARAMETERS: %s', repr(arg_params))
    return arg_params


def convert_annotation(path_csv, classes=None):
    df = pd.read_csv(path_csv)
    if 'class' in df.columns and classes:
        df = df[df['class'].isin(classes)]
    elif 'class' not in df.columns:
        df['class'] = 0

    records = []
    for idx, row in df[ANNOT_COLUMS].iterrows():
        records.append(','.join([str(v) for v in row]))
    return records


def _main(path_dataset, path_output, classes=None):
    path_dataset = os.path.abspath(os.path.expanduser(path_dataset))
    assert os.path.isdir(path_dataset), 'missing: %s' % path_dataset
    path_output = os.path.abspath(os.path.expanduser(path_output))
    assert os.path.isdir(path_output), 'missing: %s' % path_output

    name_dataset = os.path.basename(path_dataset)
    list_csv = sorted(glob.glob(os.path.join(path_dataset, '*.csv')))

    path_out_list = os.path.join(path_output, '%s_train.txt' % name_dataset)
    logging.info('creating list file: %s', path_out_list)

    with open(path_out_list, 'w') as list_file:
        for path_csv in tqdm.tqdm(list_csv):
            name = os.path.splitext(os.path.basename(path_csv))[0]

            list_images = []
            for ext in IMAGE_EXTENSIONS:
                list_images += glob.glob(os.path.join(path_dataset, name + ext))
            if not list_images:
                logging.warning('missing image: %s', os.path.join(path_dataset, name))
                continue
            recs = convert_annotation(path_csv, classes)
            path_img = sorted(list_images)[0]
            list_file.write(path_img + ' ' + ' '.join(recs) + '\n')

    logging.info('Done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_params = parse_arguments()
    _main(**arg_params)
