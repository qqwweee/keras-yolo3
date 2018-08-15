import os
import csv
import numpy as np
import cv2


def get_class_descriptions(class_descriptions_file):
    with open(class_descriptions_file) as f:
        descriptions = f.readlines()
    cls_list = []
    description_table = {}
    for line in descriptions:
        fields = line.strip().split(',')
        assert len(fields) == 2
        cls_list.append(fields[0])
        description_table.update({fields[0]: fields[1]})
    assert len(cls_list) == 500
    assert len(description_table) == 500
    return (cls_list, description_table)


def get_annotations(annotation_path, cls_list):
    ids = []
    annotations = []
    with open(annotation_path) as annofile:
        next(annofile)  # skip the 1st line
        for row in csv.reader(annofile):
            annotation = {'id': row[0], 'label': row[2],
                          'confidence': float(row[3]),
                          'x0': float(row[4]), 'x1': float(row[5]),
                          'y0': float(row[6]), 'y1': float(row[7])}
            assert annotation['label'] in cls_list
            annotations.append(annotation)
            ids.append(row[0])
    ids = list(set(ids))  # remove duplicated ids
    assert len(ids) == 1674979
    assert len(annotations) == 12195144
    return (ids, annotations)


def main():
    cls_list, cls_desc = get_class_descriptions('open-images-dataset/kaggle-2018-object-detection/challenge-2018-class-descriptions-500.csv')
    ids, annotations = get_annotations('open-images-dataset/kaggle-2018-object-detection/challenge-2018-train-annotations-bbox.csv', cls_list)
    ### for testing
    #ids = ['8d6dec80235b6fea']
    #annotations = [{'id': '8d6dec80235b6fea', 'label': '/m/09j5n',
    #                'confidence': 1,
    #                'x0': 0.760000, 'x1': 0.778125,
    #                'y0': 0.645892, 'y1': 0.673277}]

    with open('kaggle_2018_train.txt', 'w') as f:

        def write_line(img_id, img_annos):
            img_path = 'open-images-dataset/train/{}.jpg'.format(img_id)
            img_path = os.path.abspath(img_path)
            img = cv2.imread(img_path)
            if img is None:
                print('{} is not found!!!'.format(img_path))
                return
            h, w, c = img.shape
            #if h != 1024 and w != 1024:
            if h < 416 or w < 416:
                print('{}.jpg is {}x{}!'.format(img_id, w, h))
            f.write(img_path)
            for aa in img_annos:
                f.write(' {},{},{},{},{}'.format(
                        int(aa['x0']*w), int(aa['y0']*h),
                        int(aa['x1']*w), int(aa['y1']*h),
                        cls_list.index(aa['label'])))
            f.write('\n')

        #img_id = annotations[0]['id']
        #img_annos = []
        #for idx, a in enumerate(annotations):
        #    if idx % 10000 == 0:
        #        print('processing annotation #{}'.format(idx))
        #    if img_id == a['id']:
        #        img_annos.append(a)
        #    else:
        #        write_line(img_id, img_annos)
        #        img_id = a['id']
        #        img_annos = [a]
        #write_line(img_id, img_annos)
        anno_dict = {}
        for idx, a in enumerate(annotations):
            if idx % 10000 == 0:
                print('processing annotation #{}'.format(idx))
            if a['id'] not in anno_dict:
                anno_dict[a['id']] = []
            anno_dict[a['id']].append(a)
        for img_id, img_annos in anno_dict.items():
            write_line(img_id, img_annos)


if __name__ == '__main__':
    main()
