from os import getcwd
import xml.etree.ElementTree as ET

# sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets = ['train', 'val', 'test']

classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]


def convert_annotation(year, image_id, voc_folder):
    # in_file = open(f'{voc_folder}/VOC{year}/Annotations/{image_id}.xml')
    with open(f'{voc_folder}/VOC{year}/Annotations/{image_id}.xml') as in_file:
        tree = ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult)==1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            bb = (
                # int(xmlbox.find('xmin').text),
                # int(xmlbox.find('ymin').text),
                # int(xmlbox.find('xmax').text),
                # int(xmlbox.find('ymax').text)
                int(xmlbox.find('ymin').text), # top
                int(xmlbox.find('xmin').text), # left
                int(xmlbox.find('xmax').text), # right
                int(xmlbox.find('ymax').text)  # bottom
            )
            return " " + ",".join([str(val) for val in bb]) + ',' + str(cls_id)

# wd = getcwd()

def convert(voc_folder='./VOCdevkit', year=2012, output_path='./'):
    for image_set in sets:
        try:
            image_ids = open(f'{voc_folder}/VOC{year}/ImageSets/Main/{image_set}.txt').read().strip().split()
            with open(f'{output_path}/converted_{year}_{image_set}.txt', 'w') as output_file:
                for image_id in image_ids:
                    output_file.write(f'{voc_folder}/VOC{year}/JPEGImages/{image_id}.jpg')
                    output_file.write(convert_annotation(year, image_id, voc_folder))
                    output_file.write('\n')

        except FileNotFoundError as fnf_error:
            print(f'File Not Found for {image_set}: {fnf_error.strerror}')
    print('All good')
