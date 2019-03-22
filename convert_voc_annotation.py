import os
import argparse
import xml.etree.ElementTree as ET

#classes = ["bus"]
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def scan_xml_files(dir):
    files = []
    for file in os.listdir(dir):
        if file.endswith(".xml") or file.endswith(".XML"):
            files.append(os.path.join(dir, file))
    return files

def convert_annotation(file):#,list_file):
    in_file = open(file)
    tree=ET.parse(in_file)
    root = tree.getroot()
    write_lines = []

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        write_line = " " + ",".join([str(a) for a in b]) + ',' + str(cls_id)
        write_lines.append(write_line)
        #list_file.write()

    return write_lines


def modify_anno_files(input, imgdir):

    files = scan_xml_files(input)
    lines_to_write = []

    for f in files:
        write_lines = convert_annotation(f)
        if len(write_lines) == 0:
            continue

        basename = os.path.basename(f)
        img_file = basename.replace('xml','jpg')
        fullpath = os.path.join(imgdir, img_file)

        write_lines_join = ''.join(write_lines)
        write_line = '{}{}'.format(fullpath,write_lines_join)
        lines_to_write.append(write_line)


    full_text = '\n'.join(lines_to_write)
    with open('train.txt', 'w+') as the_file:
        the_file.write(full_text)


parser = argparse.ArgumentParser()
parser.add_argument( "-t", dest="input_dir", action="store", type=str, required=False, 
                  help="root directory of annotations", default='./image/' )
parser.add_argument( "-i", dest="image_dir", action="store", type=str, required=True,
                  help="directory of image files")
args = parser.parse_args()
modify_anno_files(args.input_dir, args.image_dir)
