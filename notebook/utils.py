import re
import random
import sys, os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# sys.path.append(os.path.abspath("../"))
# sys.path.append(os.path.abspath("./"))
# from yolo import YOLO

########################################################################
# 2.detection
########################################################################

def __get_row_format_string(img_path, *boxes):
    return f'{img_path} {" ".join(boxes)}'

# left, top, right, bottom, class_id  
def __get_box_format_string(x_min, y_min, x_max, y_max, class_id):
    return f'{x_min},{y_min},{x_max},{y_max},{class_id}'

def get_row(img_path, coords, classes):
    boxes_string = [
        __get_box_format_string(
            coords[i][0], 
            coords[i][1], 
            coords[i][2], 
            coords[i][3], 
            classes[i]) 
        for i in range(len(coords))]
    return __get_row_format_string(img_path, *boxes_string)



########################################################################
# 3.results-exploration
########################################################################

class BoundingBox:
   # def __init__(self, top, left, right, bottom, prediction):
   #     self.top        = int(top)
   #     self.left       = int(left)
   #     self.right      = int(right)
   #     self.bottom     = int(bottom)
   #     self.prediction = prediction
        
    def __init__(self, bb_string):
        elems = bb_string.split(',')
        assert len(elems) == 5
        self.top        = int(elems[0])
        self.left       = int(elems[1])
        self.right      = int(elems[2])
        self.bottom     = int(elems[3])
        self.prediction = elems[4]
    
    def __repr__(self):
        return f'{self.top},{self.left},{self.right},{self.bottom},{self.prediction}'
    def __str__(self):
        return f'{self.top},{self.left},{self.right},{self.bottom},{self.prediction}'

# ========================================================================

def get_predictions(detection_file_path):
    img2bbs = dict()
    with open(detection_file_path) as detection_file:
        for line in detection_file:
            line  = line.replace('\n', '').split(' ')
            path  = line[0]
            img2bbs[path] = []
            for bb in line[1:]:
                if bb:
                    img2bbs[path] = img2bbs.get(path, []) + [BoundingBox(bb)]
    return img2bbs


def explode_dict(img2bbs):
    img2bb = []
    for key, vals in img2bbs.items():
        for val in vals:
            img2bb.append( (key, val) )
    return img2bb

def back_to_dict(img2bb):
    img2bbs = dict()
    for elem in img2bb:
        path, bb = elem
        img2bbs[path] = img2bbs.get(path, []) + [bb]
    return img2bbs

def count_class(img2bb, class_id):
    class_id = str(class_id)
    return len( [elem for elem in img2bb if elem[1].prediction == class_id] )

def get_value_counts(series, id2class , not_allowed_values=['2', '7', '5', '12', '3']):
    series_allowed     = (series[~series.isin(not_allowed_values)])
    series_not_allowed = (series[series.isin(not_allowed_values)])

    series_allowed     = series_allowed.value_counts().sort_index()
    series_not_allowed = series_not_allowed.value_counts().sort_index()

    series_allowed.index     = (series_allowed.index.to_series().apply(lambda x: f'{x} - {id2class[x]}'))
    series_not_allowed.index = (series_not_allowed.index.to_series().apply(lambda x: f'{x} - {id2class[x]}'))
    return series_allowed, series_not_allowed

def write_annotations(img2bbs, path):
    with open(path, 'w+') as file:
        for img_path, bbs in img2bbs.items():
            line = f'{img_path} {" ".join([str(bb) for bb in img2bbs[img_path]])}\n'
            file.write(line)
    print('All good')
    
def write_list_to_file(l, path):
    with open(path, 'w+') as file:
        for elem in l:
            line = f'{elem}\n'
            file.write(line)
    print('All good')

def get_classes(classes_path):
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

# ========================================================================


def get_image(path):
    # return Image.open(path.replace('../data/input/', '../data/input/old/'))
    return Image.open(path)

def get_output_image(path):
    # return Image.open(path.replace('../data/input/', '../data/output/old/'))
    path = path.split('/')[:4] + path.split('/')[-1:]
    path = '/'.join(path)
    return Image.open(path.replace('../data/input/mhca-cropped/', '../data/output/mhca/'))

def print_coords(img, BoundingBox):
    new_img = img.copy()
    draw = ImageDraw.Draw(new_img)
    draw.rectangle(((BoundingBox.left, BoundingBox.top), (BoundingBox.right, BoundingBox.bottom)), 
                   outline="red")
    return new_img

def combine_images(img1, img2):
    width1, height1 = img1.size
    width2, height2 = img2.size

    total_width = width1 + width2
    max_height  = max(height1, height2)

    image = Image.new('RGB', (total_width, max_height))
    image.paste(img1, (0,0))
    image.paste(img2, (width1,0))
    return image

def get_sample_of(class_id, img2bb, sample_size=1):
    only_one_class = [elem[0] for elem in img2bb if elem[1].prediction == class_id]
    sample = random.sample(only_one_class, sample_size)
    print(sample)
    images = [combine_images(get_image(path), get_output_image(path)) for path in sample]
    return images

def get_sample_from_list(paths, sample_size=1):
    sample = random.sample(paths, sample_size)
    print(sample)
    images = [get_image(path) for path in sample]
    return images



# 4 prep

def __replace_in_string(line, regex, replacements_dict):
    result = line
    line   = line.replace('\n', '')
    match  = re.search(regex, line)
    if match:
        old = match.group(1)
        new = replacements_dict.get(old, old) # get replacement for <old> if not present leav <old>
        result = re.sub(regex, new, line)
    return result
    
def replace_in_file(file, regex, replacements_dict):
    new_lines = []
    with open(file, 'r') as old_file:
        for line in old_file:
            new_line = __replace_in_string(line, regex, replacements_dict)
            new_lines.append(new_line)
    with open(file, 'w') as new_file:
        for new_line in new_lines:
            new_file.write(new_line + '\n')
    print('All good')