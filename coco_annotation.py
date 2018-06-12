import json
from collections import defaultdict

name_box_id = defaultdict(list)
id_name = dict()
f = open(
    "mscoco2017/annotations/instances_val2017.json",
    encoding='utf-8')
data = json.load(f)

annotations = data['annotations']
for ant in annotations:
    id = ant['image_id']
    name = 'mscoco2017/val2017/%012d.jpg' % id
    cat = ant['category_id']

    if cat >= 1 and cat <= 11:
        cat = cat
    elif cat >= 13 and cat <= 25:
        cat = cat - 1
    elif cat >= 27 and cat <= 28:
        cat = cat - 2
    elif cat >= 31 and cat <= 44:
        cat = cat - 4
    elif cat >= 46 and cat <= 65:
        cat = cat - 5
    elif cat == 67:
        cat = cat - 6
    elif cat == 70:
        cat = cat - 8
    elif cat >= 72 and cat <= 82:
        cat = cat - 9
    elif cat >= 84 and cat <= 90:
        cat = cat - 10

    name_box_id[name].append([ant['bbox'], cat])

f = open('train.txt', 'w')
for key in name_box_id.keys():
    f.write(key)
    box_infos = name_box_id[key]
    for info in box_infos:
        box_info = " %d,%d,%d,%d,%d" % (int(info[0][0]) ,int(info[0][1]),int(info[0][2]) ,int(info[0][3]) , int(info[1]))
        f.write(box_info)
    f.write('\n')
f.close()
