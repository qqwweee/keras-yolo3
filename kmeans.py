import numpy as np
import json
import xml.etree.ElementTree as ET

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def iou(box, clusters):  # 1 box -> k clusters
    weight = np.minimum(clusters[:, 0], box[0])
    height = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(weight == 0) > 0 or np.count_nonzero(height == 0) > 0:
        raise ValueError("Invalid box(h=0 or w=0)")
    intersection = weight * height
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_


def avg_iou(boxes, clusters):
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def kmeans(boxes, k, dist=np.median):
    box_number = boxes.shape[0]
    distances = np.empty((box_number, k))
    last_nearest = np.zeros((box_number,))
    np.random.seed()
    clusters = boxes[np.random.choice(
        box_number, k, replace=False)]  # init k clusters
    while True:
        for row in range(box_number):
            distances[row] = 1 - iou(boxes[row], clusters)

        current_nearest = np.argmin(distances, axis=1)
        if (last_nearest == current_nearest).all():
            break  # clusters won't change
        for cluster in range(k):
            clusters[cluster] = dist(  # update clusters
                boxes[current_nearest == cluster], axis=0)

        last_nearest = current_nearest

    return clusters


def coco_boxes(k):
    dataSet = []
    f = open(
        "mscoco2017/annotations/instances_train2017.json",
        encoding='utf-8')
    data = json.load(f)
    annotations = data['annotations']

    for ant in annotations:
        box_info = ant['bbox']
        width = box_info[2]
        height = box_info[3]
        if width != 0 and height != 0:
            dataSet.append([width, height])
    dataSet = np.array(dataSet)

    return dataSet


def voc_boxes(k):
    dataSet = []
    image_ids = open(
        'VOCdevkit/VOC2012/ImageSets/Main/train.txt').read().strip().split()
    for image_id in image_ids:
        in_file = open('VOCdevkit/VOC2012/Annotations/%s.xml' % (image_id))
        tree = ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            width = int(xmlbox.find('xmax').text) - \
                int(xmlbox.find('xmin').text)
            height = int(xmlbox.find('ymax').text) - \
                int(xmlbox.find('ymin').text)
            dataSet.append([width, height])

    dataSet = np.array(dataSet)
    return dataSet


def result2txt(data):
    f = open("yolo_anchors.txt", 'w')
    row = np.shape(data)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (data[i][0], data[i][1])
        else:
            x_y = ", %d,%d" % (data[i][0], data[i][1])
        f.write(x_y)
    f.close()


if __name__ == "__main__":

    cluster_number = 9
    name = "voc"  # coco

    if name == "coco":
        dataSet = coco_boxes(cluster_number)

    elif name == "voc":
        dataSet = voc_boxes(cluster_number)

    result = kmeans(dataSet, k=cluster_number)
    result = result[np.lexsort(result.T[0, None])]
    result2txt(result)
    print("K anchors:\n {}".format(result))
    print("Accuracy: {:.2f}%".format(avg_iou(dataSet, result) * 100))
