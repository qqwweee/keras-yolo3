import numpy as np
import json
from collections import defaultdict


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


if __name__ == "__main__":
    cluster_number = 9
    dataSet = []
    name_box_id = defaultdict(list)
    id_name = dict()
    f = open(
        "mscoco2017/annotations/instances_train2017.json",
        encoding='utf-8')
    data = json.load(f)
    annotations = data['annotations']

    for ant in annotations:
        box_info = ant['bbox']
        width = box_info[2]
        height = box_info[3]
        dataSet.append([width, height])
    dataSet = np.array(dataSet)

    out = kmeans(dataSet, k=cluster_number)
    print("K anchors:\n {}".format(out))
    print("Accuracy: {:.2f}%".format(avg_iou(dataSet, out) * 100))
