"""Miscellaneous utility functions."""

import os
import logging
import warnings
from functools import reduce, partial
import multiprocessing as mproc

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

NB_THREADS = max(1, int(mproc.cpu_count() * 0.9))


def update_path(my_path, max_depth=5, abs_path=True):
    """ update path as bobble up strategy

    :param str my_path:
    :param int max_depth:
    :param bool abs_path:
    :return:

    >>> os.path.isdir(update_path('model_data'))
    True
    """
    if not my_path or my_path.startswith('/'):
        return my_path
    elif my_path.startswith('~'):
        return os.path.expanduser(my_path)

    for _ in range(max_depth):
        if os.path.exists(my_path):
            break
        my_path = os.path.join('..', my_path)

    if abs_path:
        my_path = os.path.abspath(my_path)
    return my_path


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    """resize image with unchanged aspect ratio using padding"""
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def rand(low=0, high=1):
    """ random number in given range

    :param float low:
    :param float high:
    :return float:

    >>> 0 <= rand() <= 1
    True
    """
    return np.random.rand() * (high - low) + low


def io_image_decorate(func):
    """ costume decorator to suppers debug messages from the PIL function
    to suppress PIl debug logging
    - DEBUG:PIL.PngImagePlugin:STREAM b'IHDR' 16 13
    :param func:
    :return:
    """
    def wrap(*args, **kwargs):
        log_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.INFO)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            response = func(*args, **kwargs)
        logging.getLogger().setLevel(log_level)
        return response
    return wrap


@io_image_decorate
def image_open(path_img):
    """ just a wrapper to suppers debug messages from the PIL function
    to suppress PIl debug logging - DEBUG:PIL.PngImagePlugin:STREAM b'IHDR' 16 13
    :param str path_img:
    :return Image:
    """
    return Image.open(path_img)


def randomize_image_color(image, hue, sat, val):
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1
    return image_data


def randomize_bbox(box, max_boxes, flip_horizontal, flip_vertical, iw, ih, h, w, nw, nh, dx, dy):
    box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
    box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
    if flip_horizontal:
        box[:, [0, 2]] = w - box[:, [2, 0]]
    if flip_vertical:
        box[:, [1, 3]] = h - box[:, [3, 1]]
    box[:, 0:2][box[:, 0:2] < 0] = 0
    box[:, 2][box[:, 2] > w] = w
    box[:, 3][box[:, 3] > h] = h
    box_w = box[:, 2] - box[:, 0]
    box_h = box[:, 3] - box[:, 1]
    box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
    if len(box) > max_boxes:
        box = box[:max_boxes]
    return box


def normalize_image_bbox(image, box, input_shape, max_boxes, proc_img):
    iw, ih = image.size
    h, w = input_shape
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    image_data = 0
    if proc_img:
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image) / 255.

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
        box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
        box_data[:len(box)] = box

    return image_data, box_data


def get_random_data(annotation_line, input_shape, randomize=True, max_boxes=20,
                    jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True,
                    flip_horizontal=True, flip_vertical=False):
    """randomize pre-processing for real-time data augmentation"""
    line = annotation_line.split()
    image = image_open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    if not randomize:
        # resize image
        image_data, box_data = normalize_image_bbox(image, box, input_shape, max_boxes, proc_img)
        return image_data, box_data

    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip_horizontal = rand() < .5 if flip_horizontal else False
    if flip_horizontal:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    flip_vertical = rand() < .5 if flip_vertical else False
    if flip_vertical:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # distort image
    image_data = randomize_image_color(image, hue, sat, val)

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box_data[:len(box)] = randomize_bbox(box, max_boxes,
                                             flip_horizontal, flip_vertical,
                                             iw, ih, h, w, nw, nh, dx, dy)

    return image_data, box_data


def get_class_names(path_classes):
    logging.debug('loading classes from "%s"', path_classes)
    with open(path_classes) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_nb_classes(path_train_annot=None, path_classes=None):
    if path_classes is not None and os.path.isfile(path_classes):
        class_names = get_class_names(path_classes)
        nb_classes = len(class_names)
    elif path_train_annot is not None and os.path.isfile(path_train_annot):
        with open(path_train_annot) as f:
            lines = f.readlines()
        classes = []
        for ln in lines:
            classes += [bbox.split(',')[-1] for bbox in ln.rstrip().split(' ')[1:]]
        classes = [int(c) for c in classes]
        nb_classes = len(set(classes))
    else:
        logging.warning('No input for extracting classes.')
        nb_classes = 0
    return nb_classes


def get_anchors(path_anchors):
    """loads the anchors from a file"""
    with open(path_anchors) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    """
    assert (true_boxes[..., 4] < num_classes).all(), \
        'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] \
        if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    nb_boxes = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l]
                   for l in range(num_layers)]
    y_true = [np.zeros((nb_boxes, grid_shapes[l][0], grid_shapes[l][1],
                        len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for bi in range(nb_boxes):
        # Discard zero rows.
        wh = boxes_wh[bi, valid_mask[bi]]
        if len(wh) == 0:
            continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[bi, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[bi, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[bi, t, 4].astype('int32')
                    y_true[l][bi, j, i, k, 0:4] = true_boxes[bi, t, 0:4]
                    y_true[l][bi, j, i, k, 4] = 1
                    y_true[l][bi, j, i, k, 5 + c] = 1

    return y_true


def data_generator(annotation_lines, batch_size, input_shape, anchors,
                   nb_classes, randomize=True, max_boxes=20,
                   jitter=0.3, color_hue=0.1, color_sat=1.5, color_val=1.5, proc_img=True,
                   flip_horizontal=True, flip_vertical=False, nb_jobs=NB_THREADS):
    """ data generator for fit_generator """
    nb_lines = len(annotation_lines)
    circ_i = 0
    if nb_lines == 0 or batch_size <= 0:
        return None
    if nb_jobs > 1:
        pool = mproc.Pool(nb_jobs)
    _wrap_rand_data = partial(get_random_data, input_shape=input_shape, randomize=randomize,
                              max_boxes=max_boxes, jitter=jitter, proc_img=proc_img,
                              hue=color_hue, sat=color_sat, val=color_val,
                              flip_horizontal=flip_horizontal, flip_vertical=flip_vertical)

    while True:
        if circ_i < batch_size:
            # shuffle while you are starting new cycle
            np.random.shuffle(annotation_lines)
        image_data = []
        box_data = []

        # create the list of lines to be loaded in batch
        annot_lines = annotation_lines[circ_i:circ_i + batch_size]
        batch_offset = (circ_i + batch_size) - nb_lines
        # chekck if the loaded batch size have sufficient size
        if batch_offset > 0:
            annot_lines += annotation_lines[:batch_offset]
        # multiprocessing loading of batch data
        map_process = pool.imap_unordered if nb_jobs > 1 else map
        for image, box in map_process(_wrap_rand_data, annot_lines):
            image_data.append(image)
            box_data.append(box)

        circ_i = (circ_i + batch_size) % nb_lines

        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape,
                                       anchors, nb_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

    if nb_jobs > 1:
        pool.close()
        pool.join()


def generator_bottleneck(annotation_lines, batch_size, input_shape, anchors, nb_classes,
                         bottlenecks, randomize=False):
    n = len(annotation_lines)
    circ_i = 0
    while True:
        box_data = []
        b0 = np.zeros((batch_size, bottlenecks[0].shape[1],
                       bottlenecks[0].shape[2], bottlenecks[0].shape[3]))
        b1 = np.zeros((batch_size, bottlenecks[1].shape[1],
                       bottlenecks[1].shape[2], bottlenecks[1].shape[3]))
        b2 = np.zeros((batch_size, bottlenecks[2].shape[1],
                       bottlenecks[2].shape[2], bottlenecks[2].shape[3]))
        for b in range(batch_size):
            _, box = get_random_data(annotation_lines[circ_i], input_shape,
                                     randomize=randomize, proc_img=False)
            box_data.append(box)
            b0[b] = bottlenecks[0][circ_i]
            b1[b] = bottlenecks[1][circ_i]
            b2[b] = bottlenecks[2][circ_i]
            circ_i = (circ_i + 1) % n
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, nb_classes)
        yield [b0, b1, b2, *y_true], np.zeros(batch_size)
