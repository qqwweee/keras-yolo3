"""
Visualisation module

"""

import logging

import numpy as np
from PIL import ImageDraw


def draw_bounding_box(image, predicted_class, box, score, color, thickness):
    """draw bounding box

    :param image:
    :param predicted_class:
    :param box:
    :param score:
    :param color:
    :param thickness:
    :return:

    >>> import os
    >>> from yolo3.utils import update_path, image_open
    >>> img = image_open(os.path.join(update_path('model_data'), 'bike-car-dog.jpg'))
    >>> draw = draw_bounding_box(img, 1, [150, 200, 250, 300], 0.9, color=(255, 0, 0), thickness=3)
    >>> draw  # doctest: +ELLIPSIS
    <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=520x518 at ...>
    """
    label_score = '{} ({:.2f})'.format(predicted_class, score)

    draw = ImageDraw.Draw(image)
    log_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.INFO)
    label_size = draw.textsize(label_score)
    logging.getLogger().setLevel(log_level)

    top, left, bottom, right = box
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    logging.debug(' > %s: (%i, %i), (%i, %i)', label_score, left, top, right, bottom)

    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

    for i in range(thickness):
        draw.rectangle([left + i, top + i, right - i, bottom - i],
                       outline=color)
    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],
                   fill=color)
    draw.text(list(text_origin), label_score, fill=(0, 0, 0))
    del draw
    return image
