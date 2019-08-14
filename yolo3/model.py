"""
YOLO_v3 Model Defined in Keras.
"""

import os
import logging
from functools import wraps

import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers import Input, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.utils import multi_gpu_model

from yolo3.utils import compose, update_path


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
    )


def resblock_body(x, num_filters, num_blocks):
    """A series of resblocks starting with a downsampling Convolution2D"""
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    """Darknent body having 52 Convolution2D layers"""
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    """6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer"""
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras.

    :param inputs:
    :param int num_anchors:
    :param int num_classes:
    :return:

    >>> yolo_body(Input(shape=(None, None, 3)), 6, 10)  #doctest: +ELLIPSIS
    <keras.engine.training.Model object at ...>
    """
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))

    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    return Model(inputs, [y1, y2, y3])


def yolo_body_tiny(inputs, num_anchors, num_classes):
    """Create Tiny YOLO_v3 model CNN body in keras.

    :param inputs:
    :param int num_anchors:
    :param int num_classes:
    :return:

    >>> yolo_body_tiny(Input(shape=(None, None, 3)), 6, 10)  #doctest: +ELLIPSIS
    <keras.engine.training.Model object at ...>
    """
    x1 = compose(
        DarknetConv2D_BN_Leaky(16, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(128, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(256, (3, 3)))(inputs)
    x2 = compose(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        DarknetConv2D_BN_Leaky(1024, (3, 3)),
        DarknetConv2D_BN_Leaky(256, (1, 1)))(x1)
    y1 = compose(
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))(x2)
    x3 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x2)
    y2 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(256, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))([x3, x1])

    return Model(inputs, [y1, y2])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1],
                              num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """Get corrected boxes"""
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """Process Conv layer output"""
    box_xy, box_wh, box_confidence, box_class_probs = \
        yolo_head(feats, anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs, anchors, num_classes, image_shape, max_boxes=20,
              score_threshold=.6, iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] \
        if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]],
                                                    num_classes, input_shape,
                                                    image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor,
            iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def box_iou(b1, b2):
    """Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    Example
    -------
    >>> bbox1 = K.variable(value=[250, 200, 150, 100], dtype='float32')
    >>> bbox2 = K.variable(value=[300, 250, 100, 100], dtype='float32')
    >>> iou = box_iou(bbox1, bbox2)
    >>> iou
    <tf.Tensor 'truediv_2:0' shape=(1,) dtype=float32>
    >>> K.eval(iou)
    array([0.1764706], dtype=float32)
    """
    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=0.5, print_loss=False):
    """Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or yolo_body_tiny
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    """
    num_layers = len(anchors) // 3  # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] \
        if num_layers == 3 else [[3, 4, 5], [0, 1, 2]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32,
                         K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0]))
                   for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]],
                                                     num_classes, input_shape,
                                                     calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        # Keras switch allows scalr condition, bit here is expected to have elemnt-wise
        #  also the `object_mask` has in last dimension 1 but the in/out puts has 2 (some replication)
        # raw_true_wh = tf.where(tf.greater(K.concatenate([object_mask] * 2), 0),
        #                        raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        raw_true_wh = K.switch(object_mask, raw_true_wh,
                               K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def _loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4],
                                       object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh,
                                                      K.dtype(true_box)))
            return b + 1, ignore_mask

        _, ignore_mask = K.control_flow_ops.while_loop(
            lambda b, *args: b < m, _loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        ce = K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                   from_logits=True)
        xy_loss = object_mask * box_loss_scale * ce
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        ce_loss = K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                        from_logits=True)
        confidence_loss = object_mask * ce_loss + (1 - object_mask) * ce_loss * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs,
                                                         raw_pred[..., 5:],
                                                         from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss,
                                   class_loss, K.sum(ignore_mask)],
                            message='loss: ')
    return loss


def create_model(input_shape, anchors, num_classes, weights_path=None, model_factor=3,
                 freeze_body=2, ignore_thresh=0.5, gpu_num=1):
    """create the training model"""
    _INPUT_SHAPES = {0: 32, 1: 16, 2: 8, 3: 4}
    _FACTOR_YOLO_BODY = {2: yolo_body_tiny, 3: yolo_body}
    _FACTOR_FREEZEING = {2: 20, 3: 185}
    _LOSS_ARGUMENTS = {
        'anchors': anchors,
        'num_classes': num_classes,
        'ignore_thresh': ignore_thresh
    }

    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    model_body = _FACTOR_YOLO_BODY[model_factor](image_input, num_anchors // model_factor, num_classes)
    logging.debug('Create YOLOv3 (model_factor: %i) model with %i anchors and %i classes.',
                  model_factor, num_anchors, num_classes)

    if weights_path is not None:
        # model_body = load_model(weights_path, compile=False)
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        logging.info('Load model "%s".', weights_path)
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (_FACTOR_FREEZEING[model_factor],
                   len(model_body.layers) - model_factor)[freeze_body - 1]
            logging.info('Freeze the first %i layers of total %i layers.',
                         num, len(model_body.layers))
            for i in range(num):
                model_body.layers[i].trainable = False

    model_loss_fn = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                           arguments=_LOSS_ARGUMENTS)
    y_true = [Input(shape=(h // {i: _INPUT_SHAPES[i] for i in range(model_factor)}[l],
                           w // {i: _INPUT_SHAPES[i] for i in range(model_factor)}[l],
                           num_anchors // model_factor,
                           num_classes + 5))
              for l in range(model_factor)]
    model_loss = model_loss_fn([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    logging.debug(model.summary(line_length=120))

    if gpu_num >= 2:
        model = multi_gpu_model(model, gpus=gpu_num)

    return model


def create_model_tiny(input_shape, anchors, num_classes, weights_path=None,
                      freeze_body=2, ignore_thresh=0.5, gpu_num=1):
    """create the training model, for Tiny YOLOv3 """

    return create_model(input_shape, anchors, num_classes, weights_path, model_factor=2,
                        freeze_body=freeze_body, ignore_thresh=ignore_thresh, gpu_num=gpu_num)


def create_model_bottleneck(input_shape, anchors, num_classes, freeze_body=2,
                            weights_path=None, gpu_num=1):
    """create the training model"""
    # K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l],
                           w // {0: 32, 1: 16, 2: 8}[l],
                           num_anchors // 3,
                           num_classes + 5))
              for l in range(3)]

    _LOSS_ARGUMENTS = {
        'anchors': anchors,
        'num_classes': num_classes,
        'ignore_thresh': 0.5
    }

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    logging.info('Create YOLOv3 model with %i anchors and %i classes.',
                 num_anchors, num_classes)

    if weights_path is not None:
        weights_path = update_path(weights_path)
        if os.path.isfile(weights_path):
            logging.warning('missing weights: %s', weights_path)
        else:
            logging.info('Load weights %s.', weights_path)
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            if freeze_body in [1, 2]:
                # Freeze darknet53 body or freeze all but 3 output layers.
                num = (185, len(model_body.layers) - 3)[freeze_body - 1]
                for i in range(num):
                    model_body.layers[i].trainable = False
                logging.info('Freeze the first %i layers of total %i layers.',
                             num, len(model_body.layers))

    # get output of second last layers and create bottleneck model of it
    out1 = model_body.layers[246].output
    out2 = model_body.layers[247].output
    out3 = model_body.layers[248].output
    model_bottleneck = Model([model_body.input, *y_true], [out1, out2, out3])

    # create last layer model of last layers from yolo model
    in0 = Input(shape=model_bottleneck.output[0].shape[1:].as_list())
    in1 = Input(shape=model_bottleneck.output[1].shape[1:].as_list())
    in2 = Input(shape=model_bottleneck.output[2].shape[1:].as_list())
    last_out0 = model_body.layers[249](in0)
    last_out1 = model_body.layers[250](in1)
    last_out2 = model_body.layers[251](in2)
    model_last = Model(inputs=[in0, in1, in2], outputs=[last_out0, last_out1, last_out2])
    fn_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss', arguments=_LOSS_ARGUMENTS)
    model_loss_last = fn_loss([*model_last.output, *y_true])
    last_layer_model = Model([in0, in1, in2, *y_true], model_loss_last)

    fn_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss', arguments=_LOSS_ARGUMENTS)
    model_loss = fn_loss([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    if gpu_num >= 2:
        model = multi_gpu_model(model, gpus=gpu_num)
        model_bottleneck = multi_gpu_model(model_bottleneck, gpus=gpu_num)

    return model, model_bottleneck, last_layer_model
