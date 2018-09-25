"""
Retrain the YOLO model for your own dataset.
"""

import os
import sys
import logging
from functools import partial

import numpy as np
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

sys.path += [os.path.abspath('.'), os.path.abspath('..')]
from yolo3.model import create_model, create_model_tiny, data_generator
from scripts.predict import arg_params_yolo

FULL_TRAIN = True


def parse_params():
    # class YOLO defines the default value, so suppress any default HERE
    parser = arg_params_yolo()
    parser.add_argument('--path_annot', type=str, required=True,
                        help='path to the train source')
    return parser.parse_args()


def _main(path_annot, path_weights, path_output, path_anchors, path_classes):
    class_names = get_classes(path_classes)
    nb_classes = len(class_names)
    anchors = get_anchors(path_anchors)

    input_shape = (416, 416)  # multiple of 32, hw

    is_tiny_version = len(anchors) == 6  # default setting
    _create_model = create_model_tiny if is_tiny_version else create_model
    model = _create_model(input_shape, anchors, nb_classes, freeze_body=2,
                          weights_path=path_weights)

    logging = TensorBoard(log_dir=path_output)
    checkpoint_name = 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(os.path.join(path_output, checkpoint_name),
                                 monitor='val_loss', save_weights_only=True,
                                 save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(path_annot) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    batch_size = 32
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    _yolo_loss = lambda y_true, y_pred: y_pred  # use custom yolo_loss Lambda layer.
    _data_generator = partial(data_generator, batch_size=batch_size, input_shape=input_shape,
                              anchors=anchors, nb_classes=nb_classes)

    if FULL_TRAIN:
        model.compile(optimizer=Adam(lr=1e-3),
                      loss={'yolo_loss': _yolo_loss})

        logging.info('Train on %i samples, val on %i samples, with batch size %i.',
                     num_train, num_val, batch_size)
        model.fit_generator(_data_generator(lines[:num_train]),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=_data_generator(lines[num_train:]),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=50,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint])
        model.save_weights(os.path.join(path_output, 'trained_weights_stage.h5'))

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    logging.info('Unfreeze all of the layers.')
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(optimizer=Adam(lr=1e-4),
                  loss={'yolo_loss': _yolo_loss})
    logging.info('Train on %i samples, val on %i samples, with batch size %i.',
                 num_train, num_val, batch_size)
    model.fit_generator(_data_generator(lines[:num_train]),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=_data_generator(lines[num_train:]),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=100,
                        initial_epoch=50,
                        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    model.save_weights(os.path.join(path_output, 'trained_weights_final.h5'))


def get_classes(classes_path):
    """loads the classes"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    params = parse_params()
    _main(**vars(params))
    logging.info('Done')
