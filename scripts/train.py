"""
Retrain the YOLO model for your own dataset.

>> python train.py \
       --path_annot model_data/VOC_2007_train.txt \
       --path_weights model_data/tiny-yolo.h5 \
       --path_anchors model_data/tiny-yolo_anchors.txt \
       --path_classes model_data/voc_classes.txt \
       --path_output model_data
"""

import os
import sys
import time
import copy
import json
import logging
from functools import partial

import numpy as np
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

sys.path += [os.path.abspath('.'), os.path.abspath('..')]
from yolo3.model import create_model, create_model_tiny, data_generator
from yolo3.utils import update_path, get_anchors, get_nb_classes
from scripts.predict import arg_params_yolo, path_assers

DEFAULT_CONFIG = {
    'image-size': (416, 416),
    'batch-size': 32,
    'epochs-body': 50,
    'epochs-fine': 50,
    'valid-split': 0.1,
}


def parse_params():
    # class YOLO defines the default value, so suppress any default HERE
    parser = arg_params_yolo()
    parser.add_argument('--path_annot', type=str, required=True,
                        help='path to the train source')
    parser.add_argument('--path_config', type=str, required=False,
                        help='path to the train configuration')
    arg_params = vars(parser.parse_args())
    logging.debug('PARAMETERS: \n %s', repr(arg_params))
    return arg_params


def load_config(path_config):
    if path_config is None or not os.path.isfile(path_config):
        return copy.deepcopy(DEFAULT_CONFIG)
    config = copy.deepcopy(DEFAULT_CONFIG)
    with open(path_config, 'r') as fp:
        conf_user = json.load(fp)
    config.update(conf_user)
    return config


def load_training_lines(path_annot, valid_split):
    with open(path_annot) as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * valid_split)
    num_train = len(lines) - num_val

    lines_train = lines[:num_train]
    lines_valid = lines[num_train:]
    return lines_train, lines_valid, num_val, num_train


def _main(path_annot, path_anchors, path_weights=None, path_output='.',
          path_config=None, path_classes=None, gpu_num=1):
    path_weights, path_anchors, path_classes, path_output = path_assers(
        path_weights, path_anchors, path_classes, path_output)
    path_annot = update_path(path_annot)
    assert os.path.isfile(path_annot), 'missing "%s"' % path_annot

    config = load_config(path_config)
    anchors = get_anchors(path_anchors)

    nb_classes = get_nb_classes(path_annot)
    logging.info('Using %i classes', nb_classes)

    is_tiny_version = len(anchors) == 6  # default setting
    _create_model = create_model_tiny if is_tiny_version else create_model
    name_prefix = 'tiny-' if is_tiny_version else ''
    model = _create_model(config['image-size'], anchors, nb_classes, freeze_body=2,
                          weights_path=path_weights)

    tb_logging = TensorBoard(log_dir=path_output)
    checkpoint_name = 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(os.path.join(path_output, checkpoint_name),
                                 monitor='val_loss', save_weights_only=True,
                                 save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    lines_train, lines_valid, num_val, num_train = load_training_lines(path_annot,
                                                                       config['valid-split'])

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    _yolo_loss = lambda y_true, y_pred: y_pred  # use custom yolo_loss Lambda layer.
    _data_generator = partial(data_generator, batch_size=config['batch-size'],
                              input_shape=config['image-size'],
                              anchors=anchors, nb_classes=nb_classes)

    if config['epochs-body'] > 0:
        model.compile(optimizer=Adam(lr=1e-3),
                      loss={'yolo_loss': _yolo_loss})

        logging.info('Train on %i samples, val on %i samples, with batch size %i.',
                     num_train, num_val, config['batch-size'])
        t_start = time.time()
        model.fit_generator(_data_generator(lines_train),
                            steps_per_epoch=max(1, num_train // config['batch-size']),
                            validation_data=_data_generator(lines_valid),
                            validation_steps=max(1, num_val // config['batch-size']),
                            epochs=config['epochs-body'],
                            use_multiprocessing=True,
                            initial_epoch=0,
                            callbacks=[tb_logging, checkpoint])
        logging.info('Training took %f minutes', (time.time() - t_start) / 60.)
        path_weights = os.path.join(path_output, name_prefix + 'yolo_trained_body.h5')
        logging.info('Exporting weights: %s', path_weights)
        model.save_weights(path_weights)

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    logging.info('Unfreeze all of the layers.')
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(optimizer=Adam(lr=1e-4),
                  loss={'yolo_loss': _yolo_loss})
    logging.info('Train on %i samples, val on %i samples, with batch size %i.',
                 num_train, num_val, config['batch-size'])
    t_start = time.time()
    model.fit_generator(_data_generator(lines_train),
                        steps_per_epoch=max(1, num_train // config['batch-size']),
                        validation_data=_data_generator(lines_valid),
                        validation_steps=max(1, num_val // config['batch-size']),
                        epochs=config['epochs-body'] + config['epochs-fine'],
                        use_multiprocessing=True,
                        initial_epoch=config['epochs-fine'],
                        callbacks=[tb_logging, checkpoint, reduce_lr, early_stopping])
    logging.info('Training took %f minutes', (time.time() - t_start) / 60.)
    path_weights = os.path.join(path_output, name_prefix + 'yolo_trained_final.h5')
    logging.info('Exporting weights: %s', path_weights)
    model.save_weights(path_weights)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    a_params = parse_params()
    _main(**a_params)
    logging.info('Done')
