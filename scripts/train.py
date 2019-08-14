"""
Retrain the YOLO model for your own dataset.

    python train.py \
        --path_dataset ./model_data/VOC_2007_train.txt \
        --path_weights ./model_data/tiny-yolo.h5 \
        --path_anchors ./model_data/tiny-yolo_anchors.csv \
        --path_classes ./model_data/voc_classes.txt \
        --path_output ./model_data \
        --path_config ./model_data/train_tiny-yolo.yaml

"""

import os
import sys
import time
import copy
import logging
from functools import partial

import yaml
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Input

sys.path += [os.path.abspath('.'), os.path.abspath('..')]
from yolo3.model import create_model, create_model_tiny, yolo_body, yolo_body_tiny
from yolo3.utils import update_path, get_anchors, get_dataset_class_names, get_nb_classes, data_generator
from scripts.predict import arg_params_yolo

DEFAULT_CONFIG = {
    'image-size': (416, 416),
    'batch-size':
        {'body': 16, 'fine': 16},
    'epochs':
        {'body': 50, 'fine': 50},
    'valid-split': 0.1,
    'generator': {
        'jitter': 0.3,
        'color_hue': 0.1,
        'color_sat': 1.5,
        'color_val': 1.5,
        'resize_img': True,
        'flip_horizontal': True,
        'flip_vertical': False,
        'nb_threads': 0.5,
    }
}
NAME_CHECKPOINT = 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
NAME_TRAIN_CLASSES = 'train_classes.txt'


def parse_params():
    # class YOLO defines the default value, so suppress any default HERE
    parser = arg_params_yolo()
    parser.add_argument('-d', '--path_dataset', type=str, required=True,
                        help='path to the train source - dataset,'
                             ' with single taining instance per line')
    parser.add_argument('--path_config', type=str, required=False,
                        help='path to the train configuration, using YAML format')
    arg_params = vars(parser.parse_args())
    for k in (k for k in arg_params if 'path' in k):
        if not arg_params[k]:
            continue
        arg_params[k] = update_path(arg_params[k])
        assert os.path.exists(arg_params[k]), 'missing (%s): %s' % (k, arg_params[k])
    logging.debug('PARAMETERS: \n %s', repr(arg_params))
    return arg_params


def load_config(path_config, default_config):
    config = copy.deepcopy(default_config)
    if path_config is None or not os.path.isfile(path_config):
        logging.info('Using default configuration')
        return config
    with open(path_config, 'r') as fp:
        conf_user = yaml.safe_load(fp)
    config.update(conf_user)
    return config


def load_training_lines(path_annot, valid_split):
    with open(path_annot) as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = max(1, int(len(lines) * valid_split))
    num_train = len(lines) - num_val

    lines_train = lines[:num_train]
    lines_valid = lines[num_train:]
    return lines_train, lines_valid, num_val, num_train


def _export_classes(class_names, path_output):
    if not class_names or not os.path.isdir(path_output):
        return
    if isinstance(class_names, dict):
        class_names = [class_names.get(i, '') for i in range(max(class_names) + 1)]
    path_txt = os.path.join(path_output, NAME_TRAIN_CLASSES)
    logging.info('exporting label names: %s', path_txt)
    with open(path_txt, 'w') as fp:
        fp.write(os.linesep.join([str(cls) for cls in class_names]))


def _export_model(model, is_tiny_version, image_size, anchors, nb_classes, path_output, name_prefix, name_sufix):
    path_weights = os.path.join(path_output, name_prefix + 'yolo_weights' + name_sufix + '.h5')
    logging.info('Exporting weights: %s', path_weights)
    model.save_weights(path_weights)

    path_model = os.path.join(path_output, name_prefix + 'yolo_trained' + name_sufix + '.h5')
    yolo_ = yolo_body_tiny if is_tiny_version else yolo_body
    factor_ = 2 if is_tiny_version else 3
    yolo_model = yolo_(Input(shape=(image_size[0], image_size[1], 3)), len(anchors) // factor_, nb_classes)
    # make sure model, anchors and classes match
    yolo_model.load_weights(path_weights, by_name=True, skip_mismatch=True)
    logging.info('Exporting model: %s', path_weights)
    yolo_model.save(path_model)


def _main(path_dataset, path_anchors, path_weights=None, path_output='.',
          path_config=None, path_classes=None, gpu_num=1, **kwargs):

    config = load_config(path_config, DEFAULT_CONFIG)
    anchors = get_anchors(path_anchors)

    nb_classes = get_nb_classes(path_dataset)
    logging.info('Using %i classes', nb_classes)
    _export_classes(get_dataset_class_names(path_dataset, path_classes), path_output)

    is_tiny_version = len(anchors) == 6  # default setting
    _create_model = create_model_tiny if is_tiny_version else create_model
    name_prefix = 'tiny-' if is_tiny_version else ''
    model = _create_model(config['image-size'], anchors, nb_classes, freeze_body=2,
                          weights_path=path_weights, gpu_num=gpu_num)

    tb_logging = TensorBoard(log_dir=path_output)
    checkpoint = ModelCheckpoint(os.path.join(path_output, NAME_CHECKPOINT),
                                 monitor='val_loss',
                                 save_weights_only=True,
                                 save_best_only=True,
                                 period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=1)

    lines_train, lines_valid, num_val, num_train = load_training_lines(path_dataset,
                                                                       config['valid-split'])

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    _yolo_loss = lambda y_true, y_pred: y_pred  # use custom yolo_loss Lambda layer.
    _data_generator = partial(data_generator,
                              input_shape=config['image-size'],
                              anchors=anchors,
                              nb_classes=nb_classes,
                              **config['generator'])

    if config['epochs']['body'] > 0:
        model.compile(optimizer=Adam(lr=1e-3),
                      loss={'yolo_loss': _yolo_loss})

        logging.info('Train on %i samples, val on %i samples, with batch size %i.',
                     num_train, num_val, config['batch-size']['body'])
        t_start = time.time()
        model.fit_generator(_data_generator(lines_train, batch_size=config['batch-size']['body']),
                            steps_per_epoch=max(1, num_train // config['batch-size']['body']),
                            validation_data=_data_generator(lines_valid, augument=False),
                            validation_steps=max(1, num_val // config['batch-size']['body']),
                            epochs=config['epochs']['body'],
                            use_multiprocessing=False,
                            initial_epoch=0,
                            callbacks=[tb_logging, checkpoint])
        logging.info('Training took %f minutes', (time.time() - t_start) / 60.)
        _export_model(model, is_tiny_version, config['image-size'], anchors, nb_classes,
                      path_output, name_prefix, '_body')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    logging.info('Unfreeze all of the layers.')
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(optimizer=Adam(lr=1e-4),
                  loss={'yolo_loss': _yolo_loss})
    logging.info('Train on %i samples, val on %i samples, with batch size %i.',
                 num_train, num_val, config['batch-size']['fine'])
    t_start = time.time()
    model.fit_generator(_data_generator(lines_train, batch_size=config['batch-size']['body']),
                        steps_per_epoch=max(1, num_train // config['batch-size']['fine']),
                        validation_data=_data_generator(lines_valid, augument=False),
                        validation_steps=max(1, num_val // config['batch-size']['fine']),
                        epochs=config['epochs']['body'] + config['epochs']['fine'],
                        use_multiprocessing=False,
                        initial_epoch=config['epochs']['body'],
                        callbacks=[tb_logging, checkpoint, reduce_lr, early_stopping])
    logging.info('Training took %f minutes', (time.time() - t_start) / 60.)
    _export_model(model, is_tiny_version, config['image-size'], anchors, nb_classes,
                  path_output, name_prefix, '_final')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_params = parse_params()
    _main(**arg_params)
    logging.info('Done')
