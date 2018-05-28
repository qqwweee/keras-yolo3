#! /usr/bin/env python
"""
Reads Darknet config and weights and creates Keras model with TF backend.

"""

import argparse
import configparser
import io
import os
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.layers import (Conv2D, Input, ZeroPadding2D, Add,
                          UpSampling2D, MaxPooling2D, Concatenate)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model as plot


parser = argparse.ArgumentParser(description='Darknet To Keras Converter.')
parser.add_argument('config_path', help='Path to Darknet cfg file.')
parser.add_argument('weights_path', help='Path to Darknet weights file.')
parser.add_argument('output_path', help='Path to output Keras model file.')
parser.add_argument(
    '-p',
    '--plot_model',
    help='Plot generated Keras model and save as image.',
    action='store_true')
parser.add_argument(
    '-w',
    '--weights_only',
    help='Save as Keras weights file instead of model file.',
    action='store_true')

def unique_config_sections(config_file):
    """Convert all config sections to have unique names.

    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream

# %%
def _main(args):
    config_path = os.path.expanduser(args.config_path)
    weights_path = os.path.expanduser(args.weights_path)
    assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(
        config_path)
    assert weights_path.endswith(
        '.weights'), '{} is not a .weights file'.format(weights_path)

    output_path = os.path.expanduser(args.output_path)
    assert output_path.endswith(
        '.h5'), 'output path {} is not a .h5 file'.format(output_path)
    output_root = os.path.splitext(output_path)[0]

    # Load weights and config.
    print('Loading weights.')
    weights_file = open(weights_path, 'rb')
    major, minor, revision = np.ndarray(
        shape=(3, ), dtype='int32', buffer=weights_file.read(12))
    if (major*10+minor)>=2 and major<1000 and minor<1000:
        seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    print('Weights Header: ', major, minor, revision, seen)

    print('Parsing Darknet config.')
    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    print('Creating Keras model.')
    input_layer = Input(shape=(None, None, 3))
    prev_layer = input_layer
    all_layers = []

    weight_decay = float(cfg_parser['net_0']['decay']
                         ) if 'net_0' in cfg_parser.sections() else 5e-4
    count = 0
    out_index = []
    for section in cfg_parser.sections():
        print('Parsing section {}'.format(section))
        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]

            padding = 'same' if pad == 1 and stride == 1 else 'valid'

            # Setting weights.
            # Darknet serializes convolutional weights as:
            # [bias/beta, [gamma, mean, variance], conv_weights]
            prev_layer_shape = K.int_shape(prev_layer)

            weights_shape = (size, size, prev_layer_shape[-1], filters)
            darknet_w_shape = (filters, weights_shape[2], size, size)
            weights_size = np.product(weights_shape)

            print('conv2d', 'bn'
                  if batch_normalize else '  ', activation, weights_shape)

            conv_bias = np.ndarray(
                shape=(filters, ),
                dtype='float32',
                buffer=weights_file.read(filters * 4))
            count += filters

            if batch_normalize:
                bn_weights = np.ndarray(
                    shape=(3, filters),
                    dtype='float32',
                    buffer=weights_file.read(filters * 12))
                count += 3 * filters

                bn_weight_list = [
                    bn_weights[0],  # scale gamma
                    conv_bias,  # shift beta
                    bn_weights[1],  # running mean
                    bn_weights[2]  # running var
                ]

            conv_weights = np.ndarray(
                shape=darknet_w_shape,
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))
            count += weights_size

            # DarkNet conv_weights are serialized Caffe-style:
            # (out_dim, in_dim, height, width)
            # We would like to set these to Tensorflow order:
            # (height, width, in_dim, out_dim)
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
            conv_weights = [conv_weights] if batch_normalize else [
                conv_weights, conv_bias
            ]

            # Handle activation.
            act_fn = None
            if activation == 'leaky':
                pass  # Add advanced activation later.
            elif activation != 'linear':
                raise ValueError(
                    'Unknown activation function `{}` in section {}'.format(
                        activation, section))

            # Create Conv2D layer
            if stride>1:
                # Darknet uses left and top padding instead of 'same' mode
                prev_layer = ZeroPadding2D(((1,0),(1,0)))(prev_layer)
            conv_layer = (Conv2D(
                filters, (size, size),
                strides=(stride, stride),
                kernel_regularizer=l2(weight_decay),
                use_bias=not batch_normalize,
                weights=conv_weights,
                activation=act_fn,
                padding=padding))(prev_layer)

            if batch_normalize:
                conv_layer = (BatchNormalization(
                    weights=bn_weight_list))(conv_layer)
            prev_layer = conv_layer

            if activation == 'linear':
                all_layers.append(prev_layer)
            elif activation == 'leaky':
                act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)

        elif section.startswith('route'):
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            layers = [all_layers[i] for i in ids]
            if len(layers) > 1:
                print('Concatenating route layers:', layers)
                concatenate_layer = Concatenate()(layers)
                all_layers.append(concatenate_layer)
                prev_layer = concatenate_layer
            else:
                skip_layer = layers[0]  # only one layer to route
                all_layers.append(skip_layer)
                prev_layer = skip_layer

        elif section.startswith('maxpool'):
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            all_layers.append(
                MaxPooling2D(
                    pool_size=(size, size),
                    strides=(stride, stride),
                    padding='same')(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('shortcut'):
            index = int(cfg_parser[section]['from'])
            activation = cfg_parser[section]['activation']
            assert activation == 'linear', 'Only linear activation supported.'
            all_layers.append(Add()([all_layers[index], prev_layer]))
            prev_layer = all_layers[-1]

        elif section.startswith('upsample'):
            stride = int(cfg_parser[section]['stride'])
            assert stride == 2, 'Only stride=2 supported.'
            all_layers.append(UpSampling2D(stride)(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('yolo'):
            out_index.append(len(all_layers)-1)
            all_layers.append(None)
            prev_layer = all_layers[-1]

        elif section.startswith('net'):
            pass

        else:
            raise ValueError(
                'Unsupported section header type: {}'.format(section))

    # Create and save model.
    if len(out_index)==0: out_index.append(len(all_layers)-1)
    model = Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index])
    print(model.summary())
    if args.weights_only:
        model.save_weights('{}'.format(output_path))
        print('Saved Keras weights to {}'.format(output_path))
    else:
        model.save('{}'.format(output_path))
        print('Saved Keras model to {}'.format(output_path))

    # Check to see if all weights have been read.
    remaining_weights = len(weights_file.read()) / 4
    weights_file.close()
    print('Read {} of {} from Darknet weights.'.format(count, count +
                                                       remaining_weights))
    if remaining_weights > 0:
        print('Warning: {} unused weights'.format(remaining_weights))

    if args.plot_model:
        plot(model, to_file='{}.png'.format(output_root), show_shapes=True)
        print('Saved model plot to {}.png'.format(output_root))


if __name__ == '__main__':
    _main(parser.parse_args())
