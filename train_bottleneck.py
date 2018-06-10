"""
Retrain the YOLO model for your own dataset.
"""
import os
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data


def _main():
    annotation_path = 'train.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/coco_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw

    model, bottleneck_model, last_layer_model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        # perform bottleneck training
        if not os.path.isfile("bottlenecks.npz"):
            print("calculating bottlenecks")
            batch_size=8
            bottlenecks=bottleneck_model.predict_generator(data_generator_wrapper(lines, batch_size, input_shape, anchors, num_classes, random=False, verbose=True),
             steps=(len(lines)//batch_size)+1, max_queue_size=1)
            np.savez("bottlenecks.npz", bot0=bottlenecks[0], bot1=bottlenecks[1], bot2=bottlenecks[2])
    
        # load bottleneck features from file
        dict_bot=np.load("bottlenecks.npz")
        bottlenecks_train=[dict_bot["bot0"][:num_train], dict_bot["bot1"][:num_train], dict_bot["bot2"][:num_train]]
        bottlenecks_val=[dict_bot["bot0"][num_train:], dict_bot["bot1"][num_train:], dict_bot["bot2"][num_train:]]

        # train last layers with fixed bottleneck features
        batch_size=8
        print("Training last layers with bottleneck features")
        print('with {} samples, val on {} samples and batch size {}.'.format(num_train, num_val, batch_size))
        last_layer_model.compile(optimizer='adam', loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        last_layer_model.fit_generator(bottleneck_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, bottlenecks_train),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=bottleneck_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes, bottlenecks_val),
                validation_steps=max(1, num_val//batch_size),
                epochs=30,
                initial_epoch=0, max_queue_size=1)
        model.save_weights(log_dir + 'trained_weights_stage_0.h5')
        
        # train last layers with random augmented data
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})
        batch_size = 16
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 4 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
            initial_epoch=50,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    # get output of second last layers and create bottleneck model of it
    out1=model_body.layers[246].output
    out2=model_body.layers[247].output
    out3=model_body.layers[248].output
    bottleneck_model = Model([model_body.input, *y_true], [out1, out2, out3])

    # create last layer model of last layers from yolo model
    in0 = Input(shape=bottleneck_model.output[0].shape[1:].as_list()) 
    in1 = Input(shape=bottleneck_model.output[1].shape[1:].as_list())
    in2 = Input(shape=bottleneck_model.output[2].shape[1:].as_list())
    last_out0=model_body.layers[249](in0)
    last_out1=model_body.layers[250](in1)
    last_out2=model_body.layers[251](in2)
    model_last=Model(inputs=[in0, in1, in2], outputs=[last_out0, last_out1, last_out2])
    model_loss_last =Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_last.output, *y_true])
    last_layer_model = Model([in0,in1,in2, *y_true], model_loss_last)

    
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model, bottleneck_model, last_layer_model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, random=True, verbose=False):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0 and random:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=random)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        if verbose:
            print("Progress: ",i,"/",n)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, random=True, verbose=False):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, random, verbose)

def bottleneck_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, bottlenecks):
    n = len(annotation_lines)
    i = 0
    while True:
        box_data = []
        b0=np.zeros((batch_size,bottlenecks[0].shape[1],bottlenecks[0].shape[2],bottlenecks[0].shape[3]))
        b1=np.zeros((batch_size,bottlenecks[1].shape[1],bottlenecks[1].shape[2],bottlenecks[1].shape[3]))
        b2=np.zeros((batch_size,bottlenecks[2].shape[1],bottlenecks[2].shape[2],bottlenecks[2].shape[3]))
        for b in range(batch_size):
            _, box = get_random_data(annotation_lines[i], input_shape, random=False, proc_img=False)
            box_data.append(box)
            b0[b]=bottlenecks[0][i]
            b1[b]=bottlenecks[1][i]
            b2[b]=bottlenecks[2][i]
            i = (i+1) % n
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [b0, b1, b2, *y_true], np.zeros(batch_size)

if __name__ == '__main__':
    _main()
