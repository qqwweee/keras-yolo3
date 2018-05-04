"""
Retrain the YOLO model for your own dataset.
"""
import os

import numpy as np
from PIL import Image
from keras.layers import Input, Lambda
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss
from yolo3.utils import letterbox_image

# Default anchor boxes
YOLO_ANCHORS = np.array(((10,13), (16,30), (33,23), (30,61),
    (62,45), (59,119), (116,90), (156,198), (373,326)))

def _main():
    annotation_path = 'train.txt'
    data_path = 'train.npz'
    output_path = 'model_data/my_yolo.h5'
    log_dir = 'logs/000/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32
    image_data, box_data = get_training_data(annotation_path, data_path,
        input_shape, max_boxes=100, load_previous=True)
    y_true = preprocess_true_boxes(box_data, input_shape, anchors, len(class_names))

    infer_model, model = create_model(input_shape, anchors, len(class_names),
        load_pretrained=True, freeze_body=True)

    train(model, image_data/255., y_true, log_dir=log_dir)

    infer_model.save(output_path)



def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS

def get_training_data(annotation_path, data_path, input_shape, max_boxes=100, load_previous=True):
    '''processes the data into standard shape
    annotation row format: image_file_path box1 box2 ... boxN
    box format: x_min,y_min,x_max,y_max,class_index (no space)
    '''
    if load_previous==True and os.path.isfile(data_path):
        data = np.load(data_path)
        print('Loading training data from ' + data_path)
        return data['image_data'], data['box_data']
    image_data = []
    box_data = []
    with open(annotation_path) as f:
        for line in f.readlines():
            line = line.split(' ')
            filename = line[0]
            image = Image.open(filename)
            boxed_image = letterbox_image(image, tuple(reversed(input_shape)))
            image_data.append(np.array(boxed_image,dtype='uint8'))

            boxes = np.zeros((max_boxes,5), dtype='int32')
            for i, box in enumerate(line[1:]):
                if i < max_boxes:
                    boxes[i] = np.array(list(map(int,box.split(','))))
                else:
                    break
            image_size = np.array(image.size)
            input_size = np.array(input_shape[::-1])
            new_size = (image_size * np.min(input_size/image_size)).astype('int32')
            boxes[:i+1, 0:2] = (boxes[:i+1, 0:2]*new_size/image_size + (input_size-new_size)/2).astype('int32')
            boxes[:i+1, 2:4] = (boxes[:i+1, 2:4]*new_size/image_size + (input_size-new_size)/2).astype('int32')
            box_data.append(boxes)
    image_data = np.array(image_data)
    box_data = np.array(box_data)
    np.savez(data_path, image_data=image_data, box_data=box_data)
    print('Saving training data into ' + data_path)
    return image_data, box_data


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=True):
    '''create the training model'''
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)//3
    y_true = [Input(shape=(h//32, w//32, num_anchors, num_classes+5)),
              Input(shape=(h//16, w//16, num_anchors, num_classes+5)),
              Input(shape=(h//8, w//8, num_anchors, num_classes+5))]

    model_body = yolo_body(image_input, num_anchors, num_classes)

    if load_pretrained:
        weights_path = os.path.join('model_data', 'yolo_weights.h5')
        if not os.path.exists(weights_path):
            print("CREATING WEIGHTS FILE" + weights_path)
            yolo_path = os.path.join('model_data', 'yolo.h5')
            orig_model = load_model(yolo_path, compile=False)
            orig_model.save_weights(weights_path)
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        if freeze_body:
            # Do not freeze 3 output layers.
            for i in range(len(model_body.layers)-3):
                model_body.layers[i].trainable = False

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model_body, model

def train(model, image_data, y_true, log_dir='logs/'):
    '''retrain/fine-tune the model'''
    model.compile(optimizer='adam', loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: y_pred})

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
        monitor='val_loss', save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

    model.fit([image_data, *y_true],
              np.zeros(len(image_data)),
              validation_split=.1,
              batch_size=32,
              epochs=30,
              callbacks=[logging, checkpoint, early_stopping])
    model.save_weights(log_dir + 'trained_weights.h5')
    # Further training.



if __name__ == '__main__':
    _main()
