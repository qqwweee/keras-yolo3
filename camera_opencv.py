import cv2
from base_camera import BaseCamera
# from yolo_court import YOLO
from timeit import default_timer as timer
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from yolo_court import YOLO

class Camera_cv(BaseCamera):
    video_source = 0
    model = None
    
    @staticmethod
    def set_video_source(source):
        Camera_cv.video_source = source

    @staticmethod
    def set_model(model):
        # import pdb;pdb.set_trace()
        Camera_cv.model = model

    @staticmethod
    def frames():
        # import pdb;pdb.set_trace()
        model = YOLO()
        # my_yolo=my_yolo(model)
        camera = cv2.VideoCapture(Camera_cv.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        # print("***test***")
        while True:
            # read current frame
            return_value, frame = camera.read()
            
            image0 = Image.fromarray(frame)
            
            image1 = model.detect_image(image0)
            result = np.asarray(image1[0])
            # if image1[1][0]!=0:
            #     import pdb;pdb.set_trace()
            #     result = np.asarray(image1)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
            yield cv2.imencode('.jpg', result)[1].tobytes()


class Video_cv():
    def __init__(self, source=None, model=0):
        self.source = source
        self.model = model

    def frames(self):
        camera = cv2.VideoCapture(self.source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()
            if self.model is not None:
                # print(Camera_cv.model)
                img, _ = self.model.detect_image(img)
                # encode as a jpeg image and return it
                yield cv2.imencode('.jpg', img)[1].tobytes()
            else:
                yield cv2.imencode('.jpg', img)[1].tobytes()
