import time
from base_camera import BaseCamera


class Camera(BaseCamera):
    """An emulated camera implementation that streams a repeated sequence of
    files 1.jpg, 2.jpg and 3.jpg at a rate of one frame per second."""

    imgs = []
    model = None

    @staticmethod
    def set_source(source):
        Camera.imgs = [open(s + '.jpg', 'rb').read() for s in source]
        if Camera.model is not None:
            Camera.imgs = [Camera.model.detect_image(i) for i in Camera.imgs]

    @staticmethod
    def set_model(model):
        Camera.model = model

    @staticmethod
    def frames():
        while True:
            time.sleep(0.02)
            yield Camera.imgs[int(time.time()) % 3]
