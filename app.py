#!/usr/bin/env python
from flask import Flask, render_template, Response, request, redirect, url_for
from camera import Camera
from camera_opencv import Camera_cv
from werkzeug.utils import secure_filename
import os
import sys
from yolo_court import YOLO

# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera

app = Flask(__name__)

UPLOAD_FOLDER = './cache/upload'
ALLOWED_EXTENSIONS = set(['mp4', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def gen(camera, start=False):
    """Video streaming generator function."""

    while start:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/cv_camera')
def cv_camera_feed():
    
    print("**begin**")
    camera = Camera_cv()
    return Response(
        gen(camera, True), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True,debug=True)
