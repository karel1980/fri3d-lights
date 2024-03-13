#!/usr/bin/python3

# Mostly copied from https://picamera.readthedocs.io/en/release-1.13/recipes2.html
# Run this script, then point a web browser at http:<this-ip-address>:8000
# Note: needs simplejpeg to be installed (pip3 install simplejpeg).

import io
import logging
import socketserver
from http import server
import threading
from threading import Condition
import cv2
import time

import mediapipe as mp

from picamera2 import MappedArray, Picamera2, Preview
from picamera2.encoders import JpegEncoder, H264Encoder

from picamera2.outputs import FileOutput

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


PAGE = """\
<html>
<head>
<title>picamera2 MJPEG streaming demo</title>
</head>
<body>
<h1>Picamera2 MJPEG Streaming Demo</h1>
<img src="stream.mjpg" width="640" height="480" />
</body>
</html>
"""


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


class StreamingHandler(server.BaseHTTPRequestHandler):
    def __init__(self, output, *args, **kwargs):
        self.output = output
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with self.output.condition:
                        self.output.condition.wait()
                        frame = self.output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

TUNING_FILES = [
    "/usr/share/libcamera/ipa/rpi/vc4/ov5647.json",
    "/usr/share/libcamera/ipa/rpi/vc4/ov5647_noir.json",
    "/usr/share/libcamera/ipa/rpi/pisp/ov5647.json",
    "/usr/share/libcamera/ipa/rpi/pisp/ov5647_noir.json",
]

def create_camera(output, post_callback):
    tuning = Picamera2.load_tuning_file(TUNING_FILES[1])
    picam2 = Picamera2(tuning=tuning)

    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
    picam2.post_callback = post_callback

    picam2.start_recording(JpegEncoder(), FileOutput(output))

    return picam2


class PoseDetector:
    def __init__(self, picam2, detection_callback, num_poses = 1):
        self.picam2 = picam2
        self.detection_callback = detection_callback
        self.num_poses = num_poses


    def start(self):
        self.running = True
        detection_thread = threading.Thread(target=self.detect_forever)
        detection_thread.start()

    def detect_forever(self):
        model_path = './pose_landmarker_lite.task'

        options = PoseLandmarkerOptions(
            num_poses=self.num_poses,
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.detection_callback)

        with PoseLandmarker.create_from_options(options) as landmarker:
            while self.running:
                frame = self.picam2.capture_array("main")
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                landmarker.detect_async(mp_image, int(time.time()*1000))

    def stop(self):
        self.running = False
                

class Main:
    blobs = []

    def __init__(self):
        output = StreamingOutput()
        picam2 = create_camera(output, self.draw_faces)

        self.detector = PoseDetector(picam2, self.handle_pose_detection_result)

        address = ('', 8000)
        self.server = StreamingServer(address, lambda *args, **kwargs: StreamingHandler(output, *args, **kwargs))

        self.blobs = []

    def start(self):
        self.start_detector()
        self.start_webserver()

    def start_detector(self):
        self.detector.start()

    def start_webserver(self):
        try:
            self.server.serve_forever()
        finally:
            self.detector.stop()
            picam2.stop_recording()

    def handle_pose_detection_result(self, detection_result, image, foo):
        self.blobs = [ (10, 10, 100, 100) ]

    def draw_faces(self, request):
        with MappedArray(request, "main") as m:
            cv2.rectangle(m.array, (50, 50), (200, 100), (0, 255, 0, 0), thickness = 5)
            for blob in self.blobs:
                cv2.rectangle(m.array, (blob[0], blob[1]), (blob[2], blob[3]), (0, 255, 0, 0), thickness = 5)

def main():
    Main().start()


if __name__=="__main__":
    main()

