#!/usr/bin/python3

from dataclasses import dataclass
import threading

import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

import time
import io
import logging
import socketserver
from http import server
from threading import Condition
import cv2

from picamera2 import MappedArray, Picamera2, Preview
from picamera2.encoders import JpegEncoder, H264Encoder

from picamera2.outputs import FileOutput

@dataclass
class Rectangle:
    x: int
    y: int
    width: int
    height: int


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
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
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

detections = dict(rectangles = [ Rectangle(10, 10, 200, 150) ])

def draw_visualizations(request):
    with MappedArray(request, "main") as m:
        for rectangle in detections["rectangles"]:
            # TODO: draw blobs on the image (create the overlay in other thread?)
            cv2.rectangle(m.array, (rectangle.x, rectangle.y), (rectangle.x + rectangle.width, rectangle.y + rectangle.height), (0, 255, 0, 0))

TUNING_FILES = [
    "/usr/share/libcamera/ipa/rpi/vc4/ov5647.json",
    "/usr/share/libcamera/ipa/rpi/vc4/ov5647_noir.json",
    "/usr/share/libcamera/ipa/rpi/pisp/ov5647.json",
    "/usr/share/libcamera/ipa/rpi/pisp/ov5647_noir.json",
]

class Application:
    def __init__(self, detection_callback, render_callback):
        self.detection_callback = detection_callback
        self.render_callback = render_callback

        self.output = StreamingOutput()

        tuning = Picamera2.load_tuning_file(TUNING_FILES[1])
        self.picam2 = Picamera2(tuning=tuning)

        picam2.configure(picam2.create_video_configuration(
            main={"size": (640, 480)},
            lores={"size": (320, 240)},
        ))

        self.main_size = picam2.stream_configuration("main")["size"]
        self.lores_size = picam2.stream_configuration("lores")["size"]
        self.lores_stride = picam2.stream_configuration("lores")["stride"]

        self.picam2.post_callback = self.render_callback
        self.picam2.start_recording(JpegEncoder(), FileOutput(self.output))

    def start(self):
        # Start detection thread
        detection_thread = threading.Thread(target=self.detection_callback)
        detection_thread.start()

        # TODO: start led controlling thread

        # Start web server
        self.start_server()

    def detect_poses(self):
        model_path = './pose_landmarker_lite.task'
        options = PoseLandmarkerOptions(
            num_poses=1,
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.detection_callback)

        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                buffer = self.picam2.capture_array("main")
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=buffer)
                landmarker.detect_async(mp_image, int(time.time()*1000))

    def handle_detection_result(detection_result, img, foo):
        result = dict()
        result["detection_result"] = detection_result
        result["rectangles"] = [
                Rectangle(10, 10, 200, 150)
        ]

    def start_server(self):
        try:
            address = ('', 8000)
            server = StreamingServer(address, StreamingHandler)
            server.serve_forever()
        finally:
            self.picam2.stop_recording()

