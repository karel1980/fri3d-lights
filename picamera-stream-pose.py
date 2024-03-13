#!/usr/bin/python3

# Mostly copied from https://picamera.readthedocs.io/en/release-1.13/recipes2.html
# Run this script, then point a web browser at http:<this-ip-address>:8000
# Note: needs simplejpeg to be installed (pip3 install simplejpeg).

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

faces = []
def draw_visualizations(request):
    print("REQUEST", request)
    with MappedArray(request, "main") as m:
        #print("drawing with cv2")
        for f in faces:
            (x, y, w, h) = [c * n // d for c, n, d in zip(f, (w0, h0) * 2, (w1, h1) * 2)]
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0))



TUNING_FILES = [
    "/usr/share/libcamera/ipa/rpi/vc4/ov5647.json",
    "/usr/share/libcamera/ipa/rpi/vc4/ov5647_noir.json",
    "/usr/share/libcamera/ipa/rpi/pisp/ov5647.json",
    "/usr/share/libcamera/ipa/rpi/pisp/ov5647_noir.json",
]

tuning = Picamera2.load_tuning_file(TUNING_FILES[1])
picam2 = Picamera2(tuning=tuning)

picam2.configure(picam2.create_video_configuration(
    main={"size": (640, 480)},
    lores={"size": (320, 240)},
))

(w0, h0) = picam2.stream_configuration("main")["size"]
(w1, h1) = picam2.stream_configuration("lores")["size"]
s1 = picam2.stream_configuration("lores")["stride"]

picam2.post_callback = draw_visualizations
output = StreamingOutput()

picam2.start_recording(JpegEncoder(), FileOutput(output))
#picam2.start_recording(H264Encoder(), FileOutput(output))

def handle_detection_result(detection_result, img, foo):
    print("todo: handle detection result")

model_path = './pose_landmarker_lite.task'
options = PoseLandmarkerOptions(
    num_poses=1,
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handle_detection_result)

def detect_poses():
    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            buffer = picam2.capture_array("main")
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=buffer)
            landmarker.detect_async(mp_image, int(time.time()*1000))

import threading
# Run face detection in a separate thread
detection_thread = threading.Thread(target=detect_poses)
detection_thread.start()

try:
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    server.serve_forever()
finally:
    picam2.stop_recording()
