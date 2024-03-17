import cv2
import io
import socketserver
from http import server
import threading
import numpy as np
import random

import colorsys
from ledstrip import LedStrip
from tracker import Tracker

import time
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions

class Plugin:
    def __init__(self):
        self.name = "null"

    def stop(self):
        pass

    def process(self, frame, context):
        # In this method you can analyze the frame, but don't manipulate the frame here
        pass

    def postprocess(self, frame, context):
        # Here is where you manipulate the frame
        return frame

class Person:
    def __init__(self, p_id, landmarks, color=None, missing=0):
        self.p_id = p_id
        self.landmarks = landmarks
        self.color = random_color() if color is None else color
        self.missing = missing

    def __repr__(self):
        return f"Person {self.p_id} at {self.landmarks[0].x} ({self.missing})"

    def __str__(self):
        return f"Person {self.p_id} at {self.landmarks[0].x} ({self.missing})"


def random_color():
    hsv_color = (random.random(), 1.0, 1.0)
    return [ int(x*256) for x in colorsys.hsv_to_rgb(*hsv_color) ]

class PoseDetectorPlugin:
    def __init__(self):
        self.name = "posedetector"
        self.num_poses = 1

        model_path = 'pose_landmarker_lite.task'
        options = PoseLandmarkerOptions(
            num_poses=self.num_poses,
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            result_callback=self.detection_callback)

        self.landmarker = PoseLandmarker.create_from_options(options)

        self.people = {}
        self.tracker = Tracker(self.distance_metric)

    def distance_metric(self, a, b):
        return np.linalg.norm(a[0].x - b[0].x)

    def stop(self):
        pass

    def process(self, frame, context):
        context["people"] = [p for p in self.people.values()]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.landmarker.detect_async(mp_image, int(time.time()*1000))

    def detection_callback(self, detection_result, frame, foo):
        last_track_result = self.tracker.update(detection_result.pose_landmarks)
        tracked, disappeared = last_track_result

        removable = set(self.people.keys()) - set(tracked.keys())
        for k in removable:
            del self.people[k]

        for k,v in tracked.items():
            if k not in self.people:
                self.people[k] = Person(k, v)
            else:
                self.people[k].landmarks = v

        for k,v in disappeared.items():
            if k in self.people:
                self.people[k].missing = v


    def postprocess(self, frame, context):
        return frame

class ObjectDetectorPlugin:
    def __init__(self):
        self.name = "objectdetector"
        self.num_poses = 1

        model_path = 'efficientdet_lite0.tflite'

        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            max_results=5,
            result_callback=self.detection_callback,
            category_allowlist=["person"],
            score_threshold = 0.3)

        self.detector = ObjectDetector.create_from_options(options)

        self.objects = None
        #self.tracker = Tracker(self.distance_metric)

    def distance_metric(self, a, b):
        return np.linalg.norm(a[0].x - b[0].x)

    def stop(self):
        pass

    def process(self, frame, context):
        context["objects"] = self.objects

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.detector.detect_async(mp_image, int(time.time()*1000))

    def detection_callback(self, detection_result, frame, foo):
        self.objects = detection_result
#        last_track_result = self.tracker.update(detection_result.pose_landmarks)
#        tracked, disappeared = last_track_result
#
#        removable = set(self.people.keys()) - set(tracked.keys())
#        for k in removable:
#            del self.people[k]
#
#        for k,v in tracked.items():
#            if k not in self.people:
#                self.people[k] = Person(k, v)
#            else:
#                self.people[k].landmarks = v
#
#        for k,v in disappeared.items():
#            if k in self.people:
#                self.people[k].missing = v


    def postprocess(self, frame, context):
        return frame


class ObjectToBlobCalculatorPlugin:
    def process(self, frame, context):
        objects = context.get("objects", None)
        if objects is None:
            return

        blobs = []
        for d in objects.detections:
            bb = d.bounding_box
            mu = (bb.origin_x + bb.width / 2) / frame.shape[1]
            sigma = 0.1
            blobs.append((mu, sigma, (0,255,0)))
        
        context["blobs"] = blobs

    def postprocess(self, frame, context):
        pass


class PeopleToBlobCalculatorPlugin:
    def process(self, frame, context):
        if "people" not in context:
            return

        blobs = []
        for person in context["people"]:
            blobs.append((person.landmarks[0].x, 0.1, person.color))
        
        context["blobs"] = blobs

    def postprocess(self, frame, context):
        pass


class LedOutputPlugin:
    def __init__(self, ledstrip):
        self.name = "ledoutput"
        self.ledstrip = LedStrip(60) if ledstrip is None else ledstrip
        self.num_leds = ledstrip.num_leds

    def stop(self):
        pass

    def process(self, frame, context):
        blobs = context.get("blobs", [])

        if blobs:
            led_values = self.calculate_led_colors(context["blobs"])
            self.ledstrip.set_array(led_values)

    def calculate_led_colors(self, blobs):
        led_values = np.zeros((self.num_leds, 3), np.uint8)
        for blob in blobs:
            mu, sigma, color = blob
            sigma /= 5

            intensity = bell_curve(np.linspace(1.0, 0.0, self.num_leds), mu, sigma)
            colors = (intensity[:,np.newaxis] * np.array(color)).astype(np.uint8)
            led_values = np.maximum(led_values, colors)

        return led_values

    def postprocess(self, frame,context):
        pass


class StreamingOuput:
    def __init__(self):
        self.condition = threading.Condition()
        self.frame = frame


class PiCamera:
    def __init__(self):
        import picamera2
        TUNING_FILES = [
            "/usr/share/libcamera/ipa/rpi/vc4/ov5647.json",
            "/usr/share/libcamera/ipa/rpi/vc4/ov5647_noir.json",
            "/usr/share/libcamera/ipa/rpi/pisp/ov5647.json",
            "/usr/share/libcamera/ipa/rpi/pisp/ov5647_noir.json",
        ]
        
        tuning = picamera2.Picamera2.load_tuning_file(TUNING_FILES[1])
        picam2 = picamera2.Picamera2(tuning=tuning)
        config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
      
        self.camera = picam2

    def get_frame(self):
        return self.camera.capture_array()

    def release(self):
        self.camera.stop_recording()


class CV2Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def get_frame(self):
        ret, frame = self.cap.read()

        cv2.waitKey(1) # TODO: strigger stop if 'q' is pressed?
        return frame

    def release(self):
        self.cap.release()

class CV2VideoSource:
    def __init__(self, file, loop=True):
        self.cap = cv2.VideoCapture(file)
        self.file = file
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            # Seeking didn't seem to go well, so just creating a new capture instance
            self.cap.release()
            self.cap = cv2.VideoCapture(self.file)
            ret,frame = self.cap.read()

        delay_time = int(1000 / self.fps)
        cv2.waitKey(delay_time)
        return frame

    def release(self):
        self.cap.release()


class Application:
    def __init__(self, camera):
        self.camera = camera
        self.output = StreamingOutput()
        self.plugins = []

        address = ('', 8000)
        self.server = StreamingServer(address, lambda *args, **kwargs: StreamingHandler(self.output, *args, **kwargs))

    def register_plugin(self, plugin):
        self.plugins.append(plugin)

    def start(self):
        self.running = True
        self.start_camera_thread()
        try:
            self.server.serve_forever()
        finally:
            self.running = False
            self.camera.release()
            pass

    def start_camera_thread(self):
        thread = threading.Thread(target = self.camera_loop)
        thread.start()

    def camera_loop(self):
        while self.running:
            frame = self.camera.get_frame()
            if frame is None:
                print("No frame.")
                time.sleep(1)
                continue
            context = dict()
            for plugin in self.plugins:
                plugin.process(frame, context)

            for plugin in self.plugins:
                plugin.postprocess(frame, context)

            self.output.write(frame)


class StickFigurePlugin:
    def __init__(self):
        pass

    def stop(self):
        pass

    def process(self, frame, context):
        pass
    
    def postprocess(self, frame, context):
        if "people" not in context:
            return
        pose_landmarks_list = [ p.landmarks for p in context["people"] ]
        annotated_image = frame

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
          pose_landmarks = pose_landmarks_list[idx]

          # Draw the pose landmarks.
          pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
          pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
          ])
          solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image

class DrawBoundingBoxPlugin:
    def __init__(self):
        pass

    def stop(self):
        pass

    def process(self, frame, context):
        pass
    
    def postprocess(self, frame, context):
        if "objects" not in context:
            return
        objects = context["objects"]
        if objects is None:
            return

        for d in objects.detections:
            bb = d.bounding_box
            topleft = (bb.origin_x, bb.origin_y)
            bottomright = (bb.origin_x + bb.width, bb.origin_y + bb.height)
            cv2.rectangle(frame, topleft, bottomright, (0,255,255), 5)

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1
            color = (0, 255, 255)
            thickness = 2
            position = (bb.origin_x + 20, bb.origin_y)
            cv2.putText(frame, d.categories[0].category_name, position, font, scale, color, thickness)

class StdoutPlugin:
    def __init__(self):
        pass

    def process(self, frame, context):
        #print(frame.shape)
        print(context)

    def postprocess(self, frame, context):
        pass

class LedMonitorPlugin:
    def __init__(self):
        pass

    def stop(self):
        pass

    def process(self, frame, context):
        pass
    
    def postprocess(self, frame, context):
        blobs = context.get("blobs", [])

        if not blobs:
            return

        h,w,c = frame.shape

        monitor_line = np.zeros((w, c), np.uint8)
        for mu,sigma,color in context["blobs"]:
            color = color if c == 3 else [*color, 255]
            intensity = bell_curve(np.linspace(0.0, 1.0, w), mu, sigma)
            channels = np.array(color).astype(np.uint8)
            layer = (intensity[:, np.newaxis] * channels).astype(np.uint8)
            monitor_line = np.maximum(monitor_line, layer)

        monitor_height = 5
        monitor = np.tile(monitor_line[np.newaxis, :, :], (monitor_height, 1, 1))
        monitor = monitor_line[np.newaxis,:,:]
        frame[-monitor_height:, :, :] = monitor
            

def bell_curve(x, mean, std_dev):
    return np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))

def scaled_bell_curve(x, mean, std_dev):
    return bell_curve(x, mean, std_dev) / (std_dev * np.sqrt(2 * np.pi))


PAGE = """\
<html>
<head>
<title>Camera monitor</title>
</head>
<body>
<h1>Camera monitor</h1>
<img src="stream.mjpg" width="640" height="480" />
</body>
</html>
"""


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

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
                        ret, buf = cv2.imencode('.jpg', self.output.frame)
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(buf))
                    self.end_headers()
                    self.wfile.write(buf.tobytes())
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


def main():
    camera = PiCamera()
    #camera = CV2Camera()
    #camera = CV2VideoSource('experiments/output.avi')
    #camera = CV2VideoSource('experiments/lights-on.h264')

    app = Application(camera)

    #app.register_plugin(PoseDetectorPlugin())
    #app.register_plugin(PeopleToBlobCalculatorPlugin())

    app.register_plugin(ObjectDetectorPlugin())
    app.register_plugin(ObjectToBlobCalculatorPlugin())

    #app.register_plugin(StdoutPlugin())

    try:
        app.register_plugin(LedOutputPlugin(LedStrip(60)))
    except NameError:
        print("OOPS NO LED STRIP. Try running as root")

    app.register_plugin(LedMonitorPlugin())
    #app.register_plugin(StickFigurePlugin())
    app.register_plugin(DrawBoundingBoxPlugin())

    app.start()


if __name__ == "__main__":
    main()
