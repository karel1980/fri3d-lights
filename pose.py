import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions

import cv2
import time

model_path = './pose_landmarker_lite.task'

import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


from mediapipe.framework.formats import landmark_pb2
import numpy as np


class Pose:
    def __init__(self, headless=True, callback = None, num_poses = 1):
        self.headless = headless
        self.callback = callback
        self.img = None
        self.num_poses = num_poses
        self.cap = cv2.VideoCapture(1)
        #self.cap = cv2.VideoCapture("twopeople.mp4")

        self.last_detection = None

    def handle_detection_result(self, detection_result, output_image, timestamp_ms):
        frame = output_image.numpy_view()
        h,w,_ = frame.shape

        self.last_detection = (detection_result, output_image, timestamp_ms)

    def run(self):
        options = PoseLandmarkerOptions(
            num_poses=self.num_poses,
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.handle_detection_result)

        with PoseLandmarker.create_from_options(options) as landmarker:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("no image from camera")
                    break
                
                print(frame.shape)
                print(frame.dtype)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                landmarker.detect_async(mp_image, int(time.time()*1000))

                ld = self.last_detection
                if ld is not None:
                    if self.callback != None:
                        self.callback(*(self.last_detection))
                    
                frame = cv2.flip(frame, 1)
                cv2.imshow('Webcam', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

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


def show_detection_result(detection_result, output_image, timestamp_ms):
    img = output_image.numpy_view().copy()
    img = draw_landmarks_on_image(img, detection_result)

    cv2.imshow('detection', img)

if __name__=="__main__":
    headless = False
    Pose(headless, num_poses=2, callback = show_detection_result).run()
