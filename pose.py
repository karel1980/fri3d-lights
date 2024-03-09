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
    def __init__(self, headless=True, callback = None):
        self.headless = headless
        self.callback = callback
        self.img = None
        self.cap = cv2.VideoCapture(0)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
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


    def handle_detection_result(self, detection_result, output_image, timestamp_ms):
        #print("timestamp", timestamp_ms)
        frame = output_image.numpy_view()
        h,w,_ = frame.shape

        if self.callback != None:
            self.callback(detection_result)



    def run(self):
        options = PoseLandmarkerOptions(
            num_poses=1,
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.handle_detection_result)


        with PoseLandmarker.create_from_options(options) as landmarker:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("no image from camera")
                    break

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                landmarker.detect_async(mp_image, int(time.time()*1000))

                if not self.headless:
                    #print("FRAME", frame.shape, frame.dtype)
                    #cv2.imshow('Webcam', frame)
                    if self.img is not None:
                        cv2.imshow('img', self.img)

                    # Wait for a key press for 1 millisecond
                    # If 'q' is pressed, exit the loop
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break



def show_detection_result(detection_result):
    for person in detection_result.pose_landmarks:
        nose = person[0]
        left_wrist = person[1]
        right_wrist = person[2]
        nose_val = 50 - int(nose.x * 50)
        if nose_val < 0: nose_val = 0
        if nose_val > 50: nose_val = 50

        print((" " * nose_val) + "*" + ((51 - nose_val) * " "), end="\r")

if __name__=="__main__":
    headless = True
    Pose(headless, callback = show_detection_result).run()
