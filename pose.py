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


class Main:
    def __init__(self, headless=True):
        self.headless = headless
        self.img = None

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
        frame = output_image.numpy_view()
        h,w,_ = frame.shape

        for person in detection_result.pose_landmarks:
            nose = person[0]
            left_wrist = person[1]
            right_wrist = person[2]

            if not self.headless:
                x, y = int(nose.x*w), int(nose.y*h)
                start_point = (20,20)
                end_point = (200,200)
                color = (255,0,0)
                thickness = 5

                self.img = self.draw_landmarks_on_image(frame, detection_result)
                cv2.circle(self.img, (x,y), 50, (255, 255, 255), 10)
                #cv2.circle(img, (50, 50), 5, (0, 255, 0), -1)

                # segmentation
                #segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
                #visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
                #cv2.imshow(visualized_mask)


    def run(self):
        options = PoseLandmarkerOptions(
            num_poses=2,
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.handle_detection_result)


        with PoseLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(0)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
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


if __name__=="__main__":
    headless = True
    Main(headless).run()
