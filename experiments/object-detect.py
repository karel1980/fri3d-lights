import cv2
import mediapipe as mp
import time


BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

detections = []

def handle_object_detections(result, output_image: mp.Image, timestamp_ms: int):
    global detections
    detections = result.detections


def main():
    global detections
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the webcam.")
        return

    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path='efficientdet_lite2.tflite'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        max_results=5,
        result_callback=handle_object_detections,
        category_allowlist = ["potted_plant"],
        score_threshold = 0.5)

    with ObjectDetector.create_from_options(options) as detector:
        # Loop to capture frames continuously
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Check if the frame is captured successfully
            if not ret:
                print("Error: Couldn't capture frame.")
                break

            print(frame.shape)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            detector.detect_async(mp_image, int(time.time_ns() / 1_000_000))
        
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2

            for d in detections:
                bbox = d.bounding_box
                origin = (bbox.origin_x, bbox.origin_y)
                other = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
                category = d.categories[0].category_name
                cv2.rectangle(frame, origin, other, (255,255,0), 5)
                #text_size = cv2.getTextSize(category, font, font_scale, thickness)[0]
                cv2.putText(frame, category, origin, font, font_scale, (255,255,255), 5)

            # Display the frame
            cv2.imshow('Webcam Streaming', frame)

            # Check for key press, if 'q' is pressed, exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

