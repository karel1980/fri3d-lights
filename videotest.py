import cv2

# Initialize VideoCapture object for webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    # Check if frame is successfully read
    if not ret:
        print("Error: Failed to read frame")
        break

    # Display the frame in a window
    cv2.imshow('Webcam', frame)

    # Wait for a key press for 1 millisecond
    # If 'q' is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

