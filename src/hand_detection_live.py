import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import os
import cv2
import time

from helpers import draw_landmarks_on_image

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

# Global variable to store the latest frame with landmarks
annotated_image = None

# Define the result callback function
def result_callback(result, image, timestamp_ms):
    global annotated_image
    annotated_image = draw_landmarks_on_image(image.numpy_view(), result)

# Specify model path
path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(path, 'hand_landmarker.task')

# Open the default camera
cam = cv2.VideoCapture(0)

# Initialize MediaPipe Vision task options with LIVE_STREAM mode
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create the hand landmarker instance with LIVE_STREAM mode and callback
options = HandLandmarkerOptions(
    num_hands=2,
    min_hand_presence_confidence=0.7,
    min_hand_detection_confidence=0.5,
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback)  # Pass the callback here

detector = vision.HandLandmarker.create_from_options(options)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Get the current timestamp in milliseconds
    timestamp = int(time.time() * 1000)

    # Send the image to the detector asynchronously
    detector.detect_async(mp_image, timestamp)

    # If annotated_image is updated by the callback, display it
    if annotated_image is not None:
        # Convert annotated image back to BGR for OpenCV display
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Hand Detection', annotated_image_bgr)
    else :
        cv2.imshow('Hand Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cam.release()
cv2.destroyAllWindows()
