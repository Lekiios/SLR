import mediapipe as mp
from mediapipe.tasks.python import vision
import os
import cv2
import time
import keras
import numpy as np

from helpers import draw_hand_detect_on_image as annotate, extract_hand

BOX = True
LANDMARKS = False

# Specify model path
path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(path, 'hand_landmarker.task')

model_alphabet = keras.models.load_model('../models/model_20.keras')

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

cTime = 0
pTime = 0

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    num_hands=1,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8,
    min_tracking_confidence=0.5,
    base_options=BaseOptions(model_asset_path='../models/hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO)
detector = vision.HandLandmarker.create_from_options(options)

prediction = [0]

while True:
    ret, frame = cam.read()

    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cTime = int(time.time() * 1000)
    ms = cTime - pTime
    pTime = cTime

    cv2.putText(rgb_frame,f'{str(int(ms))}', (5,15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect_for_video(mp_image, int(cTime))

    hand = extract_hand(mp_image.numpy_view(), detection_result)
    if len(hand) > 0:
        f_hand= cv2.cvtColor(hand[0], cv2.COLOR_RGB2GRAY)
        f_hand = cv2.resize(f_hand, (28, 28))
        f_hand = f_hand / 255.0
        f_hand = f_hand.reshape(1, 28, 28, 1)
        prediction = np.argmax(model_alphabet.predict(f_hand),axis=1)
        if prediction[0] >= 9:
            prediction[0] += 1
        print(prediction[0])

    annotated_image = annotate(mp_image.numpy_view(), detection_result, prediction[0], no_landmarks=not LANDMARKS, no_box=not BOX)

    # Convert display image back to BGR
    bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # Display the captured frame
    cv2.imshow('Camera', bgr_annotated_image)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()

