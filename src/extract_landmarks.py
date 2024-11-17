import mediapipe as mp
from mediapipe.tasks.python import vision
import os
import cv2
import time
import numpy as np

from helpers import extract_hand

# Specify model path
path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(path, 'hand_landmarker.task')

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
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO)
detector = vision.HandLandmarker.create_from_options(options)

def draw_landmarks_on_black(background, hand_landmarks, width, height):
    """
    Draw landmarks on a black background.
    :param background: The black background.
    :param hand_landmarks: Hand landmarks data.
    :param width: Image width for scaling.
    :param height: Image height for scaling.
    """
    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(background, (x, y), 2, (255, 255, 255), -1)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cTime = int(time.time() * 1000)
    ms = cTime - pTime
    pTime = cTime

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect_for_video(mp_image, int(cTime))

    # Create a black background
    black_background = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Draw landmarks if detection is available
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            draw_landmarks_on_black(black_background, hand_landmarks, frame_width, frame_height)

    hands = extract_hand(black_background, detection_result)

    black_background = cv2.cvtColor(black_background, cv2.COLOR_BGR2GRAY)
    black_background = black_background / 255

    if len(hands) > 0:
        hands[0] = cv2.resize(hands[0], (64, 64))
        cv2.imshow("hand", hands[0])
    # Display black background with hand landmarks
    cv2.imshow('Hand Landmarks on Black Background', black_background)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
