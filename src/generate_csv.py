import os
import pandas as pd
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from multiprocessing import Pool, cpu_count
import numpy as np

path = os.path.abspath('..')

train_path = os.path.join(path, 'dataset', 'ASL_Dataset', 'Train')
test_path = os.path.join(path, 'dataset', 'ASL_Dataset', 'Test')
model_path = os.path.join(path, 'models' ,'hand_landmarker.task')

# Label map
label_map = {chr(i): i - ord('A') for i in range(ord('A'), ord('Z') + 1)}
label_map['Nothing'] = 26
label_map['Space'] = 27

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
    base_options=BaseOptions(model_asset_path='../models/hand_landmarker.task'),running_mode=VisionRunningMode.IMAGE)

# Function to process a single image
def process_image(task):
    label, image_path = task
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        landmark_list = []
        for lm in landmarks:
            landmark_list.extend([lm.x, lm.y])
        return [label] + landmark_list
    else:
        #print(f"No hand landmarks detected in {image_path}")
        # Append NaN or zeros for missing landmarks
        num_landmarks = 21
        return [label] + [np.nan] * (num_landmarks * 2)

def initialize_detector():
    global detector
    detector = vision.HandLandmarker.create_from_options(options)

def landmarks_to_csv(input, output):
    # Collect all image paths and labels
    tasks = []
    for label_folder in sorted(os.listdir(input)):
        folder_path = os.path.join(input, label_folder)
        if os.path.isdir(folder_path) and label_folder in label_map:
            label = label_map[label_folder]
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
                tasks.append((label, image_path))

    print(f"Processing {len(tasks)} images...")

    # Use multiprocessing pool to process images in parallel
    if __name__ == '__main__':
        with Pool(processes=cpu_count(), initializer=initialize_detector) as pool:
            results = pool.map(process_image, tasks)

    # Filter out None results and prepare for CSV
    data = [result for result in results if result is not None]

    # Define column names
    num_landmarks = 21  # MediaPipe detects 21 landmarks per hand
    columns = ['Label'] + [f'X{i+1}' for i in range(num_landmarks)] + [f'Y{i+1}' for i in range(num_landmarks)]

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output, index=False)

    print("CSV file with hand landmarks created successfully!")


landmarks_to_csv(train_path, '../dataset/train_landmarks.csv')
landmarks_to_csv(test_path, '../dataset/test_landmarks.csv')