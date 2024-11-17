import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python import solutions
import cv2

# Constants
MARGIN = 40  # Margin around the bounding box
FONT_SIZE = 1
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (0, 255, 0)  # Green
BOUNDING_BOX_COLOR = (0, 0, 255)  # Red
BOUNDING_BOX_THICKNESS = 2

def get_letter(letter_id):
    """
    Get the letter corresponding to the id.
    :param letter_id: The id of the letter.
    :return: The letter.
    """
    return chr(letter_id + 65)

def draw_hand_detect_on_image(rgb_image, detection_result, letter, no_landmarks=False, no_box=False):
    """
    Draw landmarks, handedness, and bounding boxes on the image.
    :param rgb_image: The input RGB image.
    :param detection_result: Detection result containing hand landmarks and handedness.
    :param letter: The letter to display on the image.
    :param no_landmarks: If True, don't draw landmarks.
    :param no_box: If True, don't draw bounding boxes.
    :return: Annotated image.
    """

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        if not no_landmarks:
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())


        # Get the bounding box for the hand.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]

        x_min = int(min(x_coordinates) * width) - MARGIN
        y_min = int(min(y_coordinates) * height) - MARGIN
        x_max = int(max(x_coordinates) * width) + MARGIN
        y_max = int(max(y_coordinates) * height) + MARGIN

        # Ensure bounding box is within image bounds
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(width, x_max), min(height, y_max)

        if not no_box:
            # Draw the bounding box
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), BOUNDING_BOX_COLOR, BOUNDING_BOX_THICKNESS)

        # Get the top-left corner of the bounding box for text.
        text_x = x_min
        text_y = y_min - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}-{get_letter(letter)}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

def extract_hand(rgb_image, detection_result):
    """
    Extract the hand from the image.
    :param rgb_image: The input RGB image.
    :param detection_result: Detection result containing hand landmarks and handedness.
    :return: Annotated image.
    """

    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    hand = []

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # Get the bounding box for the hand.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]

        x_min = int(min(x_coordinates) * width) - MARGIN
        y_min = int(min(y_coordinates) * height) - MARGIN
        x_max = int(max(x_coordinates) * width) + MARGIN
        y_max = int(max(y_coordinates) * height) + MARGIN

        # Ensure bounding box is within image bounds
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(width, x_max), min(height, y_max)

        # Extract the hand from the image
        hand.append(annotated_image[y_min:y_max, x_min:x_max])

    return hand
