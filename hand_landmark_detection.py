"""
Hand Landmark Detection (MediaPipe Tasks API, Image Mode)

Install:
    python -m pip install mediapipe

Run:
    python hand_landmark_detection.py
"""

import os

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Example path style from official docs:
# model_path = '/absolute/path/to/gesture_recognizer.task'
# For this project we use:
MODEL_PATH = "models/hand_landmarker.task"
IMAGE_PATH = "data/hand-357335_640.jpg"
OUTPUT_PATH = "data/hand_landmarks_output.jpg"


def draw_landmarks(image_bgr, hand_landmarker_result):
    """Draw landmarks and hand connections on the image."""
    output = image_bgr.copy()
    height, width, _ = output.shape
    connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    for hand_landmarks in hand_landmarker_result.hand_landmarks:
        # Draw hand skeleton connections.
        for connection in connections:
            start = hand_landmarks[connection.start]
            end = hand_landmarks[connection.end]
            x1, y1 = int(start.x * width), int(start.y * height)
            x2, y2 = int(end.x * width), int(end.y * height)
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw each landmark point.
        for lm in hand_landmarks:
            x, y = int(lm.x * width), int(lm.y * height)
            cv2.circle(output, (x, y), 3, (0, 0, 255), -1)

    return output


def print_fingertips(hand_landmarker_result):
    """Print fingertip coordinates (x, y, z) for each detected hand."""
    fingertip_indices = {
        "THUMB_TIP": 4,
        "INDEX_TIP": 8,
        "MIDDLE_TIP": 12,
        "RING_TIP": 16,
        "PINKY_TIP": 20,
    }

    if not hand_landmarker_result.hand_landmarks:
        print("No hands detected.")
        return

    print("\nFingertip coordinates (x, y, z):")
    for i, hand_landmarks in enumerate(hand_landmarker_result.hand_landmarks, start=1):
        print(f"\nHand #{i}:")
        for name, index in fingertip_indices.items():
            lm = hand_landmarks[index]
            print(f"  {name}: x={lm.x:.4f}, y={lm.y:.4f}, z={lm.z:.4f}")


def main():
    # 1) Validate input/model paths.
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    # 2) Official-style aliases from docs.
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # 3) Create hand landmarker in IMAGE mode.
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
    )

    # 4) Load input image using MediaPipe image API.
    mp_image = mp.Image.create_from_file(IMAGE_PATH)

    # 5) Run detection.
    with HandLandmarker.create_from_options(options) as landmarker:
        hand_landmarker_result = landmarker.detect(mp_image)

    # 6) Print fingertip coordinates.
    print_fingertips(hand_landmarker_result)

    # 7) Draw landmarks on image and save output.
    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image with OpenCV: {IMAGE_PATH}")

    annotated = draw_landmarks(image_bgr, hand_landmarker_result)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, annotated)
    print(f"\nSaved output to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
