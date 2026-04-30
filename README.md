# MediaPipe Vision Demos (Python)

This repository contains beginner-friendly MediaPipe Tasks API demos for:
- Object detection
- Image classification
- Hand landmark detection

## Requirements

- Python 3.8+
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

## Included Scripts

### 1) Object Detection

Script: `object_detection.py`

Runs object detection on an image and saves an annotated result with bounding boxes.

Run:

```bash
python object_detection.py
```

Output:
- `data/output.jpg`

### 2) Image Classification

Script: `image_classification.py`

Supports batch classification and optional single-image mode.

Run batch examples:

```bash
python image_classification.py --no-display
```

Run single image:

```bash
python image_classification.py --single-image "data/sample_image.jpg" --no-display
```

### 3) Hand Landmark Detection

Script: `hand_landmark_detection.py`

Uses MediaPipe Hand Landmarker in IMAGE mode, prints fingertip coordinates `(x, y, z)`, and saves image with landmarks/connections.

Run:

```bash
python hand_landmark_detection.py
```

Default paths used by the script:
- Model: `models/hand_landmarker.task`
- Image: `data/hand-357335_640.jpg`
- Output: `data/hand_landmarks_output.jpg`

## Project Structure

```text
inClassMediaPipe/
├── object_detection.py
├── image_classification.py
├── hand_landmark_detection.py
├── requirements.txt
├── README.md
├── HAND_LANDMARK_QUICKSTART.md
├── IMAGE_CLASSIFICATION_QUICKSTART.md
├── IMAGE_CLASSIFICATION_README.md
└── data/
```

## Notes

- The first run may be slower due to model downloads.
- On headless systems, image display windows may be skipped, but output files are still saved.
- MediaPipe warnings from TensorFlow Lite delegates can appear in console and are usually non-fatal.

## References

- [MediaPipe Overview](https://ai.google.dev/edge/mediapipe)
- [Object Detector (Python)](https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/python)
- [Image Classifier (Python)](https://ai.google.dev/edge/mediapipe/solutions/vision/image_classifier/python)
- [Hand Landmarker (Python)](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
