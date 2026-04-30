# Hand Landmark Detection Quickstart

This quickstart shows how to run `hand_landmark_detection.py` with MediaPipe Tasks API.

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Run with defaults

```bash
python hand_landmark_detection.py
```

Default paths used by the script:
- Model: `models/hand_landmarker.task`
- Input image: `data/hand_sample.jpg` (or local fallback image)
- Output image: `data/hand_landmarks_output.jpg`

## 3) Run with your own image

```bash
python hand_landmark_detection.py --image "C:\path\to\your\hand_photo.jpg"
```

Optional flags:
- `--model "models/hand_landmarker.task"` custom model path
- `--output "data/my_output.jpg"` custom output path
- `--no-display` skip OpenCV window display

## 4) Example command

```bash
python hand_landmark_detection.py --image "data/hand_sample.jpg" --no-display
```

## 5) Expected output

The script will:
- initialize the hand landmarker model
- run hand landmark detection on the image
- print fingertip coordinates `(x, y, z)` if hands are found
- save annotated output image with landmarks/connections

Output image location:
- `data/hand_landmarks_output.jpg`
