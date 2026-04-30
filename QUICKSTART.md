# Quick Start Guide - Object Detection Demo

## Overview
✅ Successfully created a **MediaPipe Object Detection** demo on the `feature/object-detection` branch.

## What Was Created

### Files
- **`object_detection.py`** - Main detection script (15 KB)
- **`requirements.txt`** - Python dependencies
- **`README.md`** - Comprehensive documentation
- **`.gitignore`** - Git exclusions for large files
- **`data/`** directory - Models and images (auto-downloaded)

### Key Features
✅ Pretrained MediaPipe object detector (EfficientDet-Lite0)  
✅ Automatic model and sample image download  
✅ Full inference pipeline with 7 clear steps  
✅ Bounding box visualization with labels  
✅ Confidence score reporting  
✅ Well-documented, beginner-friendly code  
✅ Runs successfully on Windows  

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Detection Script
```bash
python object_detection.py
```

### Expected Output
The script will:
1. ✓ Download the model (~14 MB) and sample image (~92 KB)
2. ✓ Initialize the object detector
3. ✓ Load the image
4. ✓ Run detection inference
5. ✓ Print detected objects with confidence scores
6. ✓ Draw bounding boxes on the image
7. ✓ Save result to `data/output.jpg`
8. ✓ Display the annotated image

### Sample Output
```
Found 1 object(s):

Object #1:
  Class: person
  Confidence: 54.67%
  Bounding Box:
    X: 50.00, Y: 47.00
    Width: 392.00, Height: 456.00
```

## File Locations
```
inClassMediaPipe/
├── object_detection.py          ← Main script
├── requirements.txt             ← Dependencies
├── README.md                    ← Full documentation
├── .gitignore                   ← Git config
└── data/
    ├── efficientdet_lite0.tflite  ← Model (downloaded)
    ├── sample_image.jpg           ← Input image (downloaded)
    └── output.jpg                 ← Detection result
```

## Git Branch Status
```
Current Branch: feature/object-detection
Commit: 3363e4e - feat: add object detection demo using MediaPipe
Status: Ready for review/merge to main
```

## Code Structure

The script follows MediaPipe's official pipeline:

1. **Setup Phase**
   - Download model and sample image
   - Create output directories

2. **Initialization Phase**
   - Load BaseOptions with model path
   - Configure ObjectDetectorOptions
   - Create ObjectDetector instance

3. **Data Preparation**
   - Load image file with OpenCV
   - Convert BGR → RGB format
   - Create MediaPipe Image object

4. **Inference Phase**
   - Run `detector.detect(image)`
   - Get ObjectDetectorResult

5. **Post-Processing**
   - Extract detections and scores
   - Convert normalized to pixel coordinates
   - Draw bounding boxes
   - Add labels with confidence

6. **Visualization**
   - Display annotated image
   - Save to output file

## Customization Options

### Change Confidence Threshold
In `create_object_detector()`:
```python
score_threshold=0.5  # Change to 0.3 for more detections
```

### Use Your Own Image
```python
cv_image, mp_image = load_image("path/to/your/image.jpg")
```

### Increase Max Detections
In `create_object_detector()`:
```python
max_results=5  # Change to higher number
```

## System Requirements
- Python 3.8+ (tested with 3.13.7)
- Windows, macOS, or Linux
- ~150 MB disk space (for model and dependencies)
- CPU-only execution supported

## Dependencies Installed
- **mediapipe** (0.10.35) - ML inference framework
- **opencv-python** (4.13.0.92) - Image processing
- **numpy** (2.4.4+) - Numerical computing

## Model Details
- **Name**: EfficientDet-Lite0
- **Size**: ~14 MB
- **Latency**: ~50-100ms per image (CPU)
- **Classes**: ~91 object classes (COCO dataset)
- **Framework**: TensorFlow Lite

## Official Documentation
https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/python

## Next Steps
1. Review the code in `object_detection.py`
2. Run the script: `python object_detection.py`
3. Check the output image: `data/output.jpg`
4. Try customizations from the README.md
5. Merge to main branch when ready: `git checkout main && git merge feature/object-detection`

---
**Demo Status**: ✅ Complete and Tested
**Branch**: `feature/object-detection`
**Date**: April 30, 2026
