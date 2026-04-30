# Object Detection using MediaPipe - Python Demo

## Overview

This demo showcases **object detection** using **MediaPipe's Tasks API**. The script loads a pretrained object detection model, runs inference on an image, and visualizes the results with bounding boxes and confidence scores.

**Official Documentation:** https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/python

---

## Features

✅ **Pretrained Model**: Uses EfficientDet-Lite0, a lightweight model optimized for edge devices  
✅ **Automatic Downloads**: Automatically downloads the model and sample image  
✅ **Object Detection**: Detects multiple objects in a single image  
✅ **Confidence Scores**: Displays confidence for each detection  
✅ **Bounding Boxes**: Draws colored boxes around detected objects  
✅ **Clear Documentation**: Well-commented code for learning  
✅ **Easy to Extend**: Simple structure for adding custom images or models  

---

## Installation

### Prerequisites
- **Python 3.8+** (tested with Python 3.13.7)
- **pip** (Python package manager)

### Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **mediapipe** (≥0.10.0) - MediaPipe Tasks API
- **opencv-python** (≥4.8.0) - Image processing
- **numpy** (≥1.24.0) - Numerical operations

---

## Usage

### Run the Demo

```bash
python object_detection.py
```

### What Happens

1. **Model & Image Download**: The script automatically downloads:
   - Pretrained object detection model (`efficientdet_lite0.tflite`)
   - Sample image for testing

2. **Object Detection**: Runs inference on the image

3. **Results Display**:
   - Prints detected objects with class names and confidence scores
   - Draws bounding boxes on the image
   - Displays the annotated image
   - Saves output to `data/output.jpg`

### Example Output

```
======================================================================
MEDIAPIPE OBJECT DETECTION DEMO
======================================================================

Step 1: Setting up model and sample image...
✓ efficientdet_lite0.tflite already exists
✓ sample_image.jpg already exists

Step 2: Initializing object detector...
======================================================================
INITIALIZING OBJECT DETECTOR
======================================================================
✓ Object Detector initialized successfully

Step 3: Loading image...
Loading image from: data/sample_image.jpg
✓ Image loaded successfully
  Image shape: (683, 1024, 3)

Step 4: Running detection...
======================================================================
RUNNING OBJECT DETECTION
======================================================================
✓ Object detection completed

Step 5: Processing results...
======================================================================
DETECTION RESULTS
======================================================================
Found 5 object(s):

Object #1:
  Class: person
  Confidence: 87.45%
  Bounding Box:
    X: 0.15, Y: 0.20
    Width: 0.25, Height: 0.35
...
```

---

## Project Structure

```
inClassMediaPipe/
├── object_detection.py      # Main demo script
├── requirements.txt          # Python dependencies
├── data/                     # Data directory (auto-created)
│   ├── efficientdet_lite0.tflite  # Pretrained model
│   ├── sample_image.jpg           # Input image
│   └── output.jpg                 # Output with bounding boxes
└── README.md                 # This file
```

---

## Script Breakdown

The script is organized into 7 clear steps:

### **Step 1: Download Model and Sample Image**
- Uses `urllib.request` to download files
- Stores in `data/` directory
- Skips download if files already exist

### **Step 2: Initialize Object Detector**
```python
# Create BaseOptions with model path
base_options = python.BaseOptions(model_asset_path=model_path)

# Configure detector options
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,    # Only objects with >50% confidence
    max_results=5           # Maximum 5 detections
)

# Create detector
detector = vision.ObjectDetector.create_from_options(options)
```

### **Step 3: Load Image**
- Loads image using OpenCV
- Converts BGR → RGB (MediaPipe expects RGB)
- Wraps in `vision.Image` object for MediaPipe

### **Step 4: Run Detection**
```python
detection_result = detector.detect(mp_image)
```

### **Step 5: Process Results**
- Extracts object class names
- Extracts confidence scores
- Prints formatted results

### **Step 6: Visualize with Bounding Boxes**
- Converts normalized coordinates to pixel coordinates
- Draws rectangles around detected objects
- Adds label text with confidence

### **Step 7: Display and Save**
- Shows image in a window
- Saves annotated image to file

---

## Customization

### Using Your Own Image

Replace the `sample_image_url` in `setup_model_and_image()`:

```python
sample_image_url = "path/to/your/image.jpg"  # Use local path
sample_image_path = "data/your_image.jpg"
```

Or load directly in `main()`:

```python
cv_image, mp_image = load_image("path/to/your/image.jpg")
```

### Adjusting Confidence Threshold

Change the `score_threshold` in `create_object_detector()`:

```python
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.7,    # Increase to 70% for stricter detections
    max_results=5
)
```

### Using Different Models

MediaPipe provides different model variants:

- **efficientdet_lite0** (smallest, fastest) - Default
- **efficientdet_lite1** (balanced)
- **efficientdet_lite2** (most accurate)

Update the `model_url` in `setup_model_and_image()` to use a different model.

---

## Model Information

### EfficientDet-Lite0

- **Size**: ~11 MB (efficient for edge devices)
- **Latency**: ~50-100ms per image
- **Accuracy**: Good for most use cases
- **Classes Detected**: COCO dataset classes (~91 objects)

Common detected objects:
- person, car, dog, cat, bicycle, backpack, handbag, book, etc.

---

## Troubleshooting

### Issue: `No objects detected`
- **Solution**: Try lowering the `score_threshold` to 0.3 or 0.4
- Image may not contain recognizable objects

### Issue: `Cannot import mediapipe`
- **Solution**: Run `pip install mediapipe`
- Ensure Python version is 3.8+

### Issue: `Image file not found`
- **Solution**: Check the image path is correct
- Ensure the image format is supported (JPG, PNG, etc.)

### Issue: `No display window appears`
- **Solution**: This is normal on headless systems
- Check `data/output.jpg` for the result

---

## Git Workflow

This demo is developed on the `feature/object-detection` branch:

```bash
# Check current branch
git branch

# Switch to feature branch
git checkout feature/object-detection

# View changes
git status

# Commit changes
git add .
git commit -m "feat: add object detection demo"

# Push to remote (if configured)
git push origin feature/object-detection
```

---

## Performance Notes

- **First Run**: Takes longer due to model download (~50 MB)
- **Subsequent Runs**: Faster as files are cached
- **Model Inference**: ~50-100ms per image on CPU
- **GPU Support**: Can be faster with GPU (requires additional setup)

---

## References

- **MediaPipe Official Docs**: https://ai.google.dev/edge/mediapipe
- **Object Detection Guide**: https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/python
- **MediaPipe Models**: https://storage.googleapis.com/mediapipe-models/
- **OpenCV Docs**: https://docs.opencv.org/
- **MediaPipe Tasks Python API**: https://developers.google.com/mediapipe/framework/task_library/overview

---

## License

This demo is provided as-is for educational purposes.

MediaPipe is licensed under the Apache License 2.0.

---

## Version Info

- **MediaPipe**: 0.10.35
- **OpenCV**: 4.13.0.92
- **Python**: 3.8+
- **Date**: 2026

---

## Questions or Issues?

Refer to the official MediaPipe documentation or the script comments for more details.

Happy detecting! 🎯
