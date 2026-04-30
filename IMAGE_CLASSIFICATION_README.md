# Image Classification using MediaPipe - Python Demo

## Overview

This demo showcases **image classification** using **MediaPipe's Tasks API**. The script loads a pretrained image classification model, runs inference on an image, and displays the predicted categories with confidence scores.

**Official Documentation:** https://ai.google.dev/edge/mediapipe/solutions/vision/image_classifier/python

---

## Features

✅ **Pretrained Model**: Uses EfficientNet-Lite0 trained on ImageNet dataset  
✅ **Automatic Downloads**: Automatically downloads model and sample image  
✅ **Image Classification**: Classifies images into 1000+ categories  
✅ **Confidence Scores**: Displays confidence for each prediction  
✅ **Top Predictions**: Shows top 5 predicted categories  
✅ **Clear Documentation**: Well-commented code for learning  
✅ **Visual Confidence Bars**: Easy-to-read output formatting  

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

### Run the Classification Script

```bash
python image_classification.py
```

### What Happens

1. **Model & Image Download**: The script automatically downloads:
   - Pretrained image classification model (~8 MB)
   - Sample image for testing

2. **Image Classification**: Runs inference on the image

3. **Results Display**:
   - Prints top 5 predictions with categories
   - Shows confidence scores as percentage
   - Displays visual confidence bars
   - Saves image to `data/input_display.jpg`
   - Displays the image in a window (with timeout)

### Example Output

```
Found 1 object(s):

Top 5 predictions:

1. Lena
   Confidence: 45.23% [████████████████████░░░░░░░░░░░░░░░░░░]
   Index: 0

2. Woman
   Confidence: 32.15% [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░]
   Index: 1

3. Face
   Confidence: 15.67% [██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
   Index: 2
```

---

## Project Structure

```
inClassMediaPipe/
├── image_classification.py      # Main demo script
├── requirements.txt             # Python dependencies
├── README.md (or this file)    # Documentation
└── data/                        # Data directory (auto-created)
    ├── efficientnet_lite0.tflite  # Model (downloaded)
    ├── sample_image.jpg           # Input image (downloaded)
    └── input_display.jpg          # Display copy of input
```

---

## Script Breakdown

The script is organized into 6 clear steps:

### **Step 1: Download Model and Sample Image**
```python
# Automatically downloads model and image if they don't exist
model_path, image_path = setup_model_and_image()
```

### **Step 2: Initialize Image Classifier**
```python
# Create BaseOptions with model path
base_options = python.BaseOptions(model_asset_path=model_path)

# Configure classifier options
options = vision.ImageClassifierOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    max_results=5,              # Top 5 predictions
    score_threshold=0.0         # Include all predictions
)

# Create classifier
classifier = vision.ImageClassifier.create_from_options(options)
```

### **Step 3: Load Image**
- Loads image using OpenCV
- Converts BGR → RGB (MediaPipe expects RGB)
- Wraps in `mp.Image` object for MediaPipe

### **Step 4: Run Classification**
```python
classification_result = classifier.classify(mp_image)
```

### **Step 5: Process and Display Results**
- Extracts category names and confidence scores
- Prints formatted results with visual bars
- Shows top 5 predictions

### **Step 6: Display Image**
- Shows image in a window (with timeout)
- Saves image to file

---

## API Reference

### ImageClassifier Configuration

**Running Modes**:
- `IMAGE` - For single image inference (default, used here)
- `VIDEO` - For video file processing
- `LIVE_STREAM` - For camera/streaming input

**Options**:
- `max_results` (int) - Maximum predictions to return (default: -1 for all)
- `score_threshold` (float) - Minimum confidence threshold (0.0 to 1.0)
- `display_names_locale` (str) - Language for labels (default: "en")

---

## Model Information

### EfficientNet-Lite0

- **Size**: ~8 MB (efficient for edge devices)
- **Latency**: ~30-50ms per image on CPU
- **Input**: 224x224 RGB images
- **Classes**: 1000 ImageNet classes
- **Framework**: TensorFlow Lite
- **Dataset**: ImageNet

**Common categories detected**:
- Animals: dog, cat, bird, elephant, etc.
- Objects: car, bicycle, bottle, cup, etc.
- Scenes: landscape, indoor, outdoor, etc.
- Activities and more

---

## Customization

### Change Confidence Threshold

In `create_image_classifier()`:
```python
score_threshold=0.5  # Only predictions with >50% confidence
```

### Use Your Own Image

```python
cv_image, mp_image = load_image("path/to/your/image.jpg")
```

### Increase Number of Predictions

In `create_image_classifier()`:
```python
max_results=10  # Show top 10 predictions instead of 5
```

### Use Different Model

Download a different model and update:
```python
model_url = "https://path/to/model.tflite"
```

---

## System Requirements

- **Python**: 3.8+ (tested with 3.13.7)
- **OS**: Windows, macOS, or Linux
- **Disk**: ~100 MB (for model and dependencies)
- **RAM**: ~500 MB for inference
- **Processor**: CPU-only (no GPU required)

---

## Dependencies Installed

- **mediapipe** (0.10.35) - ML inference framework
- **opencv-python** (4.13.0.92) - Image processing
- **numpy** (2.4.4+) - Numerical computing

---

## Troubleshooting

### Issue: `No classifications found`
- **Solution**: Ensure image is in a supported format (JPG, PNG, BMP)
- Check that image is not corrupted

### Issue: `Cannot import mediapipe`
- **Solution**: Run `pip install mediapipe`
- Ensure Python version is 3.8+

### Issue: `Image file not found`
- **Solution**: Check the image path is correct
- Ensure image exists and is readable

### Issue: Very low confidence scores
- **Solution**: This is normal for edge images
- Try with a different image
- Adjust `score_threshold` if needed

### Issue: No display window appears
- **Solution**: This is normal on headless systems
- Check `data/input_display.jpg` for the result

---

## Performance Notes

- **First Run**: Takes longer due to model download (~20 MB total)
- **Subsequent Runs**: Faster as files are cached
- **Model Inference**: ~30-50ms per image on CPU
- **GPU Support**: Can be faster with GPU (requires additional setup)

---

## Differences from Object Detection

| Feature | Object Detection | Image Classification |
|---------|------------------|---------------------|
| Output | Bounding boxes | Category predictions |
| Result Type | Detections + boxes | Categories + scores |
| Use Case | What's in image | What is the image |
| Model | EfficientDet-Lite0 | EfficientNet-Lite0 |

---

## References

- **MediaPipe Official Docs**: https://ai.google.dev/edge/mediapipe
- **Image Classification Guide**: https://ai.google.dev/edge/mediapipe/solutions/vision/image_classifier/python
- **MediaPipe Models**: https://storage.googleapis.com/mediapipe-models/
- **ImageNet Classes**: https://www.tensorflow.org/datasets/catalog/imagenet2012
- **OpenCV Docs**: https://docs.opencv.org/

---

## Next Steps

1. Run the script: `python image_classification.py`
2. Try with different images
3. Adjust confidence thresholds
4. Explore different models
5. Combine with object detection for comprehensive analysis

---

## Version Info

- **MediaPipe**: 0.10.35
- **OpenCV**: 4.13.0.92
- **Python**: 3.8+
- **Date**: 2026

---

## License

This demo is provided as-is for educational purposes.

MediaPipe is licensed under the Apache License 2.0.

---

Happy classifying! 🎯
