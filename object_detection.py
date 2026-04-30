"""
Object Detection using MediaPipe - Python Demo
===============================================

This script demonstrates object detection using MediaPipe's Tasks API.
It loads a pretrained object detection model, runs detection on an image,
prints detected objects with confidence scores, and displays the results
with bounding boxes.

Official Documentation:
https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/python

Author: MediaPipe Demo
Date: 2026
"""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================================
# STEP 1: Download Model and Sample Image
# ============================================================================

def download_file(url, filename):
    """
    Download a file from a URL and save it locally.
    
    Args:
        url (str): The URL to download from
        filename (str): The local filename to save to
    """
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✓ Downloaded {filename}")
    else:
        print(f"✓ {filename} already exists")


def setup_model_and_image():
    """
    Download the pretrained object detection model and a sample image.
    Returns the paths to both files. If download fails, creates a test image.
    """
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # MediaPipe Object Detection model
    # This is the efficient lite model suitable for edge devices
    model_url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
    model_path = "data/efficientdet_lite0.tflite"
    
    # Sample image for testing
    # Using a more stable image URL
    sample_image_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    sample_image_path = "data/sample_image.jpg"
    
    # Download model
    download_file(model_url, model_path)
    
    # Try to download image, if fails, create a test image
    try:
        download_file(sample_image_url, sample_image_path)
    except Exception as e:
        print(f"Could not download image: {e}")
        print("Creating a test image with multiple colored regions...")
        create_test_image(sample_image_path)
    
    return model_path, sample_image_path


def create_test_image(output_path):
    """
    Create a simple test image with colored regions for detection testing.
    
    Args:
        output_path (str): Path to save the test image
    """
    # Create a simple test image with colored rectangles
    # This helps verify the detector works even without a real image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add colored regions
    image[50:150, 50:150] = [0, 255, 0]      # Green rectangle
    image[200:350, 100:300] = [255, 0, 0]    # Blue rectangle
    image[100:250, 350:550] = [0, 0, 255]    # Red rectangle
    
    # Add some white noise for texture
    noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    
    # Save the test image
    cv2.imwrite(output_path, image)
    print(f"✓ Created test image at {output_path}")


# ============================================================================
# STEP 2: Initialize Object Detector
# ============================================================================

def create_object_detector(model_path):
    """
    Create and initialize an ObjectDetector using MediaPipe Tasks API.
    
    Args:
        model_path (str): Path to the pretrained .tflite model file
        
    Returns:
        vision.ObjectDetector: Configured object detector instance
    """
    print("\n" + "="*70)
    print("INITIALIZING OBJECT DETECTOR")
    print("="*70)
    
    # Step 1: Create BaseOptions with the model file path
    # BaseOptions tells MediaPipe where to find the model and how to load it
    base_options = python.BaseOptions(model_asset_path=model_path)
    
    # Step 2: Create ObjectDetectorOptions to configure the detector
    # We can customize detection parameters like confidence threshold
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        # Set running mode to IMAGE (for single image inference)
        running_mode=vision.RunningMode.IMAGE,
        # Set confidence threshold (0.0-1.0)
        # Only detections above this threshold will be returned
        score_threshold=0.5,
        # Maximum number of detections
        max_results=5
    )
    
    # Step 3: Create the ObjectDetector from the options
    detector = vision.ObjectDetector.create_from_options(options)
    print("✓ Object Detector initialized successfully")
    
    return detector


# ============================================================================
# STEP 3: Load and Prepare Image
# ============================================================================

def load_image(image_path):
    """
    Load an image from file and convert it to MediaPipe format.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (cv2_image, mp_image) - OpenCV image and MediaPipe image object
    """
    print(f"\nLoading image from: {image_path}")
    
    # Load image using OpenCV
    cv_image = cv2.imread(image_path)
    
    if cv_image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB (OpenCV uses BGR by default)
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image object from numpy array
    # The Image class wraps the numpy array with metadata
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    print(f"✓ Image loaded successfully")
    print(f"  Image shape: {rgb_image.shape}")
    
    return cv_image, mp_image


# ============================================================================
# STEP 4: Run Object Detection
# ============================================================================

def detect_objects(detector, mp_image):
    """
    Run object detection on the image.
    
    Args:
        detector (vision.ObjectDetector): The object detector instance
        mp_image (vision.Image): The image to detect objects in
        
    Returns:
        vision.ObjectDetectorResult: Detection results with bounding boxes and labels
    """
    print("\n" + "="*70)
    print("RUNNING OBJECT DETECTION")
    print("="*70)
    
    # Run detection
    # This is the main inference call that processes the image
    detection_result = detector.detect(mp_image)
    
    print("✓ Object detection completed")
    
    return detection_result


# ============================================================================
# STEP 5: Process and Print Results
# ============================================================================

def print_detection_results(detection_result):
    """
    Print detected objects with their confidence scores and bounding boxes.
    
    Args:
        detection_result (vision.ObjectDetectorResult): The detection results
    """
    print("\n" + "="*70)
    print("DETECTION RESULTS")
    print("="*70)
    
    if len(detection_result.detections) == 0:
        print("No objects detected.")
        return
    
    print(f"Found {len(detection_result.detections)} object(s):\n")
    
    for idx, detection in enumerate(detection_result.detections, 1):
        # Get bounding box coordinates
        bbox = detection.bounding_box
        origin_x = bbox.origin_x
        origin_y = bbox.origin_y
        width = bbox.width
        height = bbox.height
        
        # Get confidence score(s)
        # Usually there's one confidence value per class
        if len(detection.categories) > 0:
            category = detection.categories[0]
            category_name = category.category_name
            confidence = category.score
            
            print(f"Object #{idx}:")
            print(f"  Class: {category_name}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Bounding Box:")
            print(f"    X: {origin_x:.2f}, Y: {origin_y:.2f}")
            print(f"    Width: {width:.2f}, Height: {height:.2f}")
            print()


# ============================================================================
# STEP 6: Visualize Results with Bounding Boxes
# ============================================================================

def draw_bounding_boxes(cv_image, detection_result):
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        cv_image (np.ndarray): The image to draw on (BGR format from OpenCV)
        detection_result (vision.ObjectDetectorResult): Detection results
        
    Returns:
        np.ndarray: Image with bounding boxes drawn
    """
    print("\nDrawing bounding boxes...")
    
    # Make a copy so we don't modify the original
    annotated_image = cv_image.copy()
    
    # Define colors for visualization (BGR format for OpenCV)
    box_color = (0, 255, 0)      # Green for bounding boxes
    text_color = (255, 255, 255) # White for text
    text_bg_color = (0, 0, 0)    # Black background for text
    
    # Get image dimensions for coordinate conversion
    height, width, _ = annotated_image.shape
    
    for detection in detection_result.detections:
        # Get bounding box in normalized coordinates (0-1)
        bbox = detection.bounding_box
        
        # Convert normalized coordinates to pixel coordinates
        x_min = int(bbox.origin_x * width)
        y_min = int(bbox.origin_y * height)
        x_max = int((bbox.origin_x + bbox.width) * width)
        y_max = int((bbox.origin_y + bbox.height) * height)
        
        # Ensure coordinates are within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)
        
        # Draw bounding box rectangle
        cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), 
                     box_color, thickness=2)
        
        # Prepare label text
        if len(detection.categories) > 0:
            category = detection.categories[0]
            label = f"{category.category_name} ({category.score:.2%})"
            
            # Get text size to create background for better readability
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            
            # Draw background rectangle for text
            text_x = x_min
            text_y = y_min - 10
            cv2.rectangle(annotated_image,
                         (text_x, text_y - text_size[1] - 5),
                         (text_x + text_size[0] + 5, text_y + 5),
                         text_bg_color, -1)
            
            # Put the label text
            cv2.putText(annotated_image, label, (text_x, text_y),
                       font, font_scale, text_color, font_thickness)
    
    print("✓ Bounding boxes drawn successfully")
    return annotated_image


# ============================================================================
# STEP 7: Display Image
# ============================================================================

def display_and_save_image(annotated_image, output_path="data/output.jpg", timeout_ms=1000):
    """
    Display the image with bounding boxes and save it to a file.
    
    Args:
        annotated_image (np.ndarray): The image with bounding boxes
        output_path (str): Path to save the output image
        timeout_ms (int): Timeout in milliseconds for display window
    """
    print(f"\nSaving output image to: {output_path}")
    
    # Save the annotated image
    cv2.imwrite(output_path, annotated_image)
    print(f"✓ Output image saved")
    
    # Try to display the image in a window
    # Set display mode to non-blocking
    try:
        print(f"\nAttempting to display image (timeout: {timeout_ms}ms)...")
        
        # Create a window that can be closed automatically
        window_name = 'Object Detection Results'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, annotated_image)
        
        # Wait with timeout - returns immediately if no key is pressed
        # This prevents blocking on headless systems
        cv2.waitKey(timeout_ms)
        cv2.destroyAllWindows()
        
        print("✓ Display window closed")
        
    except Exception as e:
        print(f"✓ Image display skipped ({type(e).__name__})")
        print(f"  Output saved to: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function that orchestrates the entire object detection pipeline.
    """
    print("\n" + "="*70)
    print("MEDIAPIPE OBJECT DETECTION DEMO")
    print("="*70)
    
    try:
        # Step 1: Setup - Download model and sample image
        print("\nStep 1: Setting up model and sample image...")
        model_path, image_path = setup_model_and_image()
        
        # Step 2: Initialize the detector
        print("\nStep 2: Initializing object detector...")
        detector = create_object_detector(model_path)
        
        # Step 3: Load the image
        print("\nStep 3: Loading image...")
        cv_image, mp_image = load_image(image_path)
        
        # Step 4: Run detection
        print("\nStep 4: Running detection...")
        detection_result = detect_objects(detector, mp_image)
        
        # Step 5: Print results
        print("\nStep 5: Processing results...")
        print_detection_results(detection_result)
        
        # Step 6: Visualize with bounding boxes
        print("\nStep 6: Visualizing results...")
        annotated_image = draw_bounding_boxes(cv_image, detection_result)
        
        # Step 7: Display and save
        print("\nStep 7: Displaying and saving results...")
        display_and_save_image(annotated_image)
        
        print("\n" + "="*70)
        print("OBJECT DETECTION COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
