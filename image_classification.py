"""
Image Classification using MediaPipe - Python Demo
====================================================

This script demonstrates image classification using MediaPipe's Tasks API.
It loads a pretrained image classification model, runs classification on an image,
and prints the predicted categories with confidence scores.

Official Documentation:
https://ai.google.dev/edge/mediapipe/solutions/vision/image_classifier/python

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
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"Could not download {filename}: {e}")
            return False
    else:
        print(f"✓ {filename} already exists")
    return True


def setup_model_and_image():
    """
    Download the pretrained image classification model and a sample image.
    Returns the paths to both files.
    """
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # MediaPipe Image Classification model
    # Using EfficientNet-Lite0 - efficient for edge devices
    # This is a TensorFlow Lite model trained on ImageNet
    model_url = "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite"
    model_path = "data/efficientnet_lite0.tflite"
    
    # Sample image for testing
    sample_image_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    sample_image_path = "data/sample_image.jpg"
    
    # Download model
    if not download_file(model_url, model_path):
        return None, sample_image_path
    
    # Download or create sample image
    if not download_file(sample_image_url, sample_image_path):
        print("Creating test image...")
        create_test_image(sample_image_path)
    
    return model_path, sample_image_path


def create_test_image(output_path):
    """
    Create a simple test image for classification testing.
    
    Args:
        output_path (str): Path to save the test image
    """
    # Create a simple test image with gradient
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create a gradient pattern
    for i in range(480):
        image[i, :] = [int(i * 255 / 480), int((480-i) * 255 / 480), 128]
    
    # Add some shapes
    cv2.circle(image, (320, 240), 50, (255, 255, 0), -1)
    cv2.rectangle(image, (100, 100), (300, 300), (0, 255, 255), 2)
    
    # Save the test image
    cv2.imwrite(output_path, image)
    print(f"✓ Created test image at {output_path}")


# ============================================================================
# STEP 2: Initialize Image Classifier
# ============================================================================

def create_image_classifier(model_path):
    """
    Create and initialize an ImageClassifier using MediaPipe Tasks API.
    
    Args:
        model_path (str): Path to the pretrained .tflite model file
        
    Returns:
        vision.ImageClassifier: Configured image classifier instance
    """
    print("\n" + "="*70)
    print("INITIALIZING IMAGE CLASSIFIER")
    print("="*70)
    
    # Step 1: Create BaseOptions with the model file path
    # BaseOptions tells MediaPipe where to find the model
    base_options = python.BaseOptions(model_asset_path=model_path)
    
    # Step 2: Create ImageClassifierOptions to configure the classifier
    # We can customize classification parameters
    options = vision.ImageClassifierOptions(
        base_options=base_options,
        # Set running mode to IMAGE (for single image inference)
        running_mode=vision.RunningMode.IMAGE,
        # Maximum number of top results to return
        max_results=5,
        # Score threshold - only results above this are returned
        score_threshold=0.0
    )
    
    # Step 3: Create the ImageClassifier from the options
    classifier = vision.ImageClassifier.create_from_options(options)
    print("✓ Image Classifier initialized successfully")
    
    return classifier


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
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    print(f"✓ Image loaded successfully")
    print(f"  Image shape: {rgb_image.shape}")
    
    return cv_image, mp_image


# ============================================================================
# STEP 4: Run Image Classification
# ============================================================================

def classify_image(classifier, mp_image):
    """
    Run image classification on the image.
    
    Args:
        classifier (vision.ImageClassifier): The image classifier instance
        mp_image (vision.Image): The image to classify
        
    Returns:
        vision.ImageClassifierResult: Classification results with categories and scores
    """
    print("\n" + "="*70)
    print("RUNNING IMAGE CLASSIFICATION")
    print("="*70)
    
    # Run classification
    # This is the main inference call that processes the image
    classification_result = classifier.classify(mp_image)
    
    print("✓ Image classification completed")
    
    return classification_result


# ============================================================================
# STEP 5: Process and Print Results
# ============================================================================

def print_classification_results(classification_result):
    """
    Print classification results with category names and confidence scores.
    
    Args:
        classification_result (vision.ImageClassifierResult): The classification results
    """
    print("\n" + "="*70)
    print("CLASSIFICATION RESULTS")
    print("="*70)
    
    if not classification_result.classifications:
        print("No classifications found.")
        return
    
    # Get classifications from the first classification head
    # (most models have only one head)
    classifications = classification_result.classifications[0]
    
    if not classifications.categories:
        print("No categories found in classifications.")
        return
    
    print(f"\nTop {len(classifications.categories)} predictions:\n")
    
    for idx, category in enumerate(classifications.categories, 1):
        # Get category information
        category_name = category.display_name if category.display_name else category.category_name
        confidence = category.score
        
        # Create a visual confidence bar
        bar_length = int(confidence * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        
        print(f"{idx}. {category_name}")
        print(f"   Confidence: {confidence:.2%} [{bar}]")
        if hasattr(category, 'index'):
            print(f"   Index: {category.index}")
        print()


# ============================================================================
# STEP 6: Display Image
# ============================================================================

def display_image(cv_image, output_path="data/input_display.jpg", timeout_ms=1000):
    """
    Display the image and save it to a file.
    
    Args:
        cv_image (np.ndarray): The image to display (BGR format from OpenCV)
        output_path (str): Path to save the image
        timeout_ms (int): Timeout in milliseconds for display window
    """
    print(f"\nSaving image to: {output_path}")
    
    # Save the image
    cv2.imwrite(output_path, cv_image)
    print(f"✓ Image saved")
    
    # Try to display the image in a window
    try:
        print(f"\nAttempting to display image (timeout: {timeout_ms}ms)...")
        
        # Create a window that can be closed automatically
        window_name = 'Image Classification Input'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, cv_image)
        
        # Wait with timeout
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
    Main function that orchestrates the entire image classification pipeline.
    """
    print("\n" + "="*70)
    print("MEDIAPIPE IMAGE CLASSIFICATION DEMO")
    print("="*70)
    
    try:
        # Step 1: Setup - Download model and sample image
        print("\nStep 1: Setting up model and sample image...")
        model_path, image_path = setup_model_and_image()
        
        if model_path is None:
            print("❌ Error: Could not download model")
            return 1
        
        # Check if user has a custom image in Downloads
        user_image_path = r"C:\Users\PREDATOR\Downloads\face-1.jpg"
        if os.path.exists(user_image_path):
            image_path = user_image_path
            print(f"✓ Using user image: {user_image_path}")
        
        # Step 2: Initialize the classifier
        print("\nStep 2: Initializing image classifier...")
        classifier = create_image_classifier(model_path)
        
        # Step 3: Load the image
        print("\nStep 3: Loading image...")
        cv_image, mp_image = load_image(image_path)
        
        # Step 4: Run classification
        print("\nStep 4: Running classification...")
        classification_result = classify_image(classifier, mp_image)
        
        # Step 5: Print results
        print("\nStep 5: Processing results...")
        print_classification_results(classification_result)
        
        # Step 6: Display image
        print("\nStep 6: Displaying image...")
        display_image(cv_image)
        
        print("\n" + "="*70)
        print("IMAGE CLASSIFICATION COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
