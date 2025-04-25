"""
Prediction script for the flower classification model.
This script loads a trained model and uses it to make predictions on new images.
"""
import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Set constants
IMG_SIZE = 224  # Same as training

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions with the flower classifier')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image file to classify')
    parser.add_argument('--model', type=str, default='../models/flower_classifier_model.h5',
                        help='Path to the trained model file')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Show top k predictions')
    
    return parser.parse_args()

def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image for prediction.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Preprocessed image tensor ready for prediction
    """
    # Load image
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure image is RGB
    
    # Convert to tensor and resize
    img = tf.image.resize(np.array(img), [IMG_SIZE, IMG_SIZE])
    
    # Normalize pixel values
    img = img / 255.0
    
    # Add batch dimension
    img = tf.expand_dims(img, 0)
    
    return img

def load_class_names(model_dir):
    """
    Load class names from the model directory.
    
    Args:
        model_dir: Directory containing the model and class names file
    
    Returns:
        List of class names
    """
    class_file = os.path.join(Path(model_dir).parent, 'class_names.txt')
    
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    else:
        print(f"Warning: Class names file not found at {class_file}")
        return [f"Class {i}" for i in range(5)]  # Default class names

def predict_and_visualize(image_path, model, class_names, top_k=3):
    """
    Make a prediction on an image and visualize the results.
    
    Args:
        image_path: Path to the image file
        model: Trained TensorFlow model
        class_names: List of class names
        top_k: Number of top predictions to show
    """
    # Load and preprocess the image
    img = load_and_preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(img)
    
    # Get top k predictions
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    top_values = predictions[0][top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    # Print predictions
    print(f"Top {top_k} predictions for {image_path}:")
    for i, (class_name, prob) in enumerate(zip(top_classes, top_values)):
        print(f"{i+1}. {class_name}: {prob:.4f} ({prob*100:.2f}%)")
    
    # Load original image for display
    display_img = Image.open(image_path)
    
    # Create figure for visualization
    plt.figure(figsize=(12, 6))
    
    # Plot the image
    plt.subplot(1, 2, 1)
    plt.imshow(display_img)
    plt.title(f"Predicted: {top_classes[0]}")
    plt.axis('off')
    
    # Plot the predictions
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(top_classes))
    plt.barh(y_pos, top_values, align='center')
    plt.yticks(y_pos, top_classes)
    plt.xlabel('Probability')
    plt.title('Top Predictions')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig('predictions.png')
    print(f"Visualization saved as predictions.png")
    
    # Show the plot
    plt.show()

def main():
    """Main prediction function."""
    args = parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' does not exist")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' does not exist")
        return
    
    # Load the model
    try:
        print(f"Loading model from {args.model}...")
        model = tf.keras.models.load_model(args.model)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load class names
    class_names = load_class_names(args.model)
    
    # Make prediction
    predict_and_visualize(args.image, model, class_names, args.top_k)

if __name__ == "__main__":
    main()