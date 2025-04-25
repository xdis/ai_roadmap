"""
Data loading and preprocessing utilities for the flower classification model.
"""
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set constants
IMG_SIZE = 224  # ResNet50 default input size
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def download_and_prepare_dataset(data_dir='../data'):
    """
    Downloads the flower dataset and prepares it for training.
    
    Args:
        data_dir: Directory where the dataset will be stored
    
    Returns:
        train_ds, val_ds: Training and validation datasets
    """
    print("Downloading and preparing dataset...")
    
    data_dir = Path(data_dir)
    
    # Create directories if they don't exist
    (data_dir / 'train').mkdir(parents=True, exist_ok=True)
    (data_dir / 'test').mkdir(parents=True, exist_ok=True)
    
    # Download and prepare the dataset
    dataset, metadata = tfds.load(
        'tf_flowers',
        split=['train[:80%]', 'train[80%:]'],
        with_info=True,
        as_supervised=True,
    )
    
    train_ds, val_ds = dataset[0], dataset[1]
    
    num_classes = metadata.features['label'].num_classes
    class_names = metadata.features['label'].names
    
    print(f"Class names: {class_names}")
    
    return train_ds, val_ds, class_names, num_classes


def preprocess_image(image, label):
    """
    Preprocesses an image for the model.
    
    Args:
        image: The input image
        label: The image label
    
    Returns:
        Preprocessed image and label
    """
    # Resize the image
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    # Normalize pixel values
    image = image / 255.0
    
    return image, label


def prepare_dataset_for_training(train_ds, val_ds, cache=True):
    """
    Prepares the dataset for training by applying preprocessing,
    shuffling, batching, and prefetching.
    
    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        cache: Whether to cache the dataset
    
    Returns:
        Prepared training and validation datasets
    """
    # Apply preprocessing
    train_ds = train_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    
    # Configure datasets for performance
    if cache:
        train_ds = train_ds.cache()
    
    train_ds = train_ds.shuffle(buffer_size=1000)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(AUTOTUNE)
    
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    
    return train_ds, val_ds


def visualize_data(dataset, class_names, n=9):
    """
    Visualizes images from the dataset.
    
    Args:
        dataset: Dataset to visualize
        class_names: List of class names
        n: Number of images to visualize
    """
    plt.figure(figsize=(13, 13))
    
    # Take n samples from the dataset
    samples = dataset.unbatch().take(n)
    
    for i, (image, label) in enumerate(samples):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image)
        plt.title(class_names[label.numpy()])
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test the data loading functions
    train_ds, val_ds, class_names, num_classes = download_and_prepare_dataset()
    train_ds, val_ds = prepare_dataset_for_training(train_ds, val_ds)
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Visualize some training examples
    batch = next(iter(train_ds))
    images, labels = batch
    
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Visualize a few examples
    unbatched_ds = train_ds.unbatch()
    visualize_data(unbatched_ds, class_names)