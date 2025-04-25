"""
Model definition for the flower classification model.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def create_cnn_model(input_shape=(224, 224, 3), num_classes=5):
    """
    Creates a simple CNN model for image classification.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
    
    Returns:
        A compiled Keras model
    """
    # Create a sequential model
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and fully connected layers
        layers.Flatten(),
        layers.Dropout(0.5),  # Add dropout for regularization
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_transfer_learning_model(input_shape=(224, 224, 3), num_classes=5):
    """
    Creates a transfer learning model based on MobileNetV2.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
    
    Returns:
        A compiled Keras model
    """
    # Load the pre-trained MobileNetV2 model
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create a new model on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def unfreeze_model_layers(model, num_layers=10):
    """
    Unfreezes the last few layers of the base model for fine-tuning.
    
    Args:
        model: The model to fine-tune
        num_layers: Number of layers to unfreeze from the end
    
    Returns:
        Updated model with some layers unfrozen
    """
    # Unfreeze the base model
    model.layers[0].trainable = True
    
    # Freeze all the layers except the last `num_layers`
    for layer in model.layers[0].layers[:-num_layers]:
        layer.trainable = False
    
    # Compile the model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Test model creation
    basic_model = create_cnn_model()
    basic_model.summary()
    
    transfer_model = create_transfer_learning_model()
    transfer_model.summary()