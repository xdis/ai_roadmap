"""
Training script for the flower classification model.
"""
import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import datetime

# Import local modules
from data_loader import download_and_prepare_dataset, prepare_dataset_for_training
from model import create_cnn_model, create_transfer_learning_model, unfreeze_model_layers

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a flower classification model')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory to store the dataset')
    parser.add_argument('--model_dir', type=str, default='../models',
                        help='Directory to save the trained model')
    parser.add_argument('--model_type', type=str, default='transfer',
                        choices=['basic', 'transfer'],
                        help='Type of model to train (basic CNN or transfer learning)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Whether to fine-tune the transfer learning model')
    parser.add_argument('--fine_tune_epochs', type=int, default=5,
                        help='Number of fine-tuning epochs')
    
    return parser.parse_args()

def plot_training_history(history):
    """Plot training and validation accuracy/loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig('../models/training_history.png')
    plt.show()

def main():
    """Main training function."""
    args = parse_args()
    
    # Create directories if they don't exist
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    
    # Load and prepare the dataset
    train_ds, val_ds, class_names, num_classes = download_and_prepare_dataset(args.data_dir)
    train_ds, val_ds = prepare_dataset_for_training(train_ds, val_ds)
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Save class names for inference
    with open(os.path.join(args.model_dir, 'class_names.txt'), 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    # Create the model
    if args.model_type == 'basic':
        model = create_cnn_model(num_classes=num_classes)
        print("Created basic CNN model")
    else:
        model = create_transfer_learning_model(num_classes=num_classes)
        print("Created transfer learning model based on MobileNetV2")
    
    # Create TensorBoard callback
    log_dir = os.path.join("../logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Create checkpoint callback
    checkpoint_path = os.path.join(args.model_dir, "checkpoint.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor='val_accuracy'
    )
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[tensorboard_callback, checkpoint_callback]
    )
    
    # Fine-tune the model if requested and if it's a transfer learning model
    if args.fine_tune and args.model_type == 'transfer':
        print(f"Fine-tuning the model for {args.fine_tune_epochs} epochs...")
        model = unfreeze_model_layers(model)
        
        # Train with a lower learning rate
        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=[tensorboard_callback, checkpoint_callback]
        )
        
        # Combine the histories
        for key in history.history:
            history.history[key].extend(fine_tune_history.history[key])
    
    # Plot training history
    plot_training_history(history)
    
    # Save the final model
    model_path = os.path.join(args.model_dir, "flower_classifier_model.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate the model
    print("Evaluating model on validation set...")
    loss, accuracy = model.evaluate(val_ds)
    print(f"Validation accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()