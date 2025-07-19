#!/usr/bin/env python3
"""
Example usage of the clean KerasCV augmentation system.

This demonstrates how to use the simplified augmentation approach that follows
proper KerasCV patterns with CutMix and MixUp.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from src.data import create_augmented_dataset, get_keras_cv_augmenter, augment_with_keras_cv

def main():
    """Example of using the clean KerasCV augmentation system."""
    
    # Load a small dataset for demonstration
    print("🔄 Loading CIFAR-10 dataset...")
    (train_ds, test_ds), ds_info = tfds.load(
        'cifar10',
        split=['train[:1000]', 'test[:100]'],  # Small subset for demo
        shuffle_files=True,
        as_supervised=True,  # Returns (image, label) tuples
        with_info=True,
    )
    
    num_classes = ds_info.features['label'].num_classes
    print(f"📊 Dataset: CIFAR-10 with {num_classes} classes")
    
    # Example 1: Create an augmenter directly
    print("\n🎨 Example 1: Creating KerasCV augmenter directly")
    augmenter = get_keras_cv_augmenter(
        augmentation_type='comprehensive', 
        num_classes=num_classes
    )
    
    if augmenter is not None:
        print("✅ KerasCV augmenter created successfully!")
        print(f"   Augmenter layers: {len(augmenter.layers)} layers")
    else:
        print("❌ KerasCV not available")
    
    # Example 2: Create augmented datasets
    print("\n🎨 Example 2: Creating augmented datasets")
    batch_size = 32
    
    # Create training dataset with augmentation
    train_augmented = create_augmented_dataset(
        train_ds, 
        num_classes=num_classes,
        mode="train",
        augmentation_type='comprehensive',
        batch_size=batch_size
    )
    
    # Create validation dataset (no augmentation)
    test_augmented = create_augmented_dataset(
        test_ds,
        num_classes=num_classes,
        mode="validation",
        augmentation_type='comprehensive',
        batch_size=batch_size
    )
    
    print("✅ Augmented datasets created!")
    
    # Example 3: Manual augmentation function
    print("\n🎨 Example 3: Manual augmentation function")
    
    # Get a batch of data
    sample_batch = next(iter(train_ds.batch(4)))
    images, labels = sample_batch
    
    print(f"   Original images shape: {images.shape}")
    print(f"   Original labels shape: {labels.shape}")
    
    # Apply augmentation manually
    aug_images, aug_labels = augment_with_keras_cv(
        images, labels, num_classes, 'comprehensive'
    )
    
    print(f"   Augmented images shape: {aug_images.shape}")
    print(f"   Augmented labels shape: {aug_labels.shape}")
    
    # Example 4: Different augmentation types
    print("\n🎨 Example 4: Different augmentation types")
    
    augmentation_types = ['light', 'comprehensive', 'aggressive']
    
    for aug_type in augmentation_types:
        print(f"\n   Testing {aug_type} augmentation:")
        augmenter = get_keras_cv_augmenter(aug_type, num_classes)
        if augmenter is not None:
            print(f"     ✅ {aug_type}: {len(augmenter.layers)} layers")
        else:
            print(f"     ❌ {aug_type}: Not available")
    
    # Example 5: Iterate through augmented dataset
    print("\n🎨 Example 5: Iterating through augmented dataset")
    
    for i, (images, labels) in enumerate(train_augmented):
        print(f"   Batch {i+1}: images={images.shape}, labels={labels.shape}")
        if i >= 2:  # Show first 3 batches
            break
    
    print("\n✅ All examples completed successfully!")
    print("\n📝 Usage Summary:")
    print("   1. Use create_augmented_dataset() for full dataset pipeline")
    print("   2. Use get_keras_cv_augmenter() for custom augmenter")
    print("   3. Use augment_with_keras_cv() for manual augmentation")
    print("   4. Supports CutMix and MixUp automatically when num_classes is provided")

if __name__ == "__main__":
    main() 