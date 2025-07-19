#!/usr/bin/env python3
"""
Test script to demonstrate random choice augmentation functionality.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from src.data import (
    augment_with_random_choice, 
    get_comprehensive_random_choice_augmentations,
    get_aggressive_random_choice_augmentations,
    get_light_random_choice_augmentations
)

def create_test_image():
    """Create a simple test image for augmentation testing."""
    # Create a simple colored pattern
    img = np.zeros((64, 64, 3), dtype=np.float32)
    
    # Add some colored rectangles
    img[10:20, 10:20] = [1.0, 0.0, 0.0]  # Red
    img[30:40, 30:40] = [0.0, 1.0, 0.0]  # Green
    img[50:60, 50:60] = [0.0, 0.0, 1.0]  # Blue
    
    # Add some text-like patterns
    img[15:25, 40:50] = [1.0, 1.0, 0.0]  # Yellow
    img[35:45, 15:25] = [1.0, 0.0, 1.0]  # Magenta
    
    return img

def test_augmentations():
    """Test different augmentation types."""
    print("🧪 Testing Random Choice Augmentation")
    print("=" * 50)
    
    # Create test image
    original_img = create_test_image()
    
    # Create sample dictionary
    sample = {'image': original_img, 'label': 0}
    
    # Test different augmentation types
    augmentation_types = ['light', 'comprehensive', 'aggressive']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Test each augmentation type
    for i, aug_type in enumerate(augmentation_types):
        print(f"\n📊 Testing {aug_type.upper()} augmentation:")
        
        # Apply augmentation multiple times
        for j in range(3):
            augmented_sample = augment_with_random_choice(sample, aug_type)
            augmented_img = augmented_sample['image']
            
            # Display
            row = j
            col = i + 1
            axes[row, col].imshow(augmented_img)
            axes[row, col].set_title(f'{aug_type.capitalize()} {j+1}')
            axes[row, col].axis('off')
            
            print(f"   Sample {j+1}: Shape {augmented_img.shape}, Range [{augmented_img.min():.3f}, {augmented_img.max():.3f}]")
    
    plt.tight_layout()
    plt.savefig('augmentation_test.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Augmentation test saved to: augmentation_test.png")
    
    return fig

def test_augmentation_layers():
    """Test the individual augmentation layers."""
    print("\n🔧 Testing Individual Augmentation Layers")
    print("=" * 50)
    
    # Test comprehensive augmentations
    comprehensive_layers = get_comprehensive_random_choice_augmentations()
    print(f"Comprehensive augmentations: {len(comprehensive_layers)} layers")
    
    # Test aggressive augmentations
    aggressive_layers = get_aggressive_random_choice_augmentations()
    print(f"Aggressive augmentations: {len(aggressive_layers)} layers")
    
    # Test light augmentations
    light_layers = get_light_random_choice_augmentations()
    print(f"Light augmentations: {len(light_layers)} layers")
    
    # Test a few specific layers
    test_img = create_test_image()
    test_img_tensor = tf.convert_to_tensor(test_img)
    
    print("\n📊 Testing specific augmentation layers:")
    
    # Test rotation
    rotation_layer = tf.keras.layers.RandomRotation(factor=(-0.2, 0.2), fill_mode='reflect')
    rotated_img = rotation_layer(test_img_tensor, training=True)
    print(f"   Rotation: Shape {rotated_img.shape}, Range [{rotated_img.numpy().min():.3f}, {rotated_img.numpy().max():.3f}]")
    
    # Test brightness
    brightness_layer = tf.keras.layers.RandomBrightness(factor=(-0.3, 0.3))
    bright_img = brightness_layer(test_img_tensor, training=True)
    print(f"   Brightness: Shape {bright_img.shape}, Range [{bright_img.numpy().min():.3f}, {bright_img.numpy().max():.3f}]")
    
    # Test saturation
    saturation_img = tf.image.random_saturation(test_img_tensor, 0.7, 1.3)
    print(f"   Saturation: Shape {saturation_img.shape}, Range [{saturation_img.numpy().min():.3f}, {saturation_img.numpy().max():.3f}]")
    
    # Test cutout
    try:
        import keras_cv
        cutout_layer = keras_cv.layers.RandomCutout(height_factor=0.2, width_factor=0.2)
        cutout_img = cutout_layer(test_img_tensor, training=True)
        print(f"   Cutout: Shape {cutout_img.shape}, Range [{cutout_img.numpy().min():.3f}, {cutout_img.numpy().max():.3f}]")
    except ImportError:
        print("   Cutout: keras_cv not available")
    
    # Test hue
    hue_img = tf.image.random_hue(test_img_tensor, 0.1)
    print(f"   Hue: Shape {hue_img.shape}, Range [{hue_img.numpy().min():.3f}, {hue_img.numpy().max():.3f}]")

def test_batch_augmentation():
    """Test batch augmentation functionality."""
    print("\n📦 Testing Batch Augmentation")
    print("=" * 50)
    
    # Create a batch of test images
    batch_size = 4
    batch_images = np.stack([create_test_image() for _ in range(batch_size)])
    batch_labels = np.array([0, 1, 2, 3])
    
    batch = {'image': batch_images, 'label': batch_labels}
    
    print(f"Batch shape: {batch_images.shape}")
    print(f"Labels: {batch_labels}")
    
    # Test batch augmentation
    from src.data import augment_with_random_choice_batch
    
    augmented_batch = augment_with_random_choice_batch(batch, 'comprehensive')
    
    print(f"Augmented batch shape: {augmented_batch['image'].shape}")
    print(f"Augmented labels: {augmented_batch['label']}")
    
    # Display batch results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(4):
        # Original
        axes[0, i].imshow(batch_images[i])
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Augmented
        axes[1, i].imshow(augmented_batch['image'][i])
        axes[1, i].set_title(f'Augmented {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('batch_augmentation_test.png', dpi=150, bbox_inches='tight')
    print(f"💾 Batch augmentation test saved to: batch_augmentation_test.png")
    
    return fig

if __name__ == "__main__":
    # Test individual augmentations
    test_augmentations()
    
    # Test augmentation layers
    test_augmentation_layers()
    
    # Test batch augmentation
    test_batch_augmentation()
    
    print("\n✅ All augmentation tests completed!")
    print("Check the generated PNG files to see the augmentation results.") 