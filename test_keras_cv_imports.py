#!/usr/bin/env python3
"""
Test script to verify Keras CV imports work correctly.
"""

import tensorflow as tf
import keras_cv

def test_keras_cv_imports():
    """Test that all Keras CV layers can be imported correctly."""
    
    print("Testing Keras CV imports...")
    
    try:
        # Test basic layers
        print("✅ Testing basic layers...")
        rotation = tf.keras.layers.RandomRotation(factor=(-0.1, 0.1))
        print("   RandomRotation: OK")
        
        # Test Keras CV layers
        print("✅ Testing Keras CV layers...")
        
        # MixUp and CutMix
        mixup = keras_cv.layers.MixUp(alpha=0.2)
        print("   MixUp: OK")
        
        cutmix = keras_cv.layers.CutMix(alpha=0.5)
        print("   CutMix: OK")
        
        # RandomCutout
        cutout = keras_cv.layers.RandomCutout(0.5, 0.5)
        print("   RandomCutout: OK")
        
        # RandAugment
        layers = keras_cv.layers.RandAugment.get_standard_policy(
            value_range=(0, 255), magnitude=0.3
        )
        print("   RandAugment.get_standard_policy: OK")
        
        # RandomAugmentationPipeline
        pipeline = keras_cv.layers.RandomAugmentationPipeline(
            layers=layers[:2], augmentations_per_image=1, rate=0.5
        )
        print("   RandomAugmentationPipeline: OK")
        
        # RandomChoice
        choice = keras_cv.layers.RandomChoice(layers=layers[:2])
        print("   RandomChoice: OK")
        
        # Additional layers
        solarize = keras_cv.layers.RandomSolarization(
            value_range=(0, 1), threshold_factor=0.5
        )
        print("   RandomSolarization: OK")
        
        posterize = keras_cv.layers.RandomPosterization(
            value_range=(0, 1), factor=(4, 8)
        )
        print("   RandomPosterization: OK")
        
        equalize = keras_cv.layers.RandomEqualization(
            value_range=(0, 1)
        )
        print("   RandomEqualization: OK")
        
        # Gaussian blur
        blur = keras_cv.layers.RandomGaussianBlur(
            kernel_size=3, factor=(0.0, 1.0)
        )
        print("   RandomGaussianBlur: OK")
        
        print("\n🎉 All Keras CV imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import error: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

def test_augmentation_functions():
    """Test that our augmentation functions can be created."""
    
    print("\nTesting augmentation function creation...")
    
    try:
        # Import our functions
        from data import (
            create_mixup_augmentation,
            create_cutmix_augmentation,
            create_randaugment_pipeline,
            create_random_choice_pipeline,
            create_advanced_augmentation_pipeline
        )
        
        # Test function creation
        print("✅ Testing function creation...")
        
        mixup_fn = create_mixup_augmentation(num_classes=10, alpha=0.2)
        print("   create_mixup_augmentation: OK")
        
        cutmix_fn = create_cutmix_augmentation(num_classes=10, alpha=0.5)
        print("   create_cutmix_augmentation: OK")
        
        randaugment_fn = create_randaugment_pipeline(
            augmentations_per_image=1, magnitude=0.3, rate=0.5
        )
        print("   create_randaugment_pipeline: OK")
        
        random_choice_fn = create_random_choice_pipeline(magnitude=0.3)
        print("   create_random_choice_pipeline: OK")
        
        advanced_fn = create_advanced_augmentation_pipeline(
            num_classes=10, augmentation_type='mixup', mixup_alpha=0.2
        )
        print("   create_advanced_augmentation_pipeline: OK")
        
        print("\n🎉 All augmentation functions created successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Function creation error: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Keras CV Import Test")
    print("="*60)
    
    # Test imports
    imports_ok = test_keras_cv_imports()
    
    # Test function creation
    functions_ok = test_augmentation_functions()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    if imports_ok and functions_ok:
        print("🎉 All tests passed! Keras CV integration is working correctly.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    print("="*60) 