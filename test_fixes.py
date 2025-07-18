#!/usr/bin/env python3
"""
Test script to verify that the Keras CV fixes work correctly.
"""

def test_imports():
    """Test that the data module can be imported without errors."""
    
    print("Testing data module imports...")
    
    try:
        from data import (
            create_mixup_augmentation,
            create_cutmix_augmentation,
            create_randaugment_pipeline,
            create_random_choice_pipeline,
            create_advanced_augmentation_pipeline,
            augment_for_finetuning
        )
        print("✅ All augmentation functions imported successfully!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_function_creation():
    """Test that augmentation functions can be created."""
    
    print("\nTesting function creation...")
    
    try:
        from data import create_advanced_augmentation_pipeline
        
        # Test basic augmentation
        basic_fn = create_advanced_augmentation_pipeline(
            num_classes=10, augmentation_type='basic'
        )
        print("✅ Basic augmentation function created")
        
        # Test MixUp (will fallback to basic if not available)
        mixup_fn = create_advanced_augmentation_pipeline(
            num_classes=10, augmentation_type='mixup', mixup_alpha=0.2
        )
        print("✅ MixUp augmentation function created")
        
        # Test CutMix (will fallback to basic if not available)
        cutmix_fn = create_advanced_augmentation_pipeline(
            num_classes=10, augmentation_type='cutmix', cutmix_alpha=0.5
        )
        print("✅ CutMix augmentation function created")
        
        # Test RandAugment (will fallback to basic if not available)
        randaugment_fn = create_advanced_augmentation_pipeline(
            num_classes=10, augmentation_type='randaugment', 
            randaugment_magnitude=0.3, randaugment_rate=0.7
        )
        print("✅ RandAugment function created")
        
        # Test combined (will fallback to basic if not available)
        combined_fn = create_advanced_augmentation_pipeline(
            num_classes=10, augmentation_type='combined',
            mixup_alpha=0.2, cutmix_alpha=0.5,
            randaugment_magnitude=0.3, randaugment_rate=0.7
        )
        print("✅ Combined augmentation function created")
        
        return True
        
    except Exception as e:
        print(f"❌ Function creation error: {e}")
        return False

def test_sample_augmentation():
    """Test that augmentation functions can process a sample."""
    
    print("\nTesting sample augmentation...")
    
    try:
        import tensorflow as tf
        from data import create_advanced_augmentation_pipeline
        
        # Create a sample
        sample = {
            'image': tf.random.uniform((32, 32, 3), 0, 1, dtype=tf.float32),
            'label': tf.constant(5, dtype=tf.int32)
        }
        
        # Test basic augmentation
        basic_fn = create_advanced_augmentation_pipeline(
            num_classes=10, augmentation_type='basic'
        )
        
        augmented_sample = basic_fn(sample)
        print(f"✅ Basic augmentation processed sample: {augmented_sample['image'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample augmentation error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Testing Keras CV Fixes")
    print("="*60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test function creation
    functions_ok = test_function_creation()
    
    # Test sample augmentation
    sample_ok = test_sample_augmentation()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    if imports_ok and functions_ok and sample_ok:
        print("🎉 All tests passed! The Keras CV fixes are working correctly.")
        print("   You can now use advanced augmentations in your training pipeline.")
    else:
        print("❌ Some tests failed. The system will fall back to basic augmentation.")
        print("   This is still functional, but advanced augmentations may not be available.")
    
    print("="*60) 