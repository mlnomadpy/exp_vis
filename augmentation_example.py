#!/usr/bin/env python3
"""
Example script demonstrating advanced data augmentation techniques.

This script shows how to use different augmentation types with the training pipeline.
"""

import subprocess
import sys

def run_augmentation_example(dataset, augmentation_type, **kwargs):
    """Run a training example with specific augmentation settings."""
    
    cmd = [
        sys.executable, "src/main.py",
        "--dataset", dataset,
        "--learning_rate", "0.01",
        "--augmentation_type", augmentation_type,
        "--num_epochs", "5",  # Short training for demo
        "--batch_size", "64",
        "--eval_every", "50",
    ]
    
    # Add augmentation-specific parameters
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Dataset: {dataset}")
    print(f"Augmentation: {augmentation_type}")
    print("="*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Run various augmentation examples."""
    
    print("🚀 Advanced Data Augmentation Examples")
    print("="*60)
    
    examples = [
        {
            "name": "Basic Augmentation",
            "dataset": "cifar10",
            "augmentation_type": "basic",
        },
        {
            "name": "MixUp Augmentation",
            "dataset": "cifar10",
            "augmentation_type": "mixup",
            "mixup_alpha": 0.2,
        },
        {
            "name": "CutMix Augmentation",
            "dataset": "cifar10",
            "augmentation_type": "cutmix",
            "cutmix_alpha": 0.5,
        },
        {
            "name": "RandAugment Pipeline",
            "dataset": "mnist",
            "augmentation_type": "randaugment",
            "randaugment_magnitude": 0.3,
            "randaugment_rate": 0.7,
        },
        {
            "name": "Random Choice Augmentation",
            "dataset": "fashion_mnist",
            "augmentation_type": "random_choice",
            "randaugment_magnitude": 0.3,
        },
        {
            "name": "Combined Augmentation",
            "dataset": "cifar10",
            "augmentation_type": "combined",
            "mixup_alpha": 0.2,
            "cutmix_alpha": 0.5,
            "randaugment_magnitude": 0.3,
            "randaugment_rate": 0.7,
        },
        {
            "name": "Aggressive MixUp",
            "dataset": "cifar10",
            "augmentation_type": "mixup",
            "mixup_alpha": 0.8,
        },
        {
            "name": "Strong RandAugment",
            "dataset": "mnist",
            "augmentation_type": "randaugment",
            "randaugment_magnitude": 0.8,
            "randaugment_rate": 0.9,
        },
    ]
    
    results = []
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print("-" * 40)
        
        success = run_augmentation_example(**example)
        results.append((example['name'], success))
        
        print(f"\nCompleted {i}/{len(examples)} examples")
        print("="*60)
    
    # Summary
    print("\n📊 Summary:")
    print("="*60)
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name}: {status}")

def print_augmentation_info():
    """Print information about available augmentation types."""
    print("\n📚 Available Augmentation Types:")
    print("="*60)
    
    info = {
        'basic': {
            'description': 'Standard augmentation with brightness, contrast, rotation, etc.',
            'best_for': 'General purpose, good starting point',
            'parameters': 'None'
        },
        'mixup': {
            'description': 'MixUp augmentation that blends images and labels',
            'best_for': 'Improving generalization, reducing overfitting',
            'parameters': 'mixup_alpha (default: 0.2)'
        },
        'cutmix': {
            'description': 'CutMix augmentation that cuts and pastes image patches',
            'best_for': 'Object detection, improving robustness',
            'parameters': 'cutmix_alpha (default: 0.5)'
        },
        'randaugment': {
            'description': 'RandAugment pipeline with multiple transformations',
            'best_for': 'State-of-the-art performance, automated augmentation',
            'parameters': 'randaugment_magnitude, randaugment_rate'
        },
        'random_choice': {
            'description': 'Random choice from a set of augmentations',
            'best_for': 'Diverse augmentation, exploration',
            'parameters': 'randaugment_magnitude'
        },
        'combined': {
            'description': 'Combined pipeline that randomly chooses between all types',
            'best_for': 'Maximum diversity, experimental setups',
            'parameters': 'All augmentation parameters'
        }
    }
    
    for aug_type, details in info.items():
        print(f"\n🔧 {aug_type.upper()}")
        print(f"   Description: {details['description']}")
        print(f"   Best for: {details['best_for']}")
        print(f"   Parameters: {details['parameters']}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--info":
        print_augmentation_info()
    else:
        main()
        print("\n💡 Run 'python augmentation_example.py --info' for detailed information about augmentation types.") 