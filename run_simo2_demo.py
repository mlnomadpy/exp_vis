#!/usr/bin/env python3
"""
SIMO2 Pretraining Demo Script

This script demonstrates how to use the SIMO2 (Self-supervised Image MOdel) pretraining method
with your existing YatCNN model architecture.

Usage examples:
    python run_simo2_demo.py --dataset cifar10 --use_simo2_pretraining
    python run_simo2_demo.py --dataset stl10 --use_simo2_pretraining --embedding_size 32
    python run_simo2_demo.py --dataset cifar100 --use_simo2_pretraining --samples_per_class 16
"""

import os
import sys
import subprocess

def run_simo2_training(dataset, embedding_size=16, samples_per_class=32, learning_rate=0.0003):
    """
    Run SIMO2 pretraining with the specified parameters.
    
    Args:
        dataset: Dataset name (e.g., 'cifar10', 'stl10', 'cifar100')
        embedding_size: Size of the embedding vectors
        samples_per_class: Number of samples per class in each batch
        learning_rate: Learning rate for training
    """
    cmd = [
        sys.executable, "src/main.py",
        "--dataset", dataset,
        "--use_simo2_pretraining",
        "--learning_rate", str(learning_rate),
        "--embedding_size", str(embedding_size),
        "--samples_per_class", str(samples_per_class),
        "--num_epochs", "100000",  # SIMO2 typically needs more epochs
        "--log_rate", "10000",     # Log every 10k steps
    ]
    
    print(f"Running SIMO2 pretraining with command:")
    print(" ".join(cmd))
    print("\n" + "="*80)
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\nSIMO2 pretraining completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nSIMO2 pretraining failed with error code {e.returncode}")
        return False

def main():
    """Main function to run SIMO2 pretraining demos."""
    
    print("🚀 SIMO2 Pretraining Demo")
    print("="*50)
    
    # Example configurations for different datasets
    demo_configs = [
        {
            "name": "CIFAR-10 (Small)",
            "dataset": "cifar10",
            "embedding_size": 16,
            "samples_per_class": 32,
            "learning_rate": 0.0003,
            "description": "Standard CIFAR-10 with 10 classes"
        },
        {
            "name": "STL-10 (Medium)",
            "dataset": "stl10", 
            "embedding_size": 32,
            "samples_per_class": 32,
            "learning_rate": 0.0003,
            "description": "STL-10 with 96x96 images, 10 classes"
        },
        {
            "name": "CIFAR-100 (Large)",
            "dataset": "cifar100",
            "embedding_size": 64,
            "samples_per_class": 16,
            "learning_rate": 0.0001,
            "description": "CIFAR-100 with 100 classes, smaller samples per class"
        }
    ]
    
    print("Available demo configurations:")
    for i, config in enumerate(demo_configs):
        print(f"{i+1}. {config['name']}")
        print(f"   Dataset: {config['dataset']}")
        print(f"   Embedding size: {config['embedding_size']}")
        print(f"   Samples per class: {config['samples_per_class']}")
        print(f"   Learning rate: {config['learning_rate']}")
        print(f"   Description: {config['description']}")
        print()
    
    # Ask user which demo to run
    while True:
        try:
            choice = input("Enter the number of the demo to run (1-3), or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                print("Exiting...")
                return
            choice = int(choice)
            if 1 <= choice <= len(demo_configs):
                break
            else:
                print(f"Please enter a number between 1 and {len(demo_configs)}")
        except ValueError:
            print("Please enter a valid number")
    
    selected_config = demo_configs[choice - 1]
    
    print(f"\n🎯 Running {selected_config['name']} demo...")
    print(f"Dataset: {selected_config['dataset']}")
    print(f"Embedding size: {selected_config['embedding_size']}")
    print(f"Samples per class: {selected_config['samples_per_class']}")
    print(f"Learning rate: {selected_config['learning_rate']}")
    
    # Confirm before running
    confirm = input("\nProceed with this configuration? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Demo cancelled.")
        return
    
    # Run the training
    success = run_simo2_training(
        dataset=selected_config['dataset'],
        embedding_size=selected_config['embedding_size'],
        samples_per_class=selected_config['samples_per_class'],
        learning_rate=selected_config['learning_rate']
    )
    
    if success:
        print(f"\n✅ {selected_config['name']} demo completed successfully!")
        print(f"Check the 'models' directory for the pretrained model.")
        print(f"Check wandb for training logs and visualizations.")
    else:
        print(f"\n❌ {selected_config['name']} demo failed.")
        print("Check the error messages above for details.")

if __name__ == "__main__":
    main() 