#!/usr/bin/env python3
"""
Example script demonstrating learning rate scheduler functionality.

This script shows how to use different learning rate schedulers with the training pipeline.
"""

import subprocess
import sys

def run_example(dataset, scheduler_type, optimizer_type, learning_rate=0.01, **kwargs):
    """Run a training example with specific scheduler settings."""
    
    cmd = [
        sys.executable, "src/main.py",
        "--dataset", dataset,
        "--learning_rate", str(learning_rate),
        "--scheduler_type", scheduler_type,
        "--optimizer_type", optimizer_type,
        "--num_epochs", "10",  # Short training for demo
        "--batch_size", "128",
        "--eval_every", "50",
    ]
    
    # Add scheduler-specific parameters
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--scheduler_{key}", str(value)])
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Dataset: {dataset}")
    print(f"Scheduler: {scheduler_type}")
    print(f"Optimizer: {optimizer_type}")
    print(f"Learning Rate: {learning_rate}")
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
    """Run various scheduler examples."""
    
    print("🚀 Learning Rate Scheduler Examples")
    print("="*60)
    
    examples = [
        {
            "name": "Constant Learning Rate",
            "dataset": "cifar10",
            "scheduler_type": "constant",
            "optimizer_type": "adam",
            "learning_rate": 0.01,
        },
        {
            "name": "Cosine Decay",
            "dataset": "cifar10", 
            "scheduler_type": "cosine",
            "optimizer_type": "adam",
            "learning_rate": 0.01,
            "alpha": 0.0,
        },
        {
            "name": "Warmup Cosine",
            "dataset": "cifar10",
            "scheduler_type": "warmup_cosine", 
            "optimizer_type": "adamw",
            "learning_rate": 0.01,
            "warmup_steps": 100,
            "end_value": 0.001,
        },
        {
            "name": "Step Decay",
            "dataset": "mnist",
            "scheduler_type": "step",
            "optimizer_type": "sgd", 
            "learning_rate": 0.1,
            "step_size": 50,
            "decay_factor": 0.5,
        },
        {
            "name": "Exponential Decay",
            "dataset": "fashion_mnist",
            "scheduler_type": "exponential",
            "optimizer_type": "rmsprop",
            "learning_rate": 0.01,
            "decay_rate": 0.95,
        },
        {
            "name": "Linear Decay",
            "dataset": "cifar10",
            "scheduler_type": "linear",
            "optimizer_type": "novograd",
            "learning_rate": 0.01,
            "end_value": 0.001,
        },
        {
            "name": "Polynomial Decay",
            "dataset": "cifar10",
            "scheduler_type": "polynomial",
            "optimizer_type": "adam",
            "learning_rate": 0.01,
            "power": 2.0,
            "end_value": 0.001,
        },
    ]
    
    results = []
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print("-" * 40)
        
        success = run_example(**example)
        results.append((example['name'], success))
        
        print(f"\nCompleted {i}/{len(examples)} examples")
        print("="*60)
    
    # Summary
    print("\n📊 Summary:")
    print("="*60)
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name}: {status}")

if __name__ == "__main__":
    main() 