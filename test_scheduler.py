#!/usr/bin/env python3
"""
Test script to verify learning rate scheduler functionality.
"""

import jax
import jax.numpy as jnp
import optax
from src.train import create_learning_rate_schedule, create_optimizer_with_scheduler

def test_schedulers():
    """Test different scheduler types."""
    learning_rate = 0.001
    total_steps = 1000
    
    schedulers_to_test = [
        'constant',
        'cosine',
        'exponential',
        'step',
        'warmup_cosine',
        'linear',
        'polynomial'
    ]
    
    print("🧪 Testing Learning Rate Schedulers")
    print("=" * 50)
    
    for scheduler_type in schedulers_to_test:
        print(f"\n📊 Testing {scheduler_type.upper()} scheduler:")
        
        # Create scheduler
        schedule = create_learning_rate_schedule(
            scheduler_type=scheduler_type,
            learning_rate=learning_rate,
            total_steps=total_steps,
            alpha=0.1,  # For cosine decay
            decay_rate=0.1,  # For exponential
            step_size=total_steps // 3,  # For step
            decay_factor=0.1,  # For step
            warmup_steps=total_steps // 10,  # For warmup_cosine
            end_value=learning_rate * 0.01,  # For linear/polynomial
            power=1.0  # For polynomial
        )
        
        # Test at different steps
        steps_to_test = [0, total_steps // 4, total_steps // 2, 3 * total_steps // 4, total_steps - 1]
        
        for step in steps_to_test:
            lr = float(schedule(step))
            print(f"   Step {step:4d}: {lr:.6f}")
    
    print("\n" + "=" * 50)
    print("✅ Scheduler tests completed!")

def test_optimizer_integration():
    """Test optimizer integration with schedulers."""
    learning_rate = 0.001
    total_steps = 1000
    
    print("\n🔧 Testing Optimizer Integration")
    print("=" * 50)
    
    optimizer_types = ['adam', 'adamw', 'sgd', 'novograd', 'rmsprop']
    scheduler_types = ['constant', 'cosine', 'warmup_cosine']
    
    for opt_type in optimizer_types:
        for sched_type in scheduler_types:
            print(f"\n📊 Testing {opt_type.upper()} + {sched_type.upper()}:")
            
            # Create optimizer with scheduler
            optimizer_constructor = create_optimizer_with_scheduler(
                optimizer_type=opt_type,
                learning_rate=learning_rate,
                scheduler_type=sched_type,
                total_steps=total_steps,
                warmup_steps=total_steps // 10
            )
            
            # Create optimizer
            optimizer = optimizer_constructor()
            
            # Test learning rate at different steps
            steps_to_test = [0, total_steps // 4, total_steps // 2, total_steps - 1]
            
            for step in steps_to_test:
                try:
                    lr = float(optimizer.learning_rate(step))
                    print(f"   Step {step:4d}: {lr:.6f}")
                except Exception as e:
                    print(f"   Step {step:4d}: Error - {e}")
    
    print("\n" + "=" * 50)
    print("✅ Optimizer integration tests completed!")

if __name__ == "__main__":
    test_schedulers()
    test_optimizer_integration() 