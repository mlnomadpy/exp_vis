# Learning Rate Scheduler Support

This document describes the comprehensive learning rate scheduler functionality added to the training pipeline.

## Overview

The training pipeline now supports multiple learning rate schedulers and optimizers, allowing for flexible training configurations. Schedulers can be used for all training phases: autoencoder pretraining, SIMO2 pretraining, and fine-tuning.

## Supported Schedulers

### 1. Constant Learning Rate
- **Type**: `constant`
- **Description**: Maintains a fixed learning rate throughout training
- **Parameters**: None
- **Use Case**: Baseline comparison, when you want to maintain consistent learning rate

### 2. Cosine Decay
- **Type**: `cosine`
- **Description**: Smoothly decreases learning rate following a cosine curve
- **Parameters**:
  - `alpha`: Minimum learning rate factor (default: 0.0)
- **Use Case**: General purpose, often provides good convergence

### 3. Exponential Decay
- **Type**: `exponential`
- **Description**: Exponentially decreases learning rate
- **Parameters**:
  - `decay_rate`: Decay factor (default: 0.1)
  - `transition_begin`: Step to start decay (default: 0)
- **Use Case**: When you need rapid initial learning followed by fine-tuning

### 4. Step Decay
- **Type**: `step`
- **Description**: Reduces learning rate by a factor at specific steps
- **Parameters**:
  - `step_size`: Steps between decay (default: total_steps // 3)
  - `decay_factor`: Factor to multiply LR by (default: 0.1)
- **Use Case**: Traditional approach, good for many deep learning tasks

### 5. Warmup Cosine
- **Type**: `warmup_cosine`
- **Description**: Starts with warmup, then follows cosine decay
- **Parameters**:
  - `warmup_steps`: Number of warmup steps (default: total_steps // 10)
  - `end_value`: Final learning rate (default: 0.0)
- **Use Case**: Transformer-style training, when you need initial warmup

### 6. Linear Decay
- **Type**: `linear`
- **Description**: Linearly decreases learning rate
- **Parameters**:
  - `end_value`: Final learning rate (default: learning_rate * 0.01)
- **Use Case**: Simple linear reduction, good for many tasks

### 7. Polynomial Decay
- **Type**: `polynomial`
- **Description**: Decreases learning rate following a polynomial curve
- **Parameters**:
  - `power`: Polynomial power (default: 1.0)
  - `end_value`: Final learning rate (default: learning_rate * 0.01)
- **Use Case**: When you need more control over decay shape

## Supported Optimizers

- **Adam**: Adaptive learning rate optimizer with momentum
- **AdamW**: Adam with weight decay correction
- **SGD**: Stochastic gradient descent with optional momentum
- **NovoGrad**: Adaptive gradient method with layer-wise normalization
- **RMSprop**: Root mean square propagation

## Usage

### Command Line Interface

```bash
# Basic usage with cosine scheduler
python src/main.py \
  --dataset cifar10 \
  --learning_rate 0.01 \
  --scheduler_type cosine \
  --optimizer_type adam \
  --scheduler_alpha 0.0

# Warmup cosine with custom parameters
python src/main.py \
  --dataset cifar10 \
  --learning_rate 0.01 \
  --scheduler_type warmup_cosine \
  --optimizer_type adamw \
  --scheduler_warmup_steps 100 \
  --scheduler_end_value 0.001

# Step decay with custom step size
python src/main.py \
  --dataset mnist \
  --learning_rate 0.1 \
  --scheduler_type step \
  --optimizer_type sgd \
  --scheduler_step_size 50 \
  --scheduler_decay_factor 0.5
```

### Using the Advanced Script

Edit `run_advanced.sh` to configure scheduler settings:

```bash
# Learning rate scheduler configuration
SCHEDULER_TYPE="cosine"           # Options: constant, cosine, exponential, step, warmup_cosine, linear, polynomial
OPTIMIZER_TYPE="novograd"         # Options: adam, adamw, sgd, novograd, rmsprop
SCHEDULER_ALPHA=0.0               # For cosine scheduler
SCHEDULER_DECAY_RATE=0.1          # For exponential scheduler
SCHEDULER_STEP_SIZE=100           # For step scheduler
SCHEDULER_DECAY_FACTOR=0.5        # For step scheduler
SCHEDULER_WARMUP_STEPS=100        # For warmup_cosine scheduler
SCHEDULER_END_VALUE=0.001         # For various schedulers
SCHEDULER_POWER=1.0               # For polynomial scheduler
```

Then run:
```bash
./run_advanced.sh
```

### Example Script

Run the example script to test different scheduler configurations:

```bash
python scheduler_example.py
```

This will run training with various scheduler types on different datasets.

## Configuration in Dataset Configs

You can also set default scheduler configurations in the dataset configs in `src/main.py`:

```python
'cifar10': {
    # ... other config ...
    'scheduler_type': 'cosine',
    'optimizer_type': 'novograd',
    'scheduler_params': {
        'alpha': 0.0,
    },
},
```

## Monitoring

### Wandb Integration

The scheduler automatically logs learning rate changes to Wandb:

- `learning_rate`: Current learning rate at each step
- `scheduler_type`: Type of scheduler being used
- `optimizer_type`: Type of optimizer being used
- `scheduler_params`: Scheduler-specific parameters

### Console Output

During training, you'll see scheduler information printed:

```
🔧 Optimizer: novograd
🔧 Scheduler: cosine
🔧 Total steps: 10000
```

## Best Practices

### 1. Scheduler Selection

- **Cosine**: Good general-purpose choice for most tasks
- **Warmup Cosine**: Use when training from scratch or with large models
- **Step**: Traditional choice, works well for many tasks
- **Exponential**: Use when you need rapid initial learning
- **Linear/Polynomial**: Use when you need simple, predictable decay

### 2. Parameter Tuning

- **Total Steps**: Should match your actual training duration
- **Warmup Steps**: Typically 5-10% of total steps
- **Decay Rate**: 0.1-0.95 depending on how aggressive you want decay
- **End Value**: Usually 1-10% of initial learning rate

### 3. Optimizer Pairing

- **Adam/AdamW**: Work well with most schedulers
- **SGD**: Often benefits from step or cosine decay
- **NovoGrad**: Good with cosine or warmup cosine
- **RMSprop**: Works well with exponential or step decay

### 4. Dataset Considerations

- **Small datasets**: Use shorter schedules or constant LR
- **Large datasets**: Use longer schedules with warmup
- **Transfer learning**: Often benefit from warmup cosine

## Examples

### CIFAR-10 Training

```bash
# Standard training with cosine decay
python src/main.py \
  --dataset cifar10 \
  --learning_rate 0.01 \
  --scheduler_type cosine \
  --optimizer_type adam \
  --num_epochs 100

# With warmup for better convergence
python src/main.py \
  --dataset cifar10 \
  --learning_rate 0.01 \
  --scheduler_type warmup_cosine \
  --optimizer_type adamw \
  --scheduler_warmup_steps 500 \
  --num_epochs 100
```

### MNIST Training

```bash
# Simple step decay
python src/main.py \
  --dataset mnist \
  --learning_rate 0.1 \
  --scheduler_type step \
  --optimizer_type sgd \
  --scheduler_step_size 30 \
  --scheduler_decay_factor 0.5 \
  --num_epochs 50
```

### Custom Dataset

```bash
# For a custom image folder
python src/main.py \
  --dataset /path/to/your/images \
  --learning_rate 0.001 \
  --scheduler_type exponential \
  --optimizer_type rmsprop \
  --scheduler_decay_rate 0.95 \
  --num_epochs 200
```

## Troubleshooting

### Common Issues

1. **Learning rate too high**: Reduce initial learning rate or use warmup
2. **Learning rate too low**: Increase initial learning rate or use slower decay
3. **Poor convergence**: Try warmup cosine or adjust decay parameters
4. **Overfitting**: Use faster decay or reduce learning rate

### Debugging

- Check the console output for scheduler configuration
- Monitor learning rate in Wandb
- Compare different scheduler types on your dataset
- Use the example script to test configurations

## Advanced Usage

### Custom Scheduler Parameters

You can fine-tune scheduler behavior by adjusting parameters:

```bash
# Aggressive exponential decay
python src/main.py \
  --scheduler_type exponential \
  --scheduler_decay_rate 0.8 \
  --scheduler_transition_begin 100

# Gentle polynomial decay
python src/main.py \
  --scheduler_type polynomial \
  --scheduler_power 0.5 \
  --scheduler_end_value 0.01
```

### Different Schedulers for Different Phases

You can use different schedulers for pretraining and fine-tuning by modifying the dataset config:

```python
# Different configs for different phases
'pretrain_config': {
    'scheduler_type': 'cosine',
    'optimizer_type': 'novograd',
},
'finetune_config': {
    'scheduler_type': 'warmup_cosine',
    'optimizer_type': 'adamw',
},
```

This comprehensive scheduler support allows you to experiment with different learning rate strategies and find the optimal configuration for your specific use case. 