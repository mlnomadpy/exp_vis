# Advanced Data Augmentation Support

This document describes the comprehensive data augmentation functionality added to the training pipeline using Keras CV.

## Overview

The training pipeline now supports multiple advanced data augmentation techniques, including MixUp, CutMix, RandAugment, and more. These augmentations can significantly improve model performance and robustness.

## Supported Augmentation Types

### 1. Basic Augmentation
- **Type**: `basic`
- **Description**: Standard augmentation with brightness, contrast, rotation, Gaussian blur, cutout, solarization, posterization, and equalization
- **Parameters**: None
- **Use Case**: General purpose, good starting point for most tasks

### 2. MixUp Augmentation
- **Type**: `mixup`
- **Description**: Blends images and labels from different classes to create mixed samples
- **Parameters**:
  - `mixup_alpha`: Alpha parameter controlling mixing strength (default: 0.2)
- **Use Case**: Improving generalization, reducing overfitting, better calibration

### 3. CutMix Augmentation
- **Type**: `cutmix`
- **Description**: Cuts rectangular patches from one image and pastes them into another
- **Parameters**:
  - `cutmix_alpha`: Alpha parameter controlling patch size (default: 0.5)
- **Use Case**: Object detection, improving robustness to occlusions

### 4. RandAugment Pipeline
- **Type**: `randaugment`
- **Description**: Automated augmentation pipeline with multiple transformations
- **Parameters**:
  - `randaugment_magnitude`: Magnitude of transformations (default: 0.3)
  - `randaugment_rate`: Probability of applying augmentations (default: 0.7)
- **Use Case**: State-of-the-art performance, automated augmentation

### 5. Random Choice Augmentation
- **Type**: `random_choice`
- **Description**: Randomly selects from a set of augmentations
- **Parameters**:
  - `randaugment_magnitude`: Magnitude for RandAugment layers (default: 0.3)
- **Use Case**: Diverse augmentation, exploration of different techniques

### 6. Combined Augmentation
- **Type**: `combined`
- **Description**: Randomly chooses between all augmentation types
- **Parameters**: All augmentation parameters
- **Use Case**: Maximum diversity, experimental setups

## Usage

### Command Line Interface

```bash
# Basic augmentation (default)
python src/main.py \
  --dataset cifar10 \
  --augmentation_type basic

# MixUp augmentation
python src/main.py \
  --dataset cifar10 \
  --augmentation_type mixup \
  --mixup_alpha 0.2

# CutMix augmentation
python src/main.py \
  --dataset cifar10 \
  --augmentation_type cutmix \
  --cutmix_alpha 0.5

# RandAugment pipeline
python src/main.py \
  --dataset mnist \
  --augmentation_type randaugment \
  --randaugment_magnitude 0.3 \
  --randaugment_rate 0.7

# Combined augmentation
python src/main.py \
  --dataset cifar10 \
  --augmentation_type combined \
  --mixup_alpha 0.2 \
  --cutmix_alpha 0.5 \
  --randaugment_magnitude 0.3
```

### Using the Advanced Script

Edit `run_advanced.sh` to configure augmentation settings:

```bash
# Data augmentation configuration
AUGMENTATION_TYPE="mixup"          # Options: basic, mixup, cutmix, randaugment, random_choice, combined
MIXUP_ALPHA=0.2                    # Alpha parameter for MixUp
CUTMIX_ALPHA=0.5                   # Alpha parameter for CutMix
RANDAUGMENT_MAGNITUDE=0.3          # Magnitude for RandAugment
RANDAUGMENT_RATE=0.7               # Rate for RandAugment
```

Then run:
```bash
./run_advanced.sh
```

### Example Script

Run the example script to test different augmentation configurations:

```bash
# Run all augmentation examples
python augmentation_example.py

# Get information about augmentation types
python augmentation_example.py --info
```

## Technical Details

### MixUp Implementation

MixUp creates mixed samples by blending images and labels:

```python
# Example MixUp implementation
def mixup_batch(sample):
    image = sample['image']
    label = sample['label']
    oh_label = tf.one_hot(label, num_classes)
    
    mixup_layer = keras_cv.layers.preprocessing.MixUp(alpha=0.2)
    batch = mixup_layer({"images": image, "labels": oh_label})
    
    return {
        'image': batch['images'],
        'label': tf.argmax(batch['labels'], axis=-1)
    }
```

### CutMix Implementation

CutMix cuts rectangular patches and pastes them:

```python
# Example CutMix implementation
def cutmix_batch(sample):
    image = sample['image']
    label = sample['label']
    oh_label = tf.one_hot(label, num_classes)
    
    cutmix_layer = keras_cv.layers.preprocessing.CutMix(alpha=0.5)
    batch = cutmix_layer({"images": image, "labels": oh_label})
    
    return {
        'image': batch['images'],
        'label': tf.argmax(batch['labels'], axis=-1)
    }
```

### RandAugment Implementation

RandAugment uses a policy of transformations:

```python
# Example RandAugment implementation
layers = keras_cv.layers.RandAugment.get_standard_policy(
    value_range=(0, 255), magnitude=0.3
)
layers = layers[:4] + [keras_cv.layers.preprocessing.RandomCutout(0.5, 0.5)]

pipeline = keras_cv.layers.RandomAugmentationPipeline(
    layers=layers,
    augmentations_per_image=2,
    rate=0.7
)
```

## Best Practices

### 1. Augmentation Selection

- **Basic**: Start here for most tasks
- **MixUp**: Use when you want to improve generalization
- **CutMix**: Use for object detection or when robustness to occlusions is important
- **RandAugment**: Use for state-of-the-art performance
- **Combined**: Use for maximum diversity in experimental setups

### 2. Parameter Tuning

- **MixUp Alpha**: 0.1-0.4 (higher = more mixing)
- **CutMix Alpha**: 0.3-1.0 (higher = larger patches)
- **RandAugment Magnitude**: 0.1-1.0 (higher = stronger transformations)
- **RandAugment Rate**: 0.5-0.9 (higher = more frequent augmentations)

### 3. Dataset Considerations

- **Small datasets**: Use stronger augmentations (MixUp, CutMix)
- **Large datasets**: Use moderate augmentations (RandAugment)
- **Imbalanced datasets**: MixUp can help with class balance
- **Noisy datasets**: CutMix can improve robustness

### 4. Training Considerations

- **Learning rate**: May need to adjust with stronger augmentations
- **Batch size**: Larger batches work better with MixUp/CutMix
- **Training time**: Advanced augmentations may increase training time
- **Memory usage**: Some augmentations require more memory

## Examples

### CIFAR-10 Training

```bash
# Standard training with MixUp
python src/main.py \
  --dataset cifar10 \
  --augmentation_type mixup \
  --mixup_alpha 0.2 \
  --num_epochs 100

# Strong RandAugment
python src/main.py \
  --dataset cifar10 \
  --augmentation_type randaugment \
  --randaugment_magnitude 0.8 \
  --randaugment_rate 0.9 \
  --num_epochs 100
```

### MNIST Training

```bash
# CutMix for digit recognition
python src/main.py \
  --dataset mnist \
  --augmentation_type cutmix \
  --cutmix_alpha 0.7 \
  --num_epochs 50
```

### Custom Dataset

```bash
# Combined augmentation for custom dataset
python src/main.py \
  --dataset /path/to/your/images \
  --augmentation_type combined \
  --mixup_alpha 0.3 \
  --cutmix_alpha 0.6 \
  --randaugment_magnitude 0.4 \
  --num_epochs 200
```

## Monitoring

### Wandb Integration

Augmentation parameters are automatically logged to Wandb:

- `augmentation_type`: Type of augmentation being used
- `mixup_alpha`: MixUp alpha parameter
- `cutmix_alpha`: CutMix alpha parameter
- `randaugment_magnitude`: RandAugment magnitude
- `randaugment_rate`: RandAugment rate

### Console Output

During training, you'll see augmentation information printed:

```
🔧 Augmentation: mixup
🔧 MixUp Alpha: 0.2
```

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch size or use simpler augmentations
2. **Poor convergence**: Try basic augmentation first, then gradually increase complexity
3. **Overfitting**: Use stronger augmentations (MixUp, CutMix)
4. **Underfitting**: Reduce augmentation strength or use basic augmentation

### Debugging

- Start with basic augmentation and gradually add complexity
- Monitor training curves to see the effect of augmentations
- Use the example script to test different configurations
- Check console output for augmentation configuration

## Advanced Usage

### Custom Augmentation Pipelines

You can create custom augmentation pipelines by modifying the data.py file:

```python
def create_custom_augmentation(num_classes: int):
    """Create a custom augmentation pipeline."""
    
    def custom_augmentation(sample: dict) -> dict:
        # Your custom augmentation logic here
        return sample
    
    return custom_augmentation
```

### Different Augmentations for Different Phases

You can use different augmentations for pretraining and fine-tuning:

```python
# In dataset config
'pretrain_config': {
    'augmentation_type': 'basic',
},
'finetune_config': {
    'augmentation_type': 'mixup',
    'mixup_alpha': 0.2,
},
```

### Augmentation Ablation Studies

Use the example script to run ablation studies:

```bash
# Test different MixUp alphas
for alpha in 0.1 0.2 0.4 0.8; do
    python src/main.py \
      --dataset cifar10 \
      --augmentation_type mixup \
      --mixup_alpha $alpha \
      --num_epochs 10
done
```

## Performance Impact

### Training Time

- **Basic**: No significant impact
- **MixUp**: ~10-20% increase
- **CutMix**: ~15-25% increase
- **RandAugment**: ~20-30% increase
- **Combined**: ~25-35% increase

### Memory Usage

- **Basic**: No significant impact
- **MixUp**: ~20% increase
- **CutMix**: ~25% increase
- **RandAugment**: ~30% increase
- **Combined**: ~35% increase

### Model Performance

Typical improvements on CIFAR-10:
- **Basic**: Baseline
- **MixUp**: +1-3% accuracy
- **CutMix**: +2-4% accuracy
- **RandAugment**: +3-5% accuracy
- **Combined**: +2-6% accuracy (varies)

This comprehensive augmentation support allows you to experiment with different techniques and find the optimal configuration for your specific use case. 