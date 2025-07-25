# exp_vis

Experimental Visualization and Training Framework for Computer Vision Models

## Features

- **Multiple Training Modes**: Support for autoencoder pretraining, SIMO2 pretraining, and fine-tuning
- **Comprehensive Learning Rate Schedulers**: 7 different scheduler types with full parameter control
- **Multiple Optimizers**: Adam, AdamW, SGD, NovoGrad, RMSprop support
- **Dataset Support**: CIFAR-10/100, MNIST, Fashion MNIST, ImageNet, and custom image folders
- **Advanced Analysis**: Saliency maps, kernel similarity, adversarial robustness testing
- **Wandb Integration**: Full experiment tracking and visualization
- **Flexible Configuration**: Command-line interface with extensive customization options

## Learning Rate Scheduler Support

The framework now includes comprehensive learning rate scheduler support with 7 different scheduler types:

- **Constant**: Fixed learning rate
- **Cosine**: Smooth cosine decay
- **Exponential**: Exponential decay
- **Step**: Step-wise decay
- **Warmup Cosine**: Warmup followed by cosine decay
- **Linear**: Linear decay
- **Polynomial**: Polynomial decay

See [SCHEDULER_README.md](SCHEDULER_README.md) for detailed documentation and examples.

## Quick Start

### Basic Training

```bash
# Train on CIFAR-10 with cosine scheduler
python src/main.py \
  --dataset cifar10 \
  --learning_rate 0.01 \
  --scheduler_type cosine \
  --optimizer_type adam

# Train with warmup cosine
python src/main.py \
  --dataset cifar10 \
  --learning_rate 0.01 \
  --scheduler_type warmup_cosine \
  --scheduler_warmup_steps 100
```

### Advanced Configuration

```bash
# Use the advanced script with full control
./run_advanced.sh
```

### Example Scheduler Configurations

```bash
# Run scheduler examples
python scheduler_example.py
```

## Documentation

- [SCHEDULER_README.md](SCHEDULER_README.md) - Comprehensive scheduler documentation
- [SIMO2_README.md](SIMO2_README.md) - SIMO2 pretraining documentation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

See the individual README files for detailed usage instructions and examples.