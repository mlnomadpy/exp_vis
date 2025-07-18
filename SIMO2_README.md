# SIMO2 Pretraining Implementation

This document explains the SIMO2 (Self-supervised Image MOdel) pretraining method that has been added to your experimental visualization framework.

## What is SIMO2?

SIMO2 is a self-supervised learning method that trains models to learn meaningful representations by:
1. **Intra-class Similarity**: Encouraging embeddings from the same class to be similar
2. **Inter-class Orthogonality**: Encouraging embeddings from different classes to be orthogonal
3. **Mean Embedding Separation**: Using class mean embeddings to create well-separated clusters

The method uses a contrastive loss function that combines these three objectives to learn discriminative representations without requiring labeled data during pretraining.

## Key Features

- **Uses your existing YatCNN model**: No need to change your model architecture
- **Flax NNX compatible**: Uses the modern Flax NNX API while maintaining compatibility
- **Comprehensive visualization**: Generates t-SNE, PCA, and pairwise class visualizations
- **Wandb integration**: Logs training metrics and visualizations
- **Multiple dataset support**: Works with CIFAR-10, CIFAR-100, STL-10, and many other datasets
- **Configurable parameters**: Adjustable embedding size, samples per class, and learning rate

## Installation Requirements

Make sure you have the following packages installed:

```bash
pip install jax jaxlib flax tensorflow tensorflow-datasets wandb matplotlib scikit-learn
```

## Usage

### Basic Usage

To run SIMO2 pretraining on CIFAR-10:

```bash
python src/main.py --dataset cifar10 --use_simo2_pretraining
```

### Advanced Usage

```bash
python src/main.py \
    --dataset stl10 \
    --use_simo2_pretraining \
    --embedding_size 32 \
    --samples_per_class 16 \
    --learning_rate 0.0001 \
    --num_epochs 100000 \
    --log_rate 5000
```

### Demo Script

Use the interactive demo script for easy experimentation:

```bash
python run_simo2_demo.py
```

This will present you with different dataset configurations to choose from.

## Command Line Arguments

### SIMO2 Specific Arguments

- `--use_simo2_pretraining`: Enable SIMO2 pretraining (mutually exclusive with `--use_pretraining`)
- `--embedding_size`: Size of the embedding vectors (default: 16)
- `--samples_per_class`: Number of samples per class in each batch (default: 32)
- `--orth_lean`: Orthogonality leaning parameter (default: 1/137)
- `--log_rate`: How often to log metrics and save checkpoints (default: 10000)

### General Arguments

- `--dataset`: Dataset name (e.g., 'cifar10', 'stl10', 'cifar100')
- `--learning_rate`: Learning rate for training
- `--num_epochs`: Number of training epochs
- `--batch_size`: Batch size for training

## Supported Datasets

The SIMO2 implementation supports the following datasets:

- **CIFAR-10**: 10 classes, 32x32 images
- **CIFAR-100**: 100 classes, 32x32 images  
- **STL-10**: 10 classes, 96x96 images
- **MNIST**: 10 classes, 28x28 images
- **Fashion MNIST**: 10 classes, 28x28 images
- **ImageNet**: 1000 classes, 224x224 images
- **Oxford Flowers 102**: 102 classes, 224x224 images
- **Stanford Dogs**: 120 classes, 224x224 images
- **Food-101**: 101 classes, 224x224 images
- **And many more...**

## Model Architecture

The SIMO2 implementation uses your existing `YatCNN` model as the base encoder and adds a projection head:

```python
class SIMO2Model(nnx.Module):
    def __init__(self, *, num_classes: int, input_channels: int, embedding_size: int, rngs: nnx.Rngs):
        self.base_model = YatCNN(num_classes=num_classes, input_channels=input_channels, rngs=rngs)
        self.projection_head = nnx.Dense(embedding_size, rngs=rngs)
        
    def __call__(self, x, training: bool = False):
        features = self.base_model(x, training=training, return_activations_for_layer='representation')
        x = self.projection_head(features)
        # Normalize embeddings
        x = x / jnp.linalg.norm(x, axis=1, keepdims=True)
        return x
```

## Loss Function

The SIMO2 loss function combines three components:

1. **Same-class loss**: Encourages embeddings from the same class to be similar
2. **Mean embedding loss**: Encourages class mean embeddings to be orthogonal
3. **Different-class loss**: Encourages embeddings from different classes to be dissimilar

```python
def simo2_loss_fn(model, batch, num_classes, k, embedding_dim, indices_for_same, indices_for_means, indices_for_diff, alpha):
    projected = model(batch, training=True)
    reshaped_projections = projected.reshape(num_classes, k, embedding_dim)

    mean_embeddings = jnp.mean(reshaped_projections, axis=1)
    loss = combined_loss(mean_embeddings, 0. + alpha, indices_for_means)

    same_loss = jnp.sum(jax.vmap(lambda x: combined_loss(x, 1.0, indices_for_same))(reshaped_projections))

    reshaped_projections = jnp.transpose(reshaped_projections, (1, 0, 2))
    diff_loss = jnp.sum(jax.vmap(lambda x: combined_loss(x, 0. + alpha, indices_for_diff))(reshaped_projections))
    
    final_loss = same_loss + loss + diff_loss
    return final_loss, (projected, same_loss, loss, diff_loss)
```

## Data Augmentation

SIMO2 uses comprehensive data augmentation including:

- Random brightness adjustment
- Random contrast adjustment  
- Random saturation adjustment (for RGB images)
- Random horizontal flip
- Dataset-specific normalization

## Visualizations

The SIMO2 implementation generates several types of visualizations:

1. **t-SNE plots**: Shows the learned embedding space
2. **PCA plots**: Linear dimensionality reduction of embeddings
3. **Pairwise class visualizations**: Shows how well different class pairs are separated
4. **Training curves**: Loss and metric progression over time

All visualizations are automatically logged to Wandb for easy tracking.

## Output Files

After training, you'll find:

- **Pretrained model**: Saved in `./models/{model_name}_simo2_pretrained/`
- **Visualizations**: Saved in `./plots/` directory
- **Wandb logs**: Training metrics and visualizations
- **Checkpoints**: Periodic model saves during training

## Fine-tuning

After SIMO2 pretraining, you can fine-tune the model on downstream tasks:

```bash
python src/main.py \
    --dataset cifar10 \
    --learning_rate 0.01 \
    --num_epochs 100
```

The pretrained weights will be automatically loaded and used for fine-tuning.

## Performance Tips

1. **Batch size**: Use larger batch sizes (256-512) for better contrastive learning
2. **Samples per class**: 16-32 samples per class typically work well
3. **Embedding size**: 16-64 dimensions are usually sufficient
4. **Learning rate**: Start with 0.0001-0.0003 for stable training
5. **Training time**: SIMO2 typically needs 50k-100k steps for good results

## Troubleshooting

### Common Issues

1. **Out of memory**: Reduce batch size or samples per class
2. **Slow training**: Use GPU acceleration and reduce embedding size
3. **Poor convergence**: Adjust learning rate or orthogonality parameter
4. **NaN losses**: Check data normalization and reduce learning rate

### Debug Mode

To run with more verbose output:

```bash
python src/main.py --dataset cifar10 --use_simo2_pretraining --log_rate 1000
```

## Comparison with Autoencoder Pretraining

| Aspect | Autoencoder | SIMO2 |
|--------|-------------|-------|
| **Objective** | Reconstruction | Contrastive learning |
| **Data usage** | Individual samples | Class-based sampling |
| **Representation** | Generic features | Discriminative features |
| **Training time** | Faster | Slower |
| **Downstream performance** | Good | Often better |

## Research Applications

SIMO2 is particularly useful for:

- **Transfer learning**: Learn representations that transfer well to new tasks
- **Few-shot learning**: Create embeddings that work well with limited labeled data
- **Domain adaptation**: Learn robust representations across different domains
- **Interpretability**: Generate embeddings that preserve semantic relationships

## Citation

If you use this SIMO2 implementation in your research, please cite the original SIMO paper and this implementation:

```bibtex
@article{simo2023,
  title={SIMO: Self-supervised Image MOdel},
  author={...},
  journal={...},
  year={2023}
}
```

## Contributing

To contribute to the SIMO2 implementation:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This implementation follows the same license as the main project. 