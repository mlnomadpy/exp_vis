# ==============================================================================
# train.py -- Training and Evaluation Logic
# ==============================================================================
import typing as tp
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm
import os
import orbax.checkpoint as orbax
from models import YatCNN, ConvAutoencoder
from data import create_image_folder_dataset, get_image_processor, get_tfds_processor, augment_for_pretraining, augment_for_finetuning
import tensorflow_datasets as tfds
import tensorflow as tf
from logger import log_metrics
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import wandb
from functools import partial

from analysis import generate_pairwise_visualizations, compute_and_visualize_val_embeddings

# SIMO2 specific imports and functions
def combined_loss(embeddings, label_batch, indices, epsilon=1/137):
    """
    Optimized implementation of SimO contrastive loss.
    
    Args:
        embeddings: normalized embeddings [batch_size, embedding_dim]
        label_batch: 1 if similar pairs, 0 if dissimilar pairs
        indices: tuple of arrays indicating which pairs to compare
        epsilon: small value for numerical stability
    
    Returns:
        Average loss over all pairs
    """
    # Extract pairs
    e1 = embeddings[indices[0]]
    e2 = embeddings[indices[1]]
    
    # Compute dot products and squared distances efficiently
    dot_products = jnp.sum(e1 * e2, axis=1)
    dot_product_squared = dot_products ** 2
    
    # More efficient squared distance computation
    squared_distances = jnp.sum((e1 - e2) ** 2, axis=1)
    
    # Compute loss based on similarity labels
    # For similar pairs (label=1): minimize squared distance and maximize dot product
    # For dissimilar pairs (label=0): maximize squared distance and minimize dot product
    loss = jnp.exp((1 - 2 * label_batch) * jnp.log1p(dot_product_squared / (epsilon + squared_distances)))
    # Average loss over all pairs
    total_loss = jnp.sum(loss)
    return total_loss

def clip_grads(grads, max_norm):
    """Clip gradients to prevent exploding gradients."""
    norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
    clip_coef = jnp.minimum(max_norm / (norm + 1e-6), 1.0)
    return jax.tree_util.tree_map(lambda g: g * clip_coef, grads)

def simo2_loss_fn(model, batch, num_classes, k, embedding_dim, indices_for_same, indices_for_means, indices_for_diff, alpha):
    """SIMO2 loss function."""
    projected = model(batch, training=True)
    reshaped_projections = projected.reshape(num_classes, k, embedding_dim)

    mean_embeddings = jnp.mean(reshaped_projections, axis=1)
    loss = combined_loss(mean_embeddings, 0. + alpha, indices_for_means)

    # We don't need the orthogonality leaning for the similarity
    same_loss = jnp.sum(jax.vmap(lambda x: combined_loss(x, 1.0, indices_for_same))(reshaped_projections))

    reshaped_projections = jnp.transpose(reshaped_projections, (1, 0, 2))

    # Compute diff_loss using vmap
    diff_loss = jnp.sum(jax.vmap(lambda x: combined_loss(x, 0. + alpha, indices_for_diff))(reshaped_projections))
    final_loss = same_loss + loss + diff_loss
    return final_loss, (projected, same_loss, loss, diff_loss)

@nnx.jit(static_argnames=['num_classes', 'k', 'embedding_dim', 'alpha'])
def simo2_train_step(model, optimizer, batch, num_classes, k, embedding_dim, indices_for_same, indices_for_means, indices_for_diff, alpha=0.001, max_grad_norm=1.0):
    """SIMO2 training step."""
    def loss_fn_wrapped(m, b):
        return simo2_loss_fn(m, b, num_classes, k, embedding_dim, indices_for_same, indices_for_means, indices_for_diff, alpha)
    
    grad_fn = nnx.value_and_grad(loss_fn_wrapped, has_aux=True)
    (loss, (projected, same_loss, mean_loss, diff_loss)), grads = grad_fn(model, batch)
    
    clipped_grads = clip_grads(grads, max_grad_norm)
    optimizer.update(clipped_grads)
    
    return loss, same_loss, mean_loss, diff_loss

def create_simo2_batch_from_dataset(dataset, batch_size, samples_per_class, num_classes, key):
    """
    Create a batch for SIMO2 training using the existing dataset infrastructure.
    
    Args:
        dataset: TensorFlow dataset
        batch_size: Size of the batch
        samples_per_class: Number of samples per class
        num_classes: Number of classes
        key: PRNG key
        
    Returns:
        Tuple of (batch_images, batch_labels)
    """
    key1, key2 = jax.random.split(key)
    
    # Determine how many classes to sample
    classes_needed = batch_size // samples_per_class
    
    # Ensure we don't try to sample more classes than available
    classes_needed = min(classes_needed, num_classes)
    
    # For datasets with many classes, limit classes per batch
    if num_classes > 20 and classes_needed > 20:
        classes_needed = min(classes_needed, 20)
    
    # Sample classes
    sampled_classes = jax.random.choice(key1, jnp.arange(num_classes), 
                                       shape=(classes_needed,), replace=False)
    
    # Convert dataset to numpy for easier manipulation
    dataset_np = list(dataset.as_numpy_iterator())
    images = np.array([item['image'] for item in dataset_np])
    labels = np.array([item['label'] for item in dataset_np])
    
    batch_images = []
    batch_labels = []
    
    # Sample samples_per_class examples from each class
    for i, class_idx in enumerate(sampled_classes):
        # Get the class index as a Python value
        class_idx_py = class_idx.item()
        
        # Get data for this class
        class_indices = np.where(labels == class_idx_py)[0]
        class_images = images[class_indices]
        
        # Handle empty class data
        if len(class_images) == 0:
            print(f"Warning: Class {class_idx_py} has no samples. Skipping.")
            continue
        
        # Select samples_per_class random samples
        if len(class_images) >= samples_per_class:
            # Fold key to get different random numbers for each class
            class_key = jax.random.fold_in(key2, i)
            indices = jax.random.choice(class_key, jnp.arange(len(class_images)), 
                                      shape=(samples_per_class,), replace=False)
            selected_samples = class_images[indices]
        else:
            # If not enough samples, repeat some
            class_key = jax.random.fold_in(key2, i)
            indices = jax.random.choice(class_key, jnp.arange(len(class_images)), 
                                      shape=(samples_per_class,), replace=True)
            selected_samples = class_images[indices]
        
        batch_images.append(selected_samples)
        batch_labels.extend([class_idx_py] * samples_per_class)
    
    # Handle the case where we didn't get any valid samples
    if len(batch_images) == 0:
        print("Warning: No valid samples in batch. Creating dummy batch.")
        # Create a dummy batch with zeros
        dummy_shape = (batch_size,) + images.shape[1:]
        batch_images = jnp.zeros(dummy_shape, dtype=jnp.float32)
        batch_labels = jnp.zeros(batch_size, dtype=jnp.int32)
    else:
        # Concatenate all samples into a single batch
        batch_images = jnp.concatenate(batch_images, axis=0)
        batch_labels = jnp.array(batch_labels)
        
        # If we didn't get enough samples, pad the batch
        if len(batch_images) < batch_size:
            print(f"Warning: Only got {len(batch_images)} samples, padding to {batch_size}")
            # Repeat samples to reach batch_size
            indices = np.random.choice(len(batch_images), batch_size - len(batch_images), replace=True)
            pad_images = batch_images[indices]
            pad_labels = batch_labels[indices]
            batch_images = jnp.concatenate([batch_images, pad_images], axis=0)
            batch_labels = jnp.concatenate([batch_labels, pad_labels], axis=0)
    
    return batch_images, batch_labels

def _pretrain_simo2_loop(
    model_name: str,
    dataset_name: str,
    rng_seed: int,
    learning_rate: float,
    optimizer_constructor: tp.Callable,
    dataset_config: dict,
    fallback_configs: dict,
):
    """
    SIMO2 pretraining loop.
    
    This implements the SIMO (Self-supervised Image MOdel) pretraining method
    using the existing data infrastructure and your YatCNN model.
    """
    print(f"\n🚀 Starting SIMO2 Pretraining for {model_name} on {dataset_name}...")
    
    # Setup wandb
    wandb.init(project=dataset_config.get('project_name', 'simo2-pretraining'), 
               tags=['cl', 'simo2'], 
               config=dataset_config)
    dataset_config['workdir'] = wandb.run.dir
    
    # Set default values for SIMO2
    embedding_size = dataset_config.get('embedding_size', 16)
    batch_size = dataset_config.get('batch_size', 256)
    samples_per_class = dataset_config.get('samples_per_class', 32)
    num_epochs = dataset_config.get('num_epochs', 100000)
    orth_lean = dataset_config.get('orth_lean', 1/137)
    log_rate = dataset_config.get('log_rate', 10000)
    
    # Load data using existing infrastructure
    is_path = os.path.isdir(dataset_name)
    if is_path:
        # Custom folder dataset
        train_ds, val_ds, class_names, train_size = create_image_folder_dataset(
            dataset_name, validation_split=0.1, seed=42
        )
        processor = get_image_processor(
            image_size=dataset_config.get('input_dim', (64, 64)), 
            num_channels=dataset_config.get('input_channels', 3)
        )
        train_ds = train_ds.map(processor, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(processor, num_parallel_calls=tf.data.AUTOTUNE)
        num_classes = len(class_names)
    else:
        # TFDS dataset
        (train_ds, val_ds), ds_info = tfds.load(
            dataset_name,
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=False,
            with_info=True,
        )
        num_classes = ds_info.features[dataset_config['label_key']].num_classes
        class_names = ds_info.features[dataset_config['label_key']].names
        train_size = ds_info.splits['train'].num_examples
        
        processor = get_tfds_processor(
            dataset_config.get('input_dim', (64, 64)), 
            dataset_config['image_key'], 
            dataset_config['label_key']
        )
        train_ds = train_ds.map(processor, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(processor, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Update config with actual values
    dataset_config['num_classes'] = num_classes
    dataset_config['dataset'] = dataset_name
    
    # Calculate number of classes represented in a batch
    batch_classes = min(batch_size // samples_per_class, num_classes)
    print(f"Each batch will contain {batch_classes} different classes with {samples_per_class} samples each")
    
    # Prepare indices for loss computation
    indices_for_same = jnp.triu_indices(samples_per_class, 1)
    indices_for_means = jnp.triu_indices(batch_classes, 1)
    indices_for_diff = jnp.triu_indices(batch_classes, 1)
    
    # Create SIMO2 model with projection head
    # We'll use the existing YatCNN as the base model and add a projection head
    from models import YatCNN
    
    # Create a SIMO2 model that includes the base model and projection head
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
    
    # Initialize model
    model = SIMO2Model(
        num_classes=num_classes, 
        input_channels=dataset_config.get('input_channels', 3), 
        embedding_size=embedding_size,
        rngs=nnx.Rngs(rng_seed)
    )
    
    # Setup optimizer
    optimizer = nnx.Optimizer(model, optimizer_constructor(learning_rate))
    
    # Training loop
    key = jax.random.PRNGKey(rng_seed)
    
    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        
        # Create batch for training using existing dataset
        batch, labels = create_simo2_batch_from_dataset(
            train_ds, 
            batch_size, 
            samples_per_class, 
            num_classes,
            subkey
        )
        
        # Perform training step
        loss, same_loss, mean_loss, diff_loss = simo2_train_step(
            model, optimizer, batch, 
            num_classes=batch_classes,
            k=samples_per_class, 
            embedding_dim=embedding_size, 
            indices_for_same=indices_for_same,
            indices_for_means=indices_for_means, 
            indices_for_diff=indices_for_diff, 
            alpha=orth_lean
        )
        
        # Log metrics
        wandb.log({
            'Training Loss': float(loss), 
            'Epoch': epoch+1, 
            'Similar Loss': float(same_loss), 
            'Mean Embedding Loss': float(mean_loss), 
            'Different Loss': float(diff_loss)
        })
        
        # Save model periodically
        if epoch % log_rate == 0 and epoch != 0:
            save_dir = os.path.abspath(f"./models/{model_name}_simo2_pretrained")
            state = nnx.state(model)
            checkpointer = orbax.PyTreeCheckpointer()
            print(f"\n💾 Saving SIMO2 pretrained model to {save_dir}...")
            checkpointer.save(save_dir, state, force=True)
    
    # Save final model
    save_dir = os.path.abspath(f"./models/{model_name}_simo2_pretrained")
    state = nnx.state(model)
    checkpointer = orbax.PyTreeCheckpointer()
    print(f"\n💾 Saving final SIMO2 pretrained model to {save_dir}...")
    checkpointer.save(save_dir, state, force=True)
    
    # Generate and visualize embeddings
    print("\n=== Generating Embedding Visualizations ===")
    embeddings, labels = compute_and_visualize_val_embeddings(model, val_ds, dataset_config)
    
    wandb.finish()
    print("SIMO2 pretraining completed!")
    
    return save_dir

def loss_fn(model, batch, num_classes: int, label_smoothing: float = 0.0):
    logits = model(batch['image'], training=True)
    one_hot_labels = jax.nn.one_hot(batch['label'], num_classes=num_classes)
    if label_smoothing > 0:
        smoothed_labels = optax.smooth_labels(one_hot_labels, alpha=label_smoothing)
    else:
        smoothed_labels = one_hot_labels
    loss = optax.softmax_cross_entropy(
        logits=logits, labels=smoothed_labels
    ).mean()
    return loss, logits

@nnx.jit(static_argnames=['num_classes', 'label_smoothing'])
def train_step(model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch, num_classes: int, label_smoothing: float):
    def loss_fn_wrapped(m, b):
        return loss_fn(m, b, num_classes=num_classes, label_smoothing=label_smoothing)
    grad_fn = nnx.value_and_grad(loss_fn_wrapped, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])
    optimizer.update(grads)

@nnx.jit(static_argnames=['num_classes'])
def eval_step(model, metrics: nnx.MultiMetric, batch, num_classes: int):
    loss, logits = loss_fn(model, batch, num_classes=num_classes, label_smoothing=0.0)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])

def autoencoder_loss_fn(model: ConvAutoencoder, batch):
    augmented_image = batch['augmented_image']
    original_image = batch['original_image']
    reconstructed_image = model(augmented_image, training=True)
    loss = jnp.mean(jnp.square(original_image - reconstructed_image))
    return loss

@nnx.jit
def pretrain_autoencoder_step(model: ConvAutoencoder, optimizer: nnx.Optimizer, batch):
    grad_fn = nnx.value_and_grad(autoencoder_loss_fn)
    loss, grads = grad_fn(model, batch)
    optimizer.update(grads)
    return loss

def compute_top_k_accuracy(logits, labels, k=5):
    """Compute top-k accuracy."""
    top_k_indices = jnp.argsort(logits, axis=-1)[:, -k:]
    labels_expanded = jnp.expand_dims(labels, axis=-1)
    correct = jnp.any(top_k_indices == labels_expanded, axis=-1)
    return jnp.mean(correct)

def compute_detailed_metrics(model, test_ds, batch_size: int, num_classes: int, class_names: list):
    """Compute comprehensive evaluation metrics."""
    print("\n📊 Computing detailed evaluation metrics...")
    
    all_predictions = []
    all_labels = []
    all_logits = []
    
    test_iter = test_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
    
    for batch in tqdm(test_iter, desc="Evaluating model"):
        logits = model(batch['image'], training=False)
        predictions = jnp.argmax(logits, axis=-1)
        
        all_predictions.extend(predictions)
        all_labels.extend(batch['label'])
        all_logits.extend(logits)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    
    # Compute metrics
    top1_accuracy = np.mean(all_predictions == all_labels)
    top5_accuracy = compute_top_k_accuracy(all_logits, all_labels, k=5)
    
    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Compute per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # Find worst performing classes
    worst_classes_idx = np.argsort(per_class_accuracy)[:5]  # Top 5 worst
    best_classes_idx = np.argsort(per_class_accuracy)[-5:]  # Top 5 best
    
    # Print results
    print(f"\n🎯 Final Evaluation Results:")
    print(f"   Top-1 Accuracy: {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)")
    print(f"   Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    print(f"\n📈 Per-Class Performance:")
    print(f"   Best performing classes:")
    for i, idx in enumerate(reversed(best_classes_idx)):
        class_name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
        print(f"     {i+1}. {class_name}: {per_class_accuracy[idx]:.4f} ({per_class_accuracy[idx]*100:.2f}%)")
    
    print(f"   Worst performing classes:")
    for i, idx in enumerate(worst_classes_idx):
        class_name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
        print(f"     {i+1}. {class_name}: {per_class_accuracy[idx]:.4f} ({per_class_accuracy[idx]*100:.2f}%)")
    
    # Log detailed metrics
    metrics_dict = {
        'final_top1_accuracy': top1_accuracy,
        'final_top5_accuracy': top5_accuracy,
        'final_precision': precision,
        'final_recall': recall,
        'final_f1_score': f1,
        'final_macro_avg_accuracy': np.mean(per_class_accuracy),
        'final_std_accuracy': np.std(per_class_accuracy),
    }
    
    # Add per-class metrics
    for i, (acc, class_name) in enumerate(zip(per_class_accuracy, class_names)):
        metrics_dict[f'class_{i}_{class_name}_accuracy'] = acc
    
    log_metrics(metrics_dict)
    
    # Save detailed metrics to file
    import json
    import datetime
    
    metrics_summary = {
        'timestamp': datetime.datetime.now().isoformat(),
        'top1_accuracy': float(top1_accuracy),
        'top5_accuracy': float(top5_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'macro_avg_accuracy': float(np.mean(per_class_accuracy)),
        'std_accuracy': float(np.std(per_class_accuracy)),
        'per_class_accuracy': {class_names[i]: float(acc) for i, acc in enumerate(per_class_accuracy)},
        'best_classes': [class_names[i] for i in reversed(best_classes_idx)],
        'worst_classes': [class_names[i] for i in worst_classes_idx],
    }
    
    # Create metrics directory if it doesn't exist
    os.makedirs('./metrics', exist_ok=True)
    
    # Save metrics to file
    metrics_file = f"./metrics/detailed_metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"💾 Detailed metrics saved to: {metrics_file}")
    
    # Generate confusion matrix visualization
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save confusion matrix plot
        cm_file = f"./metrics/confusion_matrix_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Confusion matrix saved to: {cm_file}")
        
        # Generate per-class accuracy bar plot
        plt.figure(figsize=(max(12, len(class_names)*0.3), 8))
        bars = plt.bar(range(len(class_names)), per_class_accuracy)
        plt.xlabel('Classes')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Color bars based on performance
        for i, (bar, acc) in enumerate(zip(bars, per_class_accuracy)):
            if acc >= 0.8:
                bar.set_color('green')
            elif acc >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        
        # Save per-class accuracy plot
        acc_file = f"./metrics/per_class_accuracy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(acc_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Per-class accuracy plot saved to: {acc_file}")
        
    except ImportError:
        print("⚠️  matplotlib/seaborn not available. Skipping visualization generation.")
    
    return {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class_accuracy': per_class_accuracy,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'logits': all_logits,
        'metrics_file': metrics_file
    }


def _pretrain_autoencoder_loop(
    model_name: str,
    dataset_name: str,
    rng_seed: int,
    learning_rate: float,
    optimizer_constructor: tp.Callable,
    dataset_config: dict,
    fallback_configs: dict,
):
    print(f"\n🚀 Starting Denoising Autoencoder Pretraining for {model_name} on {dataset_name}...")
    is_path = os.path.isdir(dataset_name)
    image_size = dataset_config.get('input_dim', (64, 64))
    input_channels = dataset_config.get('input_channels', 3)
    current_batch_size = dataset_config.get('pretrain_batch_size', fallback_configs['pretrain_batch_size'])
    current_num_epochs = dataset_config.get('pretrain_epochs', fallback_configs['pretrain_epochs'])
    
    # Debug: Print what config values are being used
    print(f"🔧 Pretraining config values:")
    print(f"   image_size: {image_size}")
    print(f"   input_channels: {input_channels}")
    print(f"   current_batch_size: {current_batch_size}")
    print(f"   current_num_epochs: {current_num_epochs}")
    print(f"   Full dataset_config: {dataset_config}")
    if is_path:
        train_ds, _, class_names, train_size = create_image_folder_dataset(dataset_name, validation_split=0.01, seed=42)
        processor = get_image_processor(image_size=image_size, num_channels=input_channels)
        base_train_ds = train_ds.map(processor)
    else: # TFDS
        base_train_ds, ds_info = tfds.load(
            dataset_name,
            split=dataset_config['train_split'],
            shuffle_files=True,
            as_supervised=False,
            with_info=True,
        )
        class_names = ds_info.features[dataset_config['label_key']].names
        train_size = ds_info.splits[dataset_config['train_split']].num_examples
        processor = get_tfds_processor(image_size, dataset_config['image_key'], dataset_config['label_key'])
        base_train_ds = base_train_ds.map(processor)
    def apply_augmentations(x):
        return {
            'original_image': x['image'],
            'augmented_image': augment_for_pretraining(x['image']),
            'label': x['label'] # Keep label for t-SNE
        }
    augmented_train_ds = base_train_ds.map(apply_augmentations, num_parallel_calls=tf.data.AUTOTUNE)
    autoencoder_model = ConvAutoencoder(num_classes=len(class_names), input_channels=input_channels, rngs=nnx.Rngs(rng_seed))
    steps_per_epoch = train_size // current_batch_size
    total_steps = current_num_epochs * steps_per_epoch
    schedule = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=total_steps)
    tx = optimizer_constructor(learning_rate=schedule)
    optimizer = nnx.Optimizer(autoencoder_model, tx)
    print(f"Pre-training for {current_num_epochs} epochs ({total_steps} steps)...")
    for epoch in range(current_num_epochs):
        epoch_train_iter = augmented_train_ds.shuffle(1024).batch(current_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
        pbar = tqdm(epoch_train_iter, total=steps_per_epoch, desc=f"Autoencoder Epoch {epoch + 1}/{current_num_epochs}")
        for batch_data in pbar:
            loss = pretrain_autoencoder_step(autoencoder_model, optimizer, batch_data)
            pbar.set_postfix({'reconstruction_loss': f'{loss:.6f}'})
            if jnp.isnan(loss):
                print("\n❗️ Loss is NaN. Stopping pretraining.")
                return None
    save_dir = os.path.abspath(f"./models/{model_name}_autoencoder_pretrained_encoder")
    encoder_state = nnx.state(autoencoder_model.encoder, nnx.Param)
    checkpointer = orbax.PyTreeCheckpointer()
    print(f"\n💾 Saving pretrained ENCODER state to {save_dir}...")
    checkpointer.save(save_dir, encoder_state, force=True)
    from analysis import visualize_reconstructions, visualize_tsne
    vis_iter = augmented_train_ds.batch(32).as_numpy_iterator()
    visualize_reconstructions(autoencoder_model, vis_iter, title="Denoising Autoencoder Reconstructions")
    tsne_iter = base_train_ds.batch(32).as_numpy_iterator()
    visualize_tsne(autoencoder_model.encoder, tsne_iter, class_names, title="t-SNE of Autoencoder Pretrained Embeddings")
    return save_dir

def _train_model_loop(
    model_class: tp.Type[YatCNN],
    model_name: str,
    dataset_name: str,
    rng_seed: int,
    learning_rate: float,
    optimizer_constructor: tp.Callable,
    dataset_config: dict,
    fallback_configs: dict,
    pretrained_encoder_path: tp.Optional[str] = None,
    freeze_encoder: bool = False,
):
    stage = "Fine-tuning with Pretrained Weights" if pretrained_encoder_path else "Training from Scratch"
    if freeze_encoder and pretrained_encoder_path:
        stage += " (Encoder Frozen)"
    print(f"\n🚀 Initializing {model_name} for {stage} on dataset {dataset_name}...")
    is_path = os.path.isdir(dataset_name)
    image_size = dataset_config.get('input_dim', (64, 64))
    input_channels = dataset_config.get('input_channels', 3)
    current_num_epochs = dataset_config.get('num_epochs', fallback_configs['num_epochs'])
    current_eval_every = dataset_config.get('eval_every', fallback_configs['eval_every'])
    current_batch_size = dataset_config.get('batch_size', fallback_configs['batch_size'])
    label_smooth = dataset_config.get('label_smooth', fallback_configs['label_smooth'])
    
    # Debug: Print what config values are being used
    print(f"🔧 Training config values:")
    print(f"   image_size: {image_size}")
    print(f"   input_channels: {input_channels}")
    print(f"   current_num_epochs: {current_num_epochs}")
    print(f"   current_eval_every: {current_eval_every}")
    print(f"   current_batch_size: {current_batch_size}")
    print(f"   label_smooth: {label_smooth}")
    print(f"   Full dataset_config: {dataset_config}")
    if is_path:
        split_percentage = dataset_config.get('test_split_percentage', 0.2)
        train_ds, test_ds, class_names, train_size = create_image_folder_dataset(dataset_name, validation_split=split_percentage, seed=42)
        num_classes = len(class_names)
        processor = get_image_processor(image_size=image_size, num_channels=input_channels)
        train_ds = train_ds.map(processor, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(processor, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.map(augment_for_finetuning, num_parallel_calls=tf.data.AUTOTUNE)
    else: # TFDS logic
        (train_ds, test_ds), ds_info = tfds.load(
            dataset_name,
            split=[dataset_config['train_split'], dataset_config['test_split']],
            shuffle_files=True,
            as_supervised=False,
            with_info=True,
        )
        num_classes = ds_info.features[dataset_config['label_key']].num_classes
        class_names = ds_info.features[dataset_config['label_key']].names
        train_size = ds_info.splits[dataset_config['train_split']].num_examples
        processor = get_tfds_processor(image_size, dataset_config['image_key'], dataset_config['label_key'])
        train_ds = train_ds.map(processor, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(processor, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.map(augment_for_finetuning, num_parallel_calls=tf.data.AUTOTUNE)
    model = model_class(num_classes=num_classes, input_channels=input_channels, rngs=nnx.Rngs(rng_seed))
    if pretrained_encoder_path:
        print(f"💾 Loading pretrained encoder weights from {pretrained_encoder_path}...")
        checkpointer = orbax.PyTreeCheckpointer()
        abstract_encoder_state = jax.eval_shape(lambda: nnx.state(model, nnx.Param))
        restored_params = checkpointer.restore(pretrained_encoder_path, item=abstract_encoder_state)
        nnx.update(model, restored_params)
        print("✅ Pretrained weights loaded successfully!")
    if freeze_encoder and pretrained_encoder_path:
        print("❄️ Freezing encoder weights. Only the final classification layer will be trained.")
        def path_partition_fn(path: tp.Sequence[tp.Any], value: tp.Any):
            if path and hasattr(path[0], 'key') and path[0].key == 'out_linear':
                return 'trainable'
            return 'frozen'
        params = nnx.state(model, nnx.Param)
        param_labels = jax.tree_util.tree_map_with_path(path_partition_fn, params)
        trainable_tx = optimizer_constructor(learning_rate)
        frozen_tx = optax.set_to_zero()
        tx = optax.multi_transform(
            {'trainable': trainable_tx, 'frozen': frozen_tx},
            param_labels
        )
        optimizer = nnx.Optimizer(model, tx)
    else:
        optimizer = nnx.Optimizer(model, optimizer_constructor(learning_rate))
    metrics_computer = nnx.MultiMetric(accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average('loss'))
    metrics_history = {'train_loss': [], 'train_accuracy': [], 'test_loss': [], 'test_accuracy': []}
    steps_per_epoch = train_size // current_batch_size
    total_steps = current_num_epochs * steps_per_epoch
    print(f"Starting training for {total_steps} steps...")
    global_step_counter = 0
    for epoch in range(current_num_epochs):
        epoch_train_iter = train_ds.shuffle(1024).batch(current_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
        pbar = tqdm(epoch_train_iter, total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{current_num_epochs} ({stage})")
        for batch_data in pbar:
            train_step(model, optimizer, metrics_computer, batch_data, num_classes=num_classes, label_smoothing=label_smooth)
            if global_step_counter > 0 and (global_step_counter % current_eval_every == 0 or global_step_counter >= total_steps -1):
                train_metrics = metrics_computer.compute()
                metrics_history['train_loss'].append(train_metrics['loss'])
                metrics_history['train_accuracy'].append(train_metrics['accuracy'])
                metrics_computer.reset()
                current_test_iter = test_ds.batch(current_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
                for test_batch in current_test_iter:
                    eval_step(model, metrics_computer, test_batch, num_classes)
                test_metrics = metrics_computer.compute()
                metrics_history['test_loss'].append(test_metrics['loss'])
                metrics_history['test_accuracy'].append(test_metrics['accuracy'])
                metrics_computer.reset()
                pbar.set_postfix({'Train Acc': f"{train_metrics['accuracy']:.4f}", 'Test Acc': f"{test_metrics['accuracy']:.4f}"})
                # Log progress to wandb
                log_metrics({
                    'step': global_step_counter,
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'test_loss': test_metrics['loss'],
                    'test_accuracy': test_metrics['accuracy'],
                }, step=global_step_counter)
            global_step_counter += 1
        if global_step_counter >= total_steps: break
    print(f"✅ {stage} complete on {dataset_name} after {global_step_counter} steps!")
    if metrics_history['test_accuracy']:
        print(f"    Final Test Accuracy: {metrics_history['test_accuracy'][-1]:.4f}")
    
    # Compute comprehensive evaluation metrics
    detailed_metrics = compute_detailed_metrics(model, test_ds, current_batch_size, num_classes, class_names)
    
    save_dir = os.path.abspath(f"./models/{model_name}_{dataset_name.replace('/', '_')}")
    state = nnx.state(model)
    checkpointer = orbax.PyTreeCheckpointer()
    print(f"💾 Saving final model state to {save_dir}...")
    checkpointer.save(save_dir, state, force=True)
    
    # Log training history
    for i, (train_loss, train_acc, test_loss, test_acc) in enumerate(zip(
        metrics_history['train_loss'], metrics_history['train_accuracy'],
        metrics_history['test_loss'], metrics_history['test_accuracy'])):
        log_metrics({
            'step': i,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
        }, step=i)
    
    return model, metrics_history, test_ds, class_names, detailed_metrics 