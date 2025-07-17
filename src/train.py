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

# The _pretrain_autoencoder_loop and _train_model_loop functions should be moved here as well, with their dependencies updated to use the new module structure.
# For brevity, only the function signatures and comments are included here. Move the full function bodies from main.py and update imports as needed.

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
    save_dir = os.path.abspath(f"./models/{model_name}_{dataset_name.replace('/', '_')}")
    state = nnx.state(model)
    checkpointer = orbax.PyTreeCheckpointer()
    print(f"💾 Saving final model state to {save_dir}...")
    checkpointer.save(save_dir, state, force=True)
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
    return model, metrics_history, test_ds, class_names 