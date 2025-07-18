# ==============================================================================
# data.py -- Data Loading, Preprocessing, and Augmentation
# ==============================================================================
import os
import glob
import tensorflow as tf
import numpy as np

# Try to import keras_cv and provide helpful error messages
try:
    import keras_cv
    KERAS_CV_AVAILABLE = True
except ImportError:
    print("⚠️ keras_cv not available. Advanced augmentations will be disabled.")
    print("   Install with: pip install keras-cv")
    KERAS_CV_AVAILABLE = False

# --- For Local Image Folders ---
def create_image_folder_dataset(path: str, validation_split: float, seed: int):
    class_names = sorted([d.name for d in os.scandir(path) if d.is_dir()])
    if not class_names:
        raise ValueError(f"No subdirectories found in {path}. Each subdirectory should contain images for one class.")
    class_to_index = {name: i for i, name in enumerate(class_names)}
    all_image_paths = []
    all_image_labels = []
    for class_name in class_names:
        class_dir = os.path.join(path, class_name)
        image_paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.JPEG'):
            image_paths.extend(glob.glob(os.path.join(class_dir, ext)))
        all_image_paths.extend(image_paths)
        all_image_labels.extend([class_to_index[class_name]] * len(image_paths))
    if not all_image_paths:
        raise ValueError(f"No image files found in subdirectories of {path}.")
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int32))
    image_label_ds = tf.data.Dataset.zip((path_ds, label_ds))
    dataset_size = len(all_image_paths)
    image_label_ds = image_label_ds.shuffle(buffer_size=dataset_size, seed=seed)
    val_count = int(dataset_size * validation_split)
    train_ds = image_label_ds.skip(val_count)
    test_ds = image_label_ds.take(val_count)
    train_size = dataset_size - val_count
    return train_ds, test_ds, class_names, train_size

def get_image_processor(image_size: tuple[int, int], num_channels: int):
    def process_path(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=num_channels, expand_animations=False)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0
        return {'image': img, 'label': label}
    return process_path

def get_tfds_processor(image_size: tuple[int, int], image_key: str, label_key: str):
    def _process(sample):
        img = tf.cast(sample[image_key], tf.float32)
        img = tf.image.resize(img, image_size)
        img = img / 255.0
        return {'image': img, 'label': sample[label_key]}
    return _process

# --- Basic Augmentation Functions ---
@tf.function
def augment_for_pretraining(image: tf.Tensor) -> tf.Tensor:
    """Basic augmentation for pretraining phase."""
    image = tf.image.random_brightness(image, max_delta=0.4)
    image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
    image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
    image = tf.image.random_hue(image, max_delta=0.1)
    image = tf.image.random_flip_left_right(image)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

# --- Keras CV Augmentation Layers ---
ROTATION_LAYER = tf.keras.layers.RandomRotation(
    factor=(-0.1, 0.1), fill_mode='reflect'
)

# Initialize Keras CV layers only if available
if KERAS_CV_AVAILABLE:
    try:
        GAUSSIAN_BLUR_LAYER = keras_cv.layers.RandomGaussianBlur(
            kernel_size=3, factor=(0.0, 1.0)
        )
        CUTOUT_LAYER = keras_cv.layers.RandomCutout(
            height_factor=0.2, width_factor=0.2
        )
    except AttributeError:
        print("⚠️ Some Keras CV layers not available. Using fallback augmentations.")
        GAUSSIAN_BLUR_LAYER = None
        CUTOUT_LAYER = None
else:
    GAUSSIAN_BLUR_LAYER = None
    CUTOUT_LAYER = None

# --- Advanced Augmentation Functions ---
def create_mixup_augmentation(num_classes: int, alpha: float = 0.2):
    """Create MixUp augmentation function."""
    if not KERAS_CV_AVAILABLE:
        print("⚠️ keras_cv not available. Using basic augmentation.")
        return lambda batch: batch
    
    try:
        mixup_layer = keras_cv.layers.MixUp(alpha=alpha)
    except AttributeError:
        print("⚠️ MixUp layer not available in this version of Keras CV. Using basic augmentation.")
        return lambda batch: batch
    
    def mixup_batch(batch: dict) -> dict:
        images = batch['image']
        labels = batch['label']
        # Convert to one-hot encoding
        oh_labels = tf.one_hot(labels, num_classes)
        
        # Apply MixUp
        out = mixup_layer({'images': images, 'labels': oh_labels})
        
        # Convert back to sparse labels
        mixed_labels = tf.argmax(out['labels'], axis=-1)
        
        return {
            'image': out['images'],
            'label': mixed_labels
        }
    
    return mixup_batch

def create_cutmix_augmentation(num_classes: int, alpha: float = 0.5):
    """Create CutMix augmentation function."""
    if not KERAS_CV_AVAILABLE:
        print("⚠️ keras_cv not available. Using basic augmentation.")
        return lambda batch: batch
    
    try:
        cutmix_layer = keras_cv.layers.CutMix(alpha=alpha)
    except AttributeError:
        print("⚠️ CutMix layer not available in this version of Keras CV. Using basic augmentation.")
        return lambda batch: batch
    
    def cutmix_batch(batch: dict) -> dict:
        images = batch['image']
        labels = batch['label']
        # Convert to one-hot encoding
        oh_labels = tf.one_hot(labels, num_classes)
        
        # Apply CutMix
        out = cutmix_layer({'images': images, 'labels': oh_labels})
        
        # Convert back to sparse labels
        mixed_labels = tf.argmax(out['labels'], axis=-1)
        
        return {
            'image': out['images'],
            'label': mixed_labels
        }
    
    return cutmix_batch

def create_randaugment_pipeline(
    value_range: tuple = (0, 255),
    augmentations_per_image: int = 2,
    magnitude: float = 0.3,
    magnitude_stddev: float = 0.01,
    rate: float = 0.7
):
    """Create RandAugment pipeline."""
    if not KERAS_CV_AVAILABLE:
        print("⚠️ keras_cv not available. Using basic augmentation.")
        return lambda batch: batch
    
    try:
        layers = keras_cv.layers.RandAugment.get_standard_policy(
            value_range=value_range,
            magnitude=magnitude,
            magnitude_stddev=magnitude_stddev
        )
        
        # Add additional layers
        layers = layers[:4] + [keras_cv.layers.RandomCutout(0.5, 0.5)]
    except AttributeError:
        print("⚠️ RandAugment layers not available in this version of Keras CV. Using basic augmentation.")
        return lambda batch: batch
    
    pipeline = keras_cv.layers.RandomAugmentationPipeline(
        layers=layers,
        augmentations_per_image=augmentations_per_image,
        rate=rate
    )
    
    def randaugment_batch(batch: dict) -> dict:
        images = batch['image']
        # Convert to 0-255 range for RandAugment
        images_255 = tf.cast(images * 255, tf.uint8)
        
        # Apply RandAugment
        augmented_images = pipeline(images_255, training=True)
        
        # Convert back to 0-1 range
        augmented_images = tf.cast(augmented_images, tf.float32) / 255.0
        
        return {
            'image': augmented_images,
            'label': batch['label']
        }
    
    return randaugment_batch

def create_random_choice_pipeline(
    value_range: tuple = (0, 255),
    magnitude: float = 0.3,
    magnitude_stddev: float = 0.01
):
    """Create RandomChoice augmentation pipeline."""
    if not KERAS_CV_AVAILABLE:
        print("⚠️ keras_cv not available. Using basic augmentation.")
        return lambda batch: batch
    
    try:
        layers = keras_cv.layers.RandAugment.get_standard_policy(
            value_range=value_range,
            magnitude=magnitude,
            magnitude_stddev=magnitude_stddev
        )
        
        # Add additional layers
        layers = layers[:4] + [keras_cv.layers.RandomCutout(0.5, 0.5)]
    except AttributeError:
        print("⚠️ RandomChoice layers not available in this version of Keras CV. Using basic augmentation.")
        return lambda batch: batch
    
    pipeline = keras_cv.layers.RandomChoice(layers=layers)
    
    def random_choice_batch(batch: dict) -> dict:
        images = batch['image']
        # Convert to 0-255 range for RandAugment
        images_255 = tf.cast(images * 255, tf.uint8)
        
        # Apply RandomChoice
        augmented_images = pipeline(images_255, training=True)
        
        # Convert back to 0-1 range
        augmented_images = tf.cast(augmented_images, tf.float32) / 255.0
        
        return {
            'image': augmented_images,
            'label': batch['label']
        }
    
    return random_choice_batch

def create_advanced_augmentation_pipeline(
    num_classes: int,
    augmentation_type: str = 'basic',
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 0.5,
    randaugment_magnitude: float = 0.3,
    randaugment_rate: float = 0.7
):
    """
    Create advanced augmentation pipeline with multiple techniques.
    
    Args:
        num_classes: Number of classes for one-hot encoding
        augmentation_type: Type of augmentation ('basic', 'mixup', 'cutmix', 'randaugment', 'random_choice', 'combined')
        mixup_alpha: Alpha parameter for MixUp
        cutmix_alpha: Alpha parameter for CutMix
        randaugment_magnitude: Magnitude for RandAugment
        randaugment_rate: Rate for RandAugment
    
    Returns:
        Augmentation function
    """
    
    if augmentation_type == 'basic':
        return lambda batch: batch
    
    elif augmentation_type == 'mixup':
        return create_mixup_augmentation(num_classes, mixup_alpha)
    
    elif augmentation_type == 'cutmix':
        return create_cutmix_augmentation(num_classes, cutmix_alpha)
    
    elif augmentation_type == 'randaugment':
        return create_randaugment_pipeline(
            augmentations_per_image=2,
            magnitude=randaugment_magnitude,
            rate=randaugment_rate
        )
    
    elif augmentation_type == 'random_choice':
        return create_random_choice_pipeline(magnitude=randaugment_magnitude)
    
    elif augmentation_type == 'combined':
        # Create a combined pipeline that randomly chooses between different augmentations
        mixup_fn = create_mixup_augmentation(num_classes, mixup_alpha)
        cutmix_fn = create_cutmix_augmentation(num_classes, cutmix_alpha)
        randaugment_fn = create_randaugment_pipeline(
            augmentations_per_image=2,
            magnitude=randaugment_magnitude,
            rate=randaugment_rate
        )
        
        def combined_augmentation(batch: dict) -> dict:
            # Randomly choose augmentation type
            choice = tf.random.uniform([], 0, 4, dtype=tf.int32)
            
            if choice == 0:
                return lambda batch: batch
            elif choice == 1:
                return mixup_fn
            elif choice == 2:
                return cutmix_fn
            else:
                return randaugment_fn
        
        return combined_augmentation
    
    else:
        print(f"⚠️ Unknown augmentation type '{augmentation_type}'. Using basic augmentation.")
        return lambda batch: batch

# --- Legacy Augmentation Function (for backward compatibility) ---
@tf.function
def augment_for_finetuning(sample: dict) -> dict:
    """Enhanced augmentation for fine-tuning phase with Keras CV layers."""
    image = sample['image']
    image = tf.image.random_flip_left_right(image)
    image = ROTATION_LAYER(image, training=True)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Random grayscale conversion
    if tf.random.uniform([]) > 0.9:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
    
    # Random Gaussian blur
    if GAUSSIAN_BLUR_LAYER is not None and tf.random.uniform([]) > 0.5:
        image = GAUSSIAN_BLUR_LAYER(image, training=True)
    
    # Random cutout
    if CUTOUT_LAYER is not None and tf.random.uniform([]) > 0.5:
        image = CUTOUT_LAYER(image, training=True)
    
    # Additional Keras CV augmentations
    try:
        if tf.random.uniform([]) > 0.7:
            # Random solarization
            solarize_layer = keras_cv.layers.RandomSolarization(
                value_range=(0, 1), threshold_factor=0.5
            )
            image = solarize_layer(image, training=True)
        
        if tf.random.uniform([]) > 0.7:
            # Random posterization
            posterize_layer = keras_cv.layers.RandomPosterization(
                value_range=(0, 1), factor=(4, 8)
            )
            image = posterize_layer(image, training=True)
        
        if tf.random.uniform([]) > 0.7:
            # Random equalization
            equalize_layer = keras_cv.layers.RandomEqualization(
                value_range=(0, 1)
            )
            image = equalize_layer(image, training=True)
    except AttributeError:
        # If Keras CV layers are not available, skip these augmentations
        pass
    
    augmented_image = tf.clip_by_value(image, 0.0, 1.0)
    return {**sample, 'image': augmented_image}

# --- Utility Functions ---
def get_augmentation_info():
    """Return information about available augmentation types."""
    return {
        'basic': 'Standard augmentation with brightness, contrast, rotation, etc.',
        'mixup': 'MixUp augmentation that blends images and labels',
        'cutmix': 'CutMix augmentation that cuts and pastes image patches',
        'randaugment': 'RandAugment pipeline with multiple transformations',
        'random_choice': 'Random choice from a set of augmentations',
        'combined': 'Combined pipeline that randomly chooses between all types'
    }

def get_per_image_augmentation_fn(augmentation_type: str, num_classes: int, **kwargs):
    """
    Returns a function for per-image augmentations (to be used before batching).
    """
    if augmentation_type in ['basic', 'randaugment', 'random_choice']:
        return create_advanced_augmentation_pipeline(
            num_classes=num_classes,
            augmentation_type=augmentation_type,
            **kwargs
        )
    else:
        # For batch augmentations, just return identity
        def identity(sample):
            return sample
        return identity

def get_batch_augmentation_fn(augmentation_type: str, num_classes: int, **kwargs):
    """
    Returns a function for batch augmentations (to be used after batching).
    """
    if augmentation_type == 'mixup':
        return create_mixup_augmentation(num_classes, kwargs.get('mixup_alpha', 0.2))
    elif augmentation_type == 'cutmix':
        return create_cutmix_augmentation(num_classes, kwargs.get('cutmix_alpha', 0.5))
    elif augmentation_type == 'combined':
        return create_combined_batch_augmentation(num_classes, kwargs)
    else:
        # For per-image augmentations, just return identity
        def identity(batch):
            return batch
        return identity

# --- Batch-level Combined Augmentation ---
def create_combined_batch_augmentation(num_classes, kwargs):
    mixup_fn = create_mixup_augmentation(num_classes, kwargs.get('mixup_alpha', 0.2))
    cutmix_fn = create_cutmix_augmentation(num_classes, kwargs.get('cutmix_alpha', 0.5))
    # You can add more batch-level augmentations here if needed
    def combined_batch(batch):
        choice = tf.random.uniform([], 0, 3, dtype=tf.int32)
        # 0: identity, 1: mixup, 2: cutmix
        def do_identity():
            return batch
        def do_mixup():
            return mixup_fn(batch)
        def do_cutmix():
            return cutmix_fn(batch)
        return tf.switch_case(choice, [do_identity, do_mixup, do_cutmix])
    return combined_batch

def create_augmentation_dataset(
    dataset: tf.data.Dataset,
    num_classes: int,
    augmentation_type: str = 'basic',
    **kwargs
) -> tf.data.Dataset:
    """
    Create an augmented dataset with the specified augmentation type.
    
    Args:
        dataset: Input dataset
        num_classes: Number of classes
        augmentation_type: Type of augmentation to apply
        **kwargs: Additional arguments for augmentation
    
    Returns:
        Augmented dataset
    """
    # Apply per-image augmentations
    per_image_augmentation_fn = get_per_image_augmentation_fn(augmentation_type, num_classes, **kwargs)
    dataset = dataset.map(per_image_augmentation_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply batch augmentations
    batch_augmentation_fn = get_batch_augmentation_fn(augmentation_type, num_classes, **kwargs)
    dataset = dataset.map(batch_augmentation_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset 