# ==============================================================================
# data.py -- Data Loading, Preprocessing, and Augmentation
# ==============================================================================
import os
import glob
import tensorflow as tf

# Check for keras_cv availability
try:
    import keras_cv
    KERAS_CV_AVAILABLE = True
except ImportError:
    print("⚠️  keras_cv not available. Some augmentations will be disabled.")
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

@tf.function
def augment_for_pretraining(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_brightness(image, max_delta=0.4)
    image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
    image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
    image = tf.image.random_hue(image, max_delta=0.1)
    image = tf.image.random_flip_left_right(image)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

ROTATION_LAYER = tf.keras.layers.RandomRotation(
    factor=(-0.1, 0.1), fill_mode='reflect'
)

GAUSSIAN_BLUR_LAYER = keras_cv.layers.RandomGaussianBlur(
    kernel_size=3, factor=(0.0, 1.0)
) if KERAS_CV_AVAILABLE else tf.keras.layers.Lambda(lambda x: x)

CUTOUT_LAYER = tf.keras.layers.Lambda(lambda x: tf.image.central_crop(x, 0.9))

@tf.function
def augment_for_finetuning(sample: dict) -> dict:
    image = sample['image']
    image = tf.image.random_flip_left_right(image)
    image = ROTATION_LAYER(image, training=True)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    if tf.random.uniform([]) > 0.9:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
    if tf.random.uniform([]) > 0.5 and KERAS_CV_AVAILABLE:
        image = GAUSSIAN_BLUR_LAYER(image, training=True)
    if tf.random.uniform([]) > 0.5:
        image = CUTOUT_LAYER(image, training=True)
    augmented_image = tf.clip_by_value(image, 0.0, 1.0)
    return {**sample, 'image': augmented_image}

# ==============================================================================
# Advanced KerasCV Augmentations (MixUp, CutMix, RandAugment, Pipelines)
# ==============================================================================

# --- MixUp ---
def mixup_batch_fn(num_classes, alpha=0.2):
    """
    Returns a function that applies MixUp to a batch of images and labels.
    Usage:
        ds = ds.map(mixup_batch_fn(num_classes=102, alpha=0.2), num_parallel_calls=tf.data.AUTOTUNE)
    """
    if not KERAS_CV_AVAILABLE:
        print("⚠️  MixUp not available (keras_cv not installed). Returning identity function.")
        return lambda images, labels: (images, labels)
    
    try:
        mixup = keras_cv.layers.MixUp(alpha=alpha)
        def _mixup(images, labels):
            oh_labels = tf.one_hot(labels, num_classes)
            batch = mixup({"images": images, "labels": oh_labels})
            return batch["images"], batch["labels"]
        return _mixup
    except AttributeError:
        print("⚠️  MixUp layer not available in this version of keras_cv. Returning identity function.")
        return lambda images, labels: (images, labels)

# --- CutMix ---
def cutmix_batch_fn(num_classes, alpha=0.5):
    """
    Returns a function that applies CutMix to a batch of images and labels.
    Usage:
        ds = ds.map(cutmix_batch_fn(num_classes=102, alpha=0.5), num_parallel_calls=tf.data.AUTOTUNE)
    """
    if not KERAS_CV_AVAILABLE:
        print("⚠️  CutMix not available (keras_cv not installed). Returning identity function.")
        return lambda images, labels: (images, labels)
    
    try:
        cutmix = keras_cv.layers.CutMix(alpha=alpha)
        def _cutmix(images, labels):
            oh_labels = tf.one_hot(labels, num_classes)
            batch = cutmix({"images": images, "labels": oh_labels})
            return batch["images"], batch["labels"]
        return _cutmix
    except AttributeError:
        print("⚠️  CutMix layer not available in this version of keras_cv. Returning identity function.")
        return lambda images, labels: (images, labels)

# --- RandAugment as a preprocessing layer ---
def get_randaugment_layer(value_range=(0, 255), augmentations_per_image=2, magnitude=0.3):
    """
    Returns a RandAugment layer for use in preprocessing or tf.data pipelines.
    Usage:
        layer = get_randaugment_layer()
        ds = ds.map(lambda img, lbl: (layer(img), lbl))
    """
    if not KERAS_CV_AVAILABLE:
        print("⚠️  RandAugment not available (keras_cv not installed). Returning identity layer.")
        return tf.keras.layers.Lambda(lambda x: x)
    
    try:
        return keras_cv.layers.RandAugment(
            value_range=value_range,
            augmentations_per_image=augmentations_per_image,
            magnitude=magnitude
        )
    except AttributeError:
        print("⚠️  RandAugment layer not available in this version of keras_cv. Returning identity layer.")
        return tf.keras.layers.Lambda(lambda x: x)

# --- Standard RandAugment Policy ---
def get_standard_randaugment_policy(value_range=(0, 255), magnitude=0.3, magnitude_stddev=0.01):
    """
    Returns a list of standard RandAugment layers (policy).
    Usage:
        layers = get_standard_randaugment_policy()
    """
    if not KERAS_CV_AVAILABLE:
        print("⚠️  RandAugment policy not available (keras_cv not installed). Returning empty list.")
        return []
    
    try:
        return keras_cv.layers.RandAugment.get_standard_policy(
            value_range=value_range,
            magnitude=magnitude,
            magnitude_stddev=magnitude_stddev
        )
    except AttributeError:
        print("⚠️  RandAugment policy not available in this version of keras_cv. Returning empty list.")
        return []

# --- RandomAugmentationPipeline ---
def get_random_augmentation_pipeline(auglayers, augmentations_per_image=2, rate=0.7):
    """
    Returns a RandomAugmentationPipeline layer.
    Usage:
        pipeline = get_random_augmentation_pipeline(auglayers)
        ds = ds.map(lambda img, lbl: (pipeline(img), lbl))
    """
    if not KERAS_CV_AVAILABLE:
        print("⚠️  RandomAugmentationPipeline not available (keras_cv not installed). Returning identity layer.")
        return tf.keras.layers.Lambda(lambda x: x)
    
    try:
        return keras_cv.layers.RandomAugmentationPipeline(
            layers=auglayers,
            augmentations_per_image=augmentations_per_image,
            rate=rate
        )
    except AttributeError:
        print("⚠️  RandomAugmentationPipeline not available in this version of keras_cv. Returning identity layer.")
        return tf.keras.layers.Lambda(lambda x: x)

# --- RandomChoice Pipeline ---
def get_random_choice_pipeline(auglayers):
    """
    Returns a RandomChoice layer from a list of augmentation layers.
    Usage:
        pipeline = get_random_choice_pipeline(auglayers)
        ds = ds.map(lambda img, lbl: (pipeline(img), lbl))
    """
    if not KERAS_CV_AVAILABLE:
        print("⚠️  RandomChoice not available (keras_cv not installed). Returning identity layer.")
        return tf.keras.layers.Lambda(lambda x: x)
    
    try:
        return keras_cv.layers.RandomChoice(layers=auglayers)
    except AttributeError:
        print("⚠️  RandomChoice not available in this version of keras_cv. Returning identity layer.")
        return tf.keras.layers.Lambda(lambda x: x)

# --- Comprehensive Random Choice Augmentation ---
def get_comprehensive_random_choice_augmentations():
    """
    Returns a comprehensive list of augmentation layers for random choice.
    This includes various types of augmentations that can be randomly selected.
    """
    return [
        # Geometric transformations
        tf.keras.layers.RandomRotation(factor=(-0.2, 0.2), fill_mode='reflect'),
        tf.keras.layers.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode='reflect'),
        tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode='reflect'),
        
        # Color and brightness augmentations (using TensorFlow layers)
        tf.keras.layers.RandomBrightness(factor=(-0.3, 0.3)),
        tf.keras.layers.RandomContrast(factor=(0.7, 1.3)),
        tf.keras.layers.Lambda(lambda x: tf.image.random_saturation(x, 0.7, 1.3)),
        tf.keras.layers.Lambda(lambda x: tf.image.random_hue(x, 0.1)),
        
        # Blur and noise
        keras_cv.layers.RandomGaussianBlur(kernel_size=3, factor=(0.0, 1.0)) if KERAS_CV_AVAILABLE else tf.keras.layers.Lambda(lambda x: x),
        tf.keras.layers.Lambda(lambda x: x + tf.random.normal(tf.shape(x), 0.0, 0.1)),
        
        # Cutout and masking (using TensorFlow implementation)
        tf.keras.layers.Lambda(lambda x: tf.image.random_crop(x, tf.shape(x))),  # Simple crop as cutout alternative
        tf.keras.layers.Lambda(lambda x: tf.image.central_crop(x, 0.9)),  # Center crop
        tf.keras.layers.Lambda(lambda x: tf.image.resize_with_crop_or_pad(x, tf.shape(x)[0], tf.shape(x)[1])),  # Resize with crop
        
        # Identity (no change) - allows some images to remain unchanged
        tf.keras.layers.Lambda(lambda x: x),
    ]

def get_aggressive_random_choice_augmentations():
    """
    Returns a more aggressive list of augmentation layers for random choice.
    Use this for datasets that need stronger augmentation.
    """
    return [
        # Stronger geometric transformations
        tf.keras.layers.RandomRotation(factor=(-0.3, 0.3), fill_mode='reflect'),
        tf.keras.layers.RandomTranslation(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode='reflect'),
        tf.keras.layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode='reflect'),
        
        # Stronger color augmentations (using TensorFlow layers)
        tf.keras.layers.RandomBrightness(factor=(-0.5, 0.5)),
        tf.keras.layers.RandomContrast(factor=(0.5, 1.5)),
        tf.keras.layers.Lambda(lambda x: tf.image.random_saturation(x, 0.5, 1.5)),
        tf.keras.layers.Lambda(lambda x: tf.image.random_hue(x, 0.2)),
        
        # More aggressive blur and noise
        keras_cv.layers.RandomGaussianBlur(kernel_size=5, factor=(0.0, 1.5)) if KERAS_CV_AVAILABLE else tf.keras.layers.Lambda(lambda x: x),
        tf.keras.layers.Lambda(lambda x: x + tf.random.normal(tf.shape(x), 0.0, 0.2)),
        
        # Larger cutouts (using TensorFlow implementation)
        tf.keras.layers.Lambda(lambda x: tf.image.central_crop(x, 0.8)),  # Larger center crop
        tf.keras.layers.Lambda(lambda x: tf.image.resize_with_crop_or_pad(x, tf.shape(x)[0]//2, tf.shape(x)[1])),  # Vertical crop
        tf.keras.layers.Lambda(lambda x: tf.image.resize_with_crop_or_pad(x, tf.shape(x)[0], tf.shape(x)[1]//2)),  # Horizontal crop
        
        # Identity (no change)
        tf.keras.layers.Lambda(lambda x: x),
    ]

def get_light_random_choice_augmentations():
    """
    Returns a lighter list of augmentation layers for random choice.
    Use this for datasets that need subtle augmentation.
    """
    return [
        # Subtle geometric transformations
        tf.keras.layers.RandomRotation(factor=(-0.1, 0.1), fill_mode='reflect'),
        tf.keras.layers.RandomTranslation(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05), fill_mode='reflect'),
        tf.keras.layers.RandomZoom(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05), fill_mode='reflect'),
        
        # Subtle color augmentations (using TensorFlow layers)
        tf.keras.layers.RandomBrightness(factor=(-0.1, 0.1)),
        tf.keras.layers.RandomContrast(factor=(0.9, 1.1)),
        tf.keras.layers.Lambda(lambda x: tf.image.random_saturation(x, 0.9, 1.1)),
        tf.keras.layers.Lambda(lambda x: tf.image.random_hue(x, 0.05)),
        
        # Light blur
        keras_cv.layers.RandomGaussianBlur(kernel_size=3, factor=(0.0, 0.5)) if KERAS_CV_AVAILABLE else tf.keras.layers.Lambda(lambda x: x),
        # Light noise
        tf.keras.layers.Lambda(lambda x: x + tf.random.normal(tf.shape(x), 0.0, 0.05)),
        
        # Small cutouts (using TensorFlow implementation)
        tf.keras.layers.Lambda(lambda x: tf.image.central_crop(x, 0.95)),  # Small center crop
        
        # Identity (no change)
        tf.keras.layers.Lambda(lambda x: x),
    ]

def augment_with_random_choice(sample: dict, augmentation_type: str = 'comprehensive') -> dict:
    """
    Apply random choice augmentation to a sample.
    
    Args:
        sample: Dictionary containing 'image' and 'label'
        augmentation_type: Type of augmentation ('comprehensive', 'aggressive', 'light')
    
    Returns:
        Dictionary with augmented image and original label
    """
    image = sample['image']
    
    # Select augmentation layers based on type
    if augmentation_type == 'aggressive':
        aug_layers = get_aggressive_random_choice_augmentations()
    elif augmentation_type == 'light':
        aug_layers = get_light_random_choice_augmentations()
    else:  # comprehensive (default)
        aug_layers = get_comprehensive_random_choice_augmentations()
    
    # Create random choice pipeline
    random_choice_pipeline = get_random_choice_pipeline(aug_layers)
    
    # Apply random choice augmentation
    augmented_image = random_choice_pipeline(image, training=True)
    
    # Ensure values are clipped to valid range
    augmented_image = tf.clip_by_value(augmented_image, 0.0, 1.0)
    
    return {**sample, 'image': augmented_image}

def augment_with_random_choice_batch(batch: dict, augmentation_type: str = 'comprehensive') -> dict:
    """
    Apply random choice augmentation to a batch of samples.
    
    Args:
        batch: Dictionary containing 'image' and 'label' tensors
        augmentation_type: Type of augmentation ('comprehensive', 'aggressive', 'light')
    
    Returns:
        Dictionary with augmented images and original labels
    """
    images = batch['image']
    labels = batch['label']
    
    # Select augmentation layers based on type
    if augmentation_type == 'aggressive':
        aug_layers = get_aggressive_random_choice_augmentations()
    elif augmentation_type == 'light':
        aug_layers = get_light_random_choice_augmentations()
    else:  # comprehensive (default)
        aug_layers = get_comprehensive_random_choice_augmentations()
    
    # Create random choice pipeline
    random_choice_pipeline = get_random_choice_pipeline(aug_layers)
    
    # Apply random choice augmentation to batch
    augmented_images = random_choice_pipeline(images, training=True)
    
    # Ensure values are clipped to valid range
    augmented_images = tf.clip_by_value(augmented_images, 0.0, 1.0)
    
    return {'image': augmented_images, 'label': labels}

# ==============================================================================
# Clean KerasCV Augmentation Implementation
# ==============================================================================

def get_keras_cv_augmenter(augmentation_type: str = 'comprehensive', num_classes: int = None):
    """
    Get a KerasCV Augmenter with the specified augmentation layers.
    
    Args:
        augmentation_type: Type of augmentation ('comprehensive', 'aggressive', 'light', 'basic')
        num_classes: Number of classes (required for CutMix/MixUp)
    
    Returns:
        KerasCV Augmenter layer
    """
    if not KERAS_CV_AVAILABLE:
        print("⚠️  keras_cv not available. Returning None.")
        return None
    
    try:
        layers = []
        
        # Basic augmentations (always included)
        layers.extend([
            keras_cv.layers.RandomFlip(),
            keras_cv.layers.RandomRotation(factor=(-0.1, 0.1), fill_mode='reflect'),
            keras_cv.layers.RandomTranslation(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05), fill_mode='reflect'),
            keras_cv.layers.RandomZoom(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05), fill_mode='reflect'),
            keras_cv.layers.RandomBrightness(factor=(-0.2, 0.2)),
            keras_cv.layers.RandomContrast(factor=(0.8, 1.2)),
        ])
        
        # Add mixing augmentations if num_classes is provided
        if num_classes is not None:
            layers.extend([
                keras_cv.layers.CutMix(alpha=0.5),
                keras_cv.layers.MixUp(alpha=0.2),
            ])
        
        # Add more aggressive augmentations based on type
        if augmentation_type == 'aggressive':
            layers.extend([
                keras_cv.layers.RandomRotation(factor=(-0.3, 0.3), fill_mode='reflect'),
                keras_cv.layers.RandomTranslation(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode='reflect'),
                keras_cv.layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode='reflect'),
                keras_cv.layers.RandomBrightness(factor=(-0.4, 0.4)),
                keras_cv.layers.RandomContrast(factor=(0.6, 1.4)),
                keras_cv.layers.RandomGaussianBlur(kernel_size=3, factor=(0.0, 1.0)),
            ])
            if num_classes is not None:
                layers.extend([
                    keras_cv.layers.CutMix(alpha=1.0),
                    keras_cv.layers.MixUp(alpha=0.4),
                ])
        
        elif augmentation_type == 'comprehensive':
            layers.extend([
                keras_cv.layers.RandomRotation(factor=(-0.2, 0.2), fill_mode='reflect'),
                keras_cv.layers.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode='reflect'),
                keras_cv.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode='reflect'),
                keras_cv.layers.RandomBrightness(factor=(-0.3, 0.3)),
                keras_cv.layers.RandomContrast(factor=(0.7, 1.3)),
                keras_cv.layers.RandomGaussianBlur(kernel_size=3, factor=(0.0, 0.8)),
            ])
            if num_classes is not None:
                layers.extend([
                    keras_cv.layers.CutMix(alpha=0.5),
                    keras_cv.layers.MixUp(alpha=0.2),
                ])
        
        # Light augmentations (already included in basic)
        elif augmentation_type == 'light':
            pass  # Use basic augmentations only
        
        return keras_cv.layers.Augmenter(layers=layers)
    
    except AttributeError as e:
        print(f"⚠️  Error creating KerasCV augmenter: {e}")
        return None

def augment_with_keras_cv(images, labels, num_classes: int, augmentation_type: str = 'comprehensive'):
    """
    Apply KerasCV augmentation to images and labels.
    
    Args:
        images: Batch of images
        labels: Batch of labels
        num_classes: Number of classes
        augmentation_type: Type of augmentation
    
    Returns:
        Tuple of (augmented_images, augmented_labels)
    """
    if not KERAS_CV_AVAILABLE:
        return images, labels
    
    augmenter = get_keras_cv_augmenter(augmentation_type, num_classes)
    if augmenter is None:
        return images, labels
    
    # Convert labels to one-hot
    one_hot_labels = tf.one_hot(labels, num_classes)
    
    # Apply augmentation
    inputs = {"images": images, "labels": one_hot_labels}
    output = augmenter(inputs)
    
    return output["images"], output["labels"]

def process_validation(images, labels, num_classes: int):
    """
    Process validation data (convert labels to one-hot).
    
    Args:
        images: Batch of images
        labels: Batch of labels
        num_classes: Number of classes
    
    Returns:
        Tuple of (images, one_hot_labels)
    """
    one_hot_labels = tf.one_hot(labels, num_classes)
    return images, one_hot_labels

def create_augmented_dataset(dataset, num_classes: int, mode: str = "train", 
                           augmentation_type: str = 'comprehensive', batch_size: int = 32):
    """
    Create an augmented dataset using KerasCV.
    
    Args:
        dataset: TensorFlow dataset (can be in dict format {'image': ..., 'label': ...} or tuple format (images, labels))
        num_classes: Number of classes
        mode: Dataset mode ('train' or 'validation')
        augmentation_type: Type of augmentation
        batch_size: Batch size
    
    Returns:
        Augmented TensorFlow dataset
    """
    if mode == "train":
        # Shuffle and batch
        dataset = dataset.shuffle(batch_size * 4)
        dataset = dataset.batch(batch_size)
        
        # Apply augmentation - handle both dict and tuple formats
        def augment_batch(batch_data):
            if isinstance(batch_data, dict):
                # Dict format: {'image': ..., 'label': ...}
                images = batch_data['image']
                labels = batch_data['label']
            else:
                # Tuple format: (images, labels)
                images, labels = batch_data
            
            # Apply KerasCV augmentation
            aug_images, aug_labels = augment_with_keras_cv(images, labels, num_classes, augmentation_type)
            
            # Return in dict format for consistency
            return {'image': aug_images, 'label': aug_labels}
        
        dataset = dataset.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # Batch and process validation
        dataset = dataset.batch(batch_size)
        
        def process_batch(batch_data):
            if isinstance(batch_data, dict):
                # Dict format: {'image': ..., 'label': ...}
                images = batch_data['image']
                labels = batch_data['label']
            else:
                # Tuple format: (images, labels)
                images, labels = batch_data
            
            # Return in dict format for consistency
            return {'image': images, 'label': labels}
        
        dataset = dataset.map(process_batch, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.cache()
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# --- Example usage in your notebook or script ---
# ds_train_MixUp = ds_train.map(mixup_batch_fn(num_classes=102), num_parallel_calls=tf.data.AUTOTUNE)
# ds_train_CutMix = ds_train.map(cutmix_batch_fn(num_classes=102), num_parallel_calls=tf.data.AUTOTUNE)
# randaugment_layer = get_randaugment_layer()
# ds_aug = ds_train.map(lambda img, lbl: (randaugment_layer(img), lbl))
# policy = get_standard_randaugment_policy()
# auglayers = policy[:4] + [keras_cv.layers.RandomCutout(0.5, 0.5)]
# pipeline = get_random_augmentation_pipeline(auglayers)
# ds_aug = ds_train.map(lambda img, lbl: (pipeline(img), lbl))
# pipeline2 = get_random_choice_pipeline(auglayers)
# ds_aug2 = ds_train.map(lambda img, lbl: (pipeline2(img), lbl)) 