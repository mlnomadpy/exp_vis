# ==============================================================================
# data.py -- Data Loading, Preprocessing, and Augmentation
# ==============================================================================
import os
import glob
import tensorflow as tf
import keras_cv

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
)
CUTOUT_LAYER = keras_cv.layers.RandomCutout(
    height_factor=0.2, width_factor=0.2
)

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
    if tf.random.uniform([]) > 0.5:
        image = GAUSSIAN_BLUR_LAYER(image, training=True)
    if tf.random.uniform([]) > 0.5:
        image = CUTOUT_LAYER(image, training=True)
    augmented_image = tf.clip_by_value(image, 0.0, 1.0)
    return {**sample, 'image': augmented_image}
