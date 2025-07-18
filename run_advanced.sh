#!/bin/bash

# Example advanced usage of the training pipeline with full control over arguments
# You can modify these values as needed

# Dataset options: 
# - TFDS datasets: 'cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'imagenet2012', 
#   'caltech101', 'oxford_flowers102', 'stanford_dogs', 'cats_vs_dogs', 'stl10'
# - Custom folder: path to your image folder
DATASET="cifar10"
LEARNING_RATE=0.005
USE_PRETRAINING="--use_pretraining"           # Remove this line to disable pretraining
FREEZE_ENCODER=""                            # Set to "--freeze_encoder" to freeze encoder during fine-tuning
RUN_SALIENCY="--run_saliency_analysis"        # Remove to skip saliency analysis
RUN_KERNEL="--run_kernel_analysis"            # Remove to skip kernel similarity analysis
RUN_ADVERSARIAL="--run_adversarial_analysis"  # Remove to skip adversarial robustness analysis
ADVERSARIAL_EPSILON=0.02
# --use_simo2_pretraining

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

# Dataset/training config overrides (uncomment and edit as needed)
# Note: Some datasets have specific defaults:
# - mnist/fashion_mnist: 1 channel, 28x28, 10 classes
# - cifar10/cifar100: 3 channels, 32x32, 10/100 classes  
# - stl10: 3 channels, 96x96, 10 classes
# - imagenet2012: 3 channels, 224x224, 1000 classes
# - caltech101/oxford_flowers102/stanford_dogs: 3 channels, 224x224
INPUT_CHANNELS=3           # e.g. 1 for grayscale, 3 for RGB
INPUT_DIM="32,32"          # e.g. 28,28 or 64,64 or 96,96 or 224,224
LABEL_SMOOTH=0.1           # e.g. 0.0 for no smoothing
NUM_EPOCHS=100             # e.g. 50
EVAL_EVERY=300             # e.g. 100
BATCH_SIZE=256             # e.g. 64
PRETRAIN_EPOCHS=100        # e.g. 20
PRETRAIN_BATCH_SIZE=256    # e.g. 128
TEST_SPLIT_PERCENTAGE=0.2  # e.g. 0.1
IMAGE_KEY="image"         # e.g. "image"
LABEL_KEY="label"         # e.g. "label"
TRAIN_SPLIT="train"       # e.g. "train[:80%]"
TEST_SPLIT="test"         # e.g. "train[80%:]"

python src/main.py \
  --dataset "$DATASET" \
  --learning_rate $LEARNING_RATE \
  $USE_PRETRAINING \
  $FREEZE_ENCODER \
  $RUN_SALIENCY \
  $RUN_KERNEL \
  $RUN_ADVERSARIAL \
  --adversarial_epsilon $ADVERSARIAL_EPSILON \
  --input_channels $INPUT_CHANNELS \
  --input_dim $INPUT_DIM \
  --label_smooth $LABEL_SMOOTH \
  --num_epochs $NUM_EPOCHS \
  --eval_every $EVAL_EVERY \
  --batch_size $BATCH_SIZE \
  --pretrain_epochs $PRETRAIN_EPOCHS \
  --pretrain_batch_size $PRETRAIN_BATCH_SIZE \
  --test_split_percentage $TEST_SPLIT_PERCENTAGE \
  --image_key $IMAGE_KEY \
  --label_key $LABEL_KEY \
  --train_split $TRAIN_SPLIT \
  --test_split $TEST_SPLIT \
  --scheduler_type $SCHEDULER_TYPE \
  --optimizer_type $OPTIMIZER_TYPE \
  --scheduler_alpha $SCHEDULER_ALPHA \
  --scheduler_decay_rate $SCHEDULER_DECAY_RATE \
  --scheduler_step_size $SCHEDULER_STEP_SIZE \
  --scheduler_decay_factor $SCHEDULER_DECAY_FACTOR \
  --scheduler_warmup_steps $SCHEDULER_WARMUP_STEPS \
  --scheduler_end_value $SCHEDULER_END_VALUE \
  --scheduler_power $SCHEDULER_POWER 