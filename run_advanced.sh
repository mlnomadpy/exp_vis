#!/bin/bash

# Example advanced usage of the training pipeline with full control over arguments
# You can modify these values as needed

# Dataset options: 'cifar10' or a path to a custom folder
DATASET="cifar10"
LEARNING_RATE=0.005
USE_PRETRAINING="--use_pretraining"           # Remove this line to disable pretraining
FREEZE_ENCODER=""                            # Set to "--freeze_encoder" to freeze encoder during fine-tuning
RUN_SALIENCY="--run_saliency_analysis"        # Remove to skip saliency analysis
RUN_KERNEL="--run_kernel_analysis"            # Remove to skip kernel similarity analysis
RUN_ADVERSARIAL="--run_adversarial_analysis"  # Remove to skip adversarial robustness analysis
ADVERSARIAL_EPSILON=0.02

# Dataset/training config overrides (uncomment and edit as needed)
INPUT_CHANNELS=3           # e.g. 1 for grayscale, 3 for RGB
INPUT_DIM="32,32"          # e.g. 28,28 or 64,64
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
  --test_split $TEST_SPLIT 