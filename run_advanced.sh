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

python src/main.py \
  --dataset "$DATASET" \
  --learning_rate $LEARNING_RATE \
  $USE_PRETRAINING \
  $FREEZE_ENCODER \
  $RUN_SALIENCY \
  $RUN_KERNEL \
  $RUN_ADVERSARIAL \
  --adversarial_epsilon $ADVERSARIAL_EPSILON 