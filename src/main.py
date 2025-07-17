# ==============================================================================
# main.py -- Entrypoint for Training and Analysis
# ==============================================================================
import os
import tensorflow as tf
import wandb
import argparse
from runner import run_training_and_analysis

def main():
    parser = argparse.ArgumentParser(description="Run training and analysis pipeline.")
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name or path')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--use_pretraining', action='store_true', help='Use autoencoder pretraining')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder during fine-tuning')
    parser.add_argument('--run_saliency_analysis', action='store_true', help='Run saliency analysis')
    parser.add_argument('--run_kernel_analysis', action='store_true', help='Run kernel similarity analysis')
    parser.add_argument('--run_adversarial_analysis', action='store_true', help='Run adversarial robustness analysis')
    parser.add_argument('--adversarial_epsilon', type=float, default=0.01, help='Epsilon for adversarial attack')
    args = parser.parse_args()

    dataset_configs = {
        'cifar10': {
            'input_channels': 3, 'input_dim': (32, 32), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 100, 'eval_every': 300, 'batch_size': 256,
            'pretrain_epochs': 100, 'pretrain_batch_size': 256,
        },
        'custom_folder': {
            'input_channels': 3, 'input_dim': (32, 32),
            'test_split_percentage': 0.2,
            'num_epochs': 10, 'eval_every': 200, 'batch_size': 128,
            'pretrain_epochs': 100, 'pretrain_batch_size': 256,
        }
    }

    fallback_configs = {
        'num_epochs': 10, 'eval_every': 200, 'batch_size': 64, 'label_smooth':0.1,
        'pretrain_epochs': 20, 'pretrain_batch_size': 128,
    }

    dataset_to_run = args.dataset
    learning_rate = args.learning_rate
    use_pretraining = args.use_pretraining
    freeze_encoder_during_finetune = args.freeze_encoder
    run_saliency_analysis = args.run_saliency_analysis
    run_kernel_analysis = args.run_kernel_analysis
    run_adversarial_analysis = args.run_adversarial_analysis
    adversarial_epsilon = args.adversarial_epsilon

    is_path = os.path.isdir(dataset_to_run)
    if not is_path and dataset_to_run not in dataset_configs:
        print("="*80)
        print(f"ERROR: Dataset '{dataset_to_run}' not found as a directory or in `dataset_configs`.")
        print("Please update the `dataset_to_run` variable or add a configuration for it.")
        print("="*80)
        return
    if is_path and 'custom_folder' not in dataset_configs:
        print("ERROR: `dataset_to_run` is a path, but no 'custom_folder' config found.")
        return

    # --- WANDB INTEGRATION ---
    wandb.init(
        project="exp_vis",
        config={
            "dataset": dataset_to_run,
            "learning_rate": learning_rate,
            "use_pretraining": use_pretraining,
            "freeze_encoder": freeze_encoder_during_finetune,
            "run_saliency_analysis": run_saliency_analysis,
            "run_kernel_analysis": run_kernel_analysis,
            "run_adversarial_analysis": run_adversarial_analysis,
            "adversarial_epsilon": adversarial_epsilon,
        }
    )
    config = wandb.config

    run_training_and_analysis(
        dataset_name=dataset_to_run,
        dataset_configs=dataset_configs,
        fallback_configs=fallback_configs,
        learning_rate=learning_rate,
        use_pretraining=use_pretraining,
        freeze_encoder=freeze_encoder_during_finetune,
        run_saliency_analysis=run_saliency_analysis,
        run_kernel_analysis=run_kernel_analysis,
        run_adversarial_analysis=run_adversarial_analysis,
        adversarial_epsilon=adversarial_epsilon,
    )

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    main()
