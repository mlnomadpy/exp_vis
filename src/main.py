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
    parser.add_argument('--input_channels', type=int, default=None, help='Number of input channels')
    parser.add_argument('--input_dim', type=str, default=None, help='Input dimension as H,W (e.g. 32,32)')
    parser.add_argument('--label_smooth', type=float, default=None, help='Label smoothing factor')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--eval_every', type=int, default=None, help='Evaluation frequency (steps)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--pretrain_epochs', type=int, default=None, help='Pretraining epochs')
    parser.add_argument('--pretrain_batch_size', type=int, default=None, help='Pretraining batch size')
    parser.add_argument('--test_split_percentage', type=float, default=None, help='Test split percentage for custom folder')
    parser.add_argument('--image_key', type=str, default=None, help='Image key for TFDS')
    parser.add_argument('--label_key', type=str, default=None, help='Label key for TFDS')
    parser.add_argument('--train_split', type=str, default=None, help='Train split for TFDS')
    parser.add_argument('--test_split', type=str, default=None, help='Test split for TFDS')
    args = parser.parse_args()

    dataset_configs = {
        'cifar10': {
            'input_channels': 3, 'input_dim': (32, 32), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 100, 'eval_every': 300, 'batch_size': 256,
            'pretrain_epochs': 100, 'pretrain_batch_size': 256,
        },
        'cifar100': {
            'input_channels': 3, 'input_dim': (32, 32), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 100, 'eval_every': 300, 'batch_size': 256,
            'pretrain_epochs': 100, 'pretrain_batch_size': 256,
        },
        'mnist': {
            'input_channels': 1, 'input_dim': (28, 28), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 50, 'eval_every': 200, 'batch_size': 256,
            'pretrain_epochs': 50, 'pretrain_batch_size': 256,
        },
        'fashion_mnist': {
            'input_channels': 1, 'input_dim': (28, 28), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 50, 'eval_every': 200, 'batch_size': 256,
            'pretrain_epochs': 50, 'pretrain_batch_size': 256,
        },
        'imagenet2012': {
            'input_channels': 3, 'input_dim': (224, 224), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'validation',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 90, 'eval_every': 1000, 'batch_size': 128,
            'pretrain_epochs': 50, 'pretrain_batch_size': 128,
        },
        'caltech101': {
            'input_channels': 3, 'input_dim': (224, 224), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 50, 'eval_every': 200, 'batch_size': 128,
            'pretrain_epochs': 30, 'pretrain_batch_size': 128,
        },
        'oxford_flowers102': {
            'input_channels': 3, 'input_dim': (224, 224), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 50, 'eval_every': 200, 'batch_size': 128,
            'pretrain_epochs': 30, 'pretrain_batch_size': 128,
        },
        'stanford_dogs': {
            'input_channels': 3, 'input_dim': (224, 224), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 50, 'eval_every': 200, 'batch_size': 128,
            'pretrain_epochs': 30, 'pretrain_batch_size': 128,
        },
        'cats_vs_dogs': {
            'input_channels': 3, 'input_dim': (224, 224), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 30, 'eval_every': 200, 'batch_size': 128,
            'pretrain_epochs': 20, 'pretrain_batch_size': 128,
        },
        'stl10': {
            'input_channels': 3, 'input_dim': (96, 96), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 100, 'eval_every': 300, 'batch_size': 128,
            'pretrain_epochs': 50, 'pretrain_batch_size': 128,
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

    # Update dataset_configs for the selected dataset with CLI args
    dataset_to_run = args.dataset
    if dataset_to_run not in dataset_configs:
        dataset_configs[dataset_to_run] = {}
    config = dataset_configs[dataset_to_run]
    if args.input_channels is not None:
        config['input_channels'] = args.input_channels
    if args.input_dim is not None:
        dims = tuple(map(int, args.input_dim.split(',')))
        config['input_dim'] = dims
    if args.label_smooth is not None:
        config['label_smooth'] = args.label_smooth
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    if args.eval_every is not None:
        config['eval_every'] = args.eval_every
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.pretrain_epochs is not None:
        config['pretrain_epochs'] = args.pretrain_epochs
    if args.pretrain_batch_size is not None:
        config['pretrain_batch_size'] = args.pretrain_batch_size
    if args.test_split_percentage is not None:
        config['test_split_percentage'] = args.test_split_percentage
    if args.image_key is not None:
        config['image_key'] = args.image_key
    if args.label_key is not None:
        config['label_key'] = args.label_key
    if args.train_split is not None:
        config['train_split'] = args.train_split
    if args.test_split is not None:
        config['test_split'] = args.test_split

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
