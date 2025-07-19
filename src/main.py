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
    parser.add_argument('--use_simo2_pretraining', action='store_true', help='Use SIMO2 pretraining')
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
    # SIMO2 specific arguments
    parser.add_argument('--embedding_size', type=int, default=16, help='Embedding size for SIMO2')
    parser.add_argument('--samples_per_class', type=int, default=32, help='Samples per class for SIMO2')
    parser.add_argument('--orth_lean', type=float, default=1/137, help='Orthogonality leaning for SIMO2')
    parser.add_argument('--log_rate', type=int, default=10000, help='Logging rate for SIMO2')
    
    # Learning rate scheduler arguments
    parser.add_argument('--scheduler_type', type=str, default='constant', 
                       choices=['constant', 'cosine', 'exponential', 'step', 'warmup_cosine', 'linear', 'polynomial'],
                       help='Learning rate scheduler type')
    parser.add_argument('--optimizer_type', type=str, default='novograd',
                       choices=['adam', 'adamw', 'sgd', 'novograd', 'rmsprop'],
                       help='Optimizer type')
    parser.add_argument('--scheduler_alpha', type=float, default=0.0, help='Alpha parameter for cosine scheduler')
    parser.add_argument('--scheduler_decay_rate', type=float, default=0.1, help='Decay rate for exponential scheduler')
    parser.add_argument('--scheduler_step_size', type=int, default=None, help='Step size for step scheduler')
    parser.add_argument('--scheduler_decay_factor', type=float, default=0.1, help='Decay factor for step scheduler')
    parser.add_argument('--scheduler_warmup_steps', type=int, default=None, help='Warmup steps for warmup_cosine scheduler')
    parser.add_argument('--scheduler_end_value', type=float, default=None, help='End value for schedulers')
    parser.add_argument('--scheduler_power', type=float, default=1.0, help='Power for polynomial scheduler')
    parser.add_argument('--orthogonality_weight', type=float, default=0.0, help='Weight for orthogonality regularization of output layer')
    
    # Augmentation arguments
    parser.add_argument('--use_keras_cv_augmentation', action='store_true', default=True, help='Use KerasCV augmentation (includes CutMix and MixUp)')
    parser.add_argument('--augmentation_type', type=str, default='comprehensive', 
                       choices=['comprehensive', 'aggressive', 'light'],
                       help='Type of random choice augmentation')
    parser.add_argument('--pretrain_augmentation_type', type=str, default='comprehensive',
                       choices=['comprehensive', 'aggressive', 'light'],
                       help='Type of random choice augmentation for pretraining')
    
    args = parser.parse_args()

    dataset_configs = {
        'cifar10': {
            'input_channels': 3, 'input_dim': (32, 32), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 100, 'eval_every': 300, 'batch_size': 256,
            'pretrain_epochs': 100, 'pretrain_batch_size': 256,
            # SIMO2 specific config
            'embedding_size': 16, 'samples_per_class': 32, 'orth_lean': 1/137, 'log_rate': 10000,
            'num_classes': 10, 'dataset': 'cifar10', 'apply_normalization': True,
            # Scheduler config
            'scheduler_type': 'cosine', 'optimizer_type': 'novograd',
            'scheduler_params': {'alpha': 0.0},
            # Augmentation config
            'use_random_choice_augmentation': True,
            'augmentation_type': 'comprehensive',
            'pretrain_augmentation_type': 'comprehensive',
        },
        'cifar100': {
            'input_channels': 3, 'input_dim': (32, 32), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 100, 'eval_every': 300, 'batch_size': 256,
            'pretrain_epochs': 100, 'pretrain_batch_size': 256,
            # SIMO2 specific config
            'embedding_size': 16, 'samples_per_class': 32, 'orth_lean': 1/137, 'log_rate': 10000,
            'num_classes': 100, 'dataset': 'cifar100', 'apply_normalization': True,
            # Scheduler config
            'scheduler_type': 'cosine', 'optimizer_type': 'novograd',
            'scheduler_params': {'alpha': 0.0},
            # Augmentation config
            'use_random_choice_augmentation': True,
            'augmentation_type': 'comprehensive',
            'pretrain_augmentation_type': 'comprehensive',
        },
        'mnist': {
            'input_channels': 1, 'input_dim': (28, 28), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 50, 'eval_every': 200, 'batch_size': 256,
            'pretrain_epochs': 50, 'pretrain_batch_size': 256,
            # SIMO2 specific config
            'embedding_size': 16, 'samples_per_class': 32, 'orth_lean': 1/137, 'log_rate': 10000,
            'num_classes': 10, 'dataset': 'mnist', 'apply_normalization': True,
            # Scheduler config
            'scheduler_type': 'cosine', 'optimizer_type': 'novograd',
            'scheduler_params': {'alpha': 0.0},
            # Augmentation config
            'use_random_choice_augmentation': True,
            'augmentation_type': 'light',
            'pretrain_augmentation_type': 'light',
        },
        'fashion_mnist': {
            'input_channels': 1, 'input_dim': (28, 28), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 50, 'eval_every': 200, 'batch_size': 256,
            'pretrain_epochs': 50, 'pretrain_batch_size': 256,
            # SIMO2 specific config
            'embedding_size': 16, 'samples_per_class': 32, 'orth_lean': 1/137, 'log_rate': 10000,
            'num_classes': 10, 'dataset': 'fashion_mnist', 'apply_normalization': True,
            # Augmentation config
            'use_random_choice_augmentation': True,
            'augmentation_type': 'light',
            'pretrain_augmentation_type': 'light',
        },
        'imagenet2012': {
            'input_channels': 3, 'input_dim': (224, 224), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'validation',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 90, 'eval_every': 1000, 'batch_size': 128,
            'pretrain_epochs': 50, 'pretrain_batch_size': 128,
            # SIMO2 specific config
            'embedding_size': 16, 'samples_per_class': 32, 'orth_lean': 1/137, 'log_rate': 10000,
            'num_classes': 1000, 'dataset': 'imagenet2012', 'apply_normalization': True,
            # Augmentation config
            'use_random_choice_augmentation': True,
            'augmentation_type': 'aggressive',
            'pretrain_augmentation_type': 'aggressive',
        },
        'caltech101': {
            'input_channels': 3, 'input_dim': (224, 224), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 50, 'eval_every': 200, 'batch_size': 128,
            'pretrain_epochs': 30, 'pretrain_batch_size': 128,
            # SIMO2 specific config
            'embedding_size': 16, 'samples_per_class': 32, 'orth_lean': 1/137, 'log_rate': 10000,
            'num_classes': 101, 'dataset': 'caltech101', 'apply_normalization': True,
            # Augmentation config
            'use_random_choice_augmentation': True,
            'augmentation_type': 'comprehensive',
            'pretrain_augmentation_type': 'comprehensive',
        },
        'oxford_flowers102': {
            'input_channels': 3, 'input_dim': (224, 224), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 50, 'eval_every': 200, 'batch_size': 128,
            'pretrain_epochs': 30, 'pretrain_batch_size': 128,
            # SIMO2 specific config
            'embedding_size': 16, 'samples_per_class': 32, 'orth_lean': 1/137, 'log_rate': 10000,
            'num_classes': 102, 'dataset': 'oxford_flowers102', 'apply_normalization': True,
            # Augmentation config
            'use_random_choice_augmentation': True,
            'augmentation_type': 'comprehensive',
            'pretrain_augmentation_type': 'comprehensive',
        },
        'stanford_dogs': {
            'input_channels': 3, 'input_dim': (224, 224), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 50, 'eval_every': 200, 'batch_size': 128,
            'pretrain_epochs': 30, 'pretrain_batch_size': 128,
            # SIMO2 specific config
            'embedding_size': 16, 'samples_per_class': 32, 'orth_lean': 1/137, 'log_rate': 10000,
            'num_classes': 120, 'dataset': 'stanford_dogs', 'apply_normalization': True,
            # Augmentation config
            'use_random_choice_augmentation': True,
            'augmentation_type': 'comprehensive',
            'pretrain_augmentation_type': 'comprehensive',
        },
        'cats_vs_dogs': {
            'input_channels': 3, 'input_dim': (224, 224), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 30, 'eval_every': 200, 'batch_size': 128,
            'pretrain_epochs': 20, 'pretrain_batch_size': 128,
            # SIMO2 specific config
            'embedding_size': 16, 'samples_per_class': 32, 'orth_lean': 1/137, 'log_rate': 10000,
            'num_classes': 2, 'dataset': 'cats_vs_dogs', 'apply_normalization': True,
            # Augmentation config
            'use_random_choice_augmentation': True,
            'augmentation_type': 'comprehensive',
            'pretrain_augmentation_type': 'comprehensive',
        },
        'stl10': {
            'input_channels': 3, 'input_dim': (96, 96), 'label_smooth': 0.1,
            'train_split': 'train', 'test_split': 'test',
            'image_key': 'image', 'label_key': 'label', 
            'num_epochs': 100, 'eval_every': 300, 'batch_size': 128,
            'pretrain_epochs': 50, 'pretrain_batch_size': 128,
            # SIMO2 specific config
            'embedding_size': 16, 'samples_per_class': 32, 'orth_lean': 1/137, 'log_rate': 10000,
            'num_classes': 10, 'dataset': 'stl10', 'apply_normalization': True,
            # Augmentation config
            'use_random_choice_augmentation': True,
            'augmentation_type': 'comprehensive',
            'pretrain_augmentation_type': 'comprehensive',
        },
        'custom_folder': {
            'input_channels': 3, 'input_dim': (32, 32),
            'test_split_percentage': 0.2,
            'num_epochs': 10, 'eval_every': 200, 'batch_size': 128,
            'pretrain_epochs': 100, 'pretrain_batch_size': 256,
            # SIMO2 specific config (will be updated based on actual data)
            'embedding_size': 16, 'samples_per_class': 32, 'orth_lean': 1/137, 'log_rate': 10000,
            'apply_normalization': True,
            # Augmentation config
            'use_random_choice_augmentation': True,
            'augmentation_type': 'comprehensive',
            'pretrain_augmentation_type': 'comprehensive',
        }
    }

    fallback_configs = {
        'num_epochs': 10, 'eval_every': 200, 'batch_size': 64, 'label_smooth':0.1,
        'pretrain_epochs': 20, 'pretrain_batch_size': 128,
        # SIMO2 fallback configs
        'embedding_size': 16, 'samples_per_class': 32, 'orth_lean': 1/137, 'log_rate': 10000,
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
    # SIMO2 specific argument updates
    if args.embedding_size is not None:
        config['embedding_size'] = args.embedding_size
    if args.samples_per_class is not None:
        config['samples_per_class'] = args.samples_per_class
    if args.orth_lean is not None:
        config['orth_lean'] = args.orth_lean
    if args.log_rate is not None:
        config['log_rate'] = args.log_rate
    
    # Scheduler and optimizer argument updates
    if args.scheduler_type is not None:
        config['scheduler_type'] = args.scheduler_type
    if args.optimizer_type is not None:
        config['optimizer_type'] = args.optimizer_type
    
    # Build scheduler parameters dictionary
    scheduler_params = {}
    if args.scheduler_alpha is not None:
        scheduler_params['alpha'] = args.scheduler_alpha
    if args.scheduler_decay_rate is not None:
        scheduler_params['decay_rate'] = args.scheduler_decay_rate
    if args.scheduler_step_size is not None:
        scheduler_params['step_size'] = args.scheduler_step_size
    if args.scheduler_decay_factor is not None:
        scheduler_params['decay_factor'] = args.scheduler_decay_factor
    if args.scheduler_warmup_steps is not None:
        scheduler_params['warmup_steps'] = args.scheduler_warmup_steps
    if args.scheduler_end_value is not None:
        scheduler_params['end_value'] = args.scheduler_end_value
    if args.scheduler_power is not None:
        scheduler_params['power'] = args.scheduler_power
    
    if scheduler_params:
        config['scheduler_params'] = scheduler_params
    
    # Augmentation argument updates
    if args.use_random_choice_augmentation is not None:
        config['use_random_choice_augmentation'] = args.use_random_choice_augmentation
    if args.augmentation_type is not None:
        config['augmentation_type'] = args.augmentation_type
    if args.pretrain_augmentation_type is not None:
        config['pretrain_augmentation_type'] = args.pretrain_augmentation_type

    learning_rate = args.learning_rate
    use_pretraining = args.use_pretraining
    use_simo2_pretraining = args.use_simo2_pretraining
    freeze_encoder_during_finetune = args.freeze_encoder
    run_saliency_analysis = args.run_saliency_analysis
    run_kernel_analysis = args.run_kernel_analysis
    run_adversarial_analysis = args.run_adversarial_analysis
    adversarial_epsilon = args.adversarial_epsilon

    # Validate pretraining options
    if use_pretraining and use_simo2_pretraining:
        print("ERROR: Cannot use both autoencoder pretraining and SIMO2 pretraining simultaneously.")
        print("Please choose either --use_pretraining or --use_simo2_pretraining, not both.")
        return

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
    # Get the final config that will be used for training
    final_config = dataset_configs.get(dataset_to_run) if not is_path else dataset_configs.get('custom_folder', fallback_configs)
    
    wandb.init(
        project="exp_vis",
        config={
            "dataset": dataset_to_run,
            "learning_rate": learning_rate,
            "use_pretraining": use_pretraining,
            "use_simo2_pretraining": use_simo2_pretraining,
            "freeze_encoder": freeze_encoder_during_finetune,
            "run_saliency_analysis": run_saliency_analysis,
            "run_kernel_analysis": run_kernel_analysis,
            "run_adversarial_analysis": run_adversarial_analysis,
            "adversarial_epsilon": adversarial_epsilon,
            "orthogonality_weight": args.orthogonality_weight,
            "scheduler_type": args.scheduler_type,
            "optimizer_type": args.optimizer_type,
            "scheduler_params": scheduler_params if 'scheduler_params' in locals() else {},
            # Include the actual dataset config that will be used
            "dataset_config": final_config,
        }
    )
    wandb_config = wandb.config

    # Debug: Print the final configuration being used
    print(f"\n🔧 Final configuration for dataset '{dataset_to_run}':")
    for key, value in final_config.items():
        print(f"   {key}: {value}")
    
    # Debug: Print CLI arguments that were passed
    print(f"\n🔧 CLI arguments that were passed:")
    for arg, value in vars(args).items():
        if value is not None and arg != 'dataset':
            print(f"   --{arg}: {value}")

    run_training_and_analysis(
        dataset_name=dataset_to_run,
        dataset_config=final_config,
        fallback_configs=fallback_configs,
        learning_rate=learning_rate,
        use_pretraining=use_pretraining,
        use_simo2_pretraining=use_simo2_pretraining,
        freeze_encoder=freeze_encoder_during_finetune,
        run_saliency_analysis=run_saliency_analysis,
        run_kernel_analysis=run_kernel_analysis,
        run_adversarial_analysis=run_adversarial_analysis,
        adversarial_epsilon=adversarial_epsilon,
        orthogonality_weight=args.orthogonality_weight,
    )

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    main()
