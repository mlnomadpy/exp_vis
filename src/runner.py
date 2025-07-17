# ==============================================================================
# runner.py -- Main Training and Analysis Pipeline
# ==============================================================================
import os
import wandb
from src.data import create_image_folder_dataset, get_image_processor, get_tfds_processor, augment_for_pretraining, augment_for_finetuning
from src.models import YatCNN, ConvAutoencoder
from src.train import _pretrain_autoencoder_loop, _train_model_loop
from src.analysis import plot_training_curves, print_final_metrics, detailed_test_evaluation, plot_confusion_matrix, visualize_tsne, visualize_reconstructions

def run_training_and_analysis(
    dataset_name: str,
    dataset_configs: dict,
    fallback_configs: dict,
    learning_rate: float,
    use_pretraining: bool = True,
    freeze_encoder: bool = False,
    run_saliency_analysis: bool = True,
    run_kernel_analysis: bool = True,
    run_adversarial_analysis: bool = True,
    adversarial_epsilon: float = 0.01,
):
    print("\n" + "="*80 + f"\nRUNNING TRAINING & ANALYSIS FOR: {dataset_name.upper()}\n" + "="*80)
    is_path = os.path.isdir(dataset_name)
    if not is_path and dataset_name not in dataset_configs:
        raise ValueError(f"Dataset '{dataset_name}' is not a valid path and has no configuration.")
    pretrained_encoder_path = None
    if use_pretraining:
        print(f"\n🚀 STEP 1: Starting Autoencoder Pretraining...")
        pretrained_encoder_path = _pretrain_autoencoder_loop(
            model_name="YatCNN", dataset_name=dataset_name,
            rng_seed=42, learning_rate=learning_rate, optimizer_constructor=wandb.config.get('optimizer', None) or __import__('optax').novograd,
            dataset_configs=dataset_configs, fallback_configs=fallback_configs,
        )
    if use_pretraining and pretrained_encoder_path is None:
        print("\nPretraining failed or was skipped. Aborting fine-tuning.")
        return
    print(f"\n🚀 STEP 2: {'Fine-tuning' if use_pretraining else 'Training'} Model...");
    model, metrics_history, test_ds, class_names = _train_model_loop(
        YatCNN, "YatCNN", dataset_name, 0, learning_rate, __import__('optax').novograd,
        dataset_configs=dataset_configs, fallback_configs=fallback_configs,
        pretrained_encoder_path=pretrained_encoder_path,
        freeze_encoder=freeze_encoder
    )
    print("\n📊 STEP 3: Running Final Performance Analysis...")
    plot_training_curves(metrics_history, "YatCNN")
    print_final_metrics(metrics_history, "YatCNN")
    config = dataset_configs.get(dataset_name) if not is_path else dataset_configs.get('custom_folder', fallback_configs)
    test_iter = test_ds.batch(config.get('batch_size')).as_numpy_iterator()
    predictions_data = detailed_test_evaluation(model, test_iter, class_names=class_names, model_name="YatCNN")
    if len(class_names) <= 50:
        plot_confusion_matrix(predictions_data, "YatCNN")
    print("\n🔬 STEP 4: Model Interpretability and Robustness Analysis...")
    if run_saliency_analysis:
        from src.analysis import plot_saliency_maps
        plot_saliency_maps(model, test_ds, class_names)
    if run_kernel_analysis:
        from src.analysis import visualize_kernel_similarity
        visualize_kernel_similarity(model)
    if run_adversarial_analysis:
        from src.analysis import test_adversarial_robustness
        adv_test_iter = test_ds.batch(config.get('batch_size')).as_numpy_iterator()
        test_adversarial_robustness(model, adv_test_iter, class_names, epsilon=adversarial_epsilon)
    print("\n🎨 Generating t-SNE plot for the test set representations...")
    tsne_test_iter = test_ds.batch(config.get('batch_size', fallback_configs['batch_size'])).as_numpy_iterator()
    visualize_tsne(model, tsne_test_iter, class_names, title="t-SNE of Final Model Representations on Test Set")
    print("\n" + "="*80 + f"\nANALYSIS FOR {dataset_name.upper()} COMPLETE! ✅\n" + "="*80)
    # --- WANDB LOGGING ---
    for i, (train_loss, train_acc, test_loss, test_acc) in enumerate(zip(
        metrics_history['train_loss'], metrics_history['train_accuracy'],
        metrics_history['test_loss'], metrics_history['test_accuracy'])):
        wandb.log({
            'step': i,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
        })
    # Optionally log confusion matrix, t-SNE, or model artifacts here
    return {'model': model, 'metrics_history': metrics_history, 'predictions_data': predictions_data} 