# ==============================================================================
# runner.py -- Main Training and Analysis Pipeline
# ==============================================================================
import os
import wandb
from data import create_image_folder_dataset, get_image_processor, get_tfds_processor, augment_for_pretraining, augment_for_finetuning
from models import YatCNN, ConvAutoencoder
from train import _pretrain_autoencoder_loop, _train_model_loop, _pretrain_simo2_loop
from analysis import plot_training_curves, print_final_metrics, detailed_test_evaluation, plot_confusion_matrix, visualize_tsne, visualize_reconstructions
from logger import init_wandb, log_metrics

def run_training_and_analysis(
    dataset_name: str,
    dataset_config: dict,
    fallback_configs: dict,
    learning_rate: float,
    use_pretraining: bool = True,
    use_simo2_pretraining: bool = False,
    freeze_encoder: bool = False,
    run_saliency_analysis: bool = True,
    run_kernel_analysis: bool = True,
    run_adversarial_analysis: bool = True,
    adversarial_epsilon: float = 0.01,
    orthogonality_weight: float = 0.0,
):
    print("\n" + "="*80 + f"\nRUNNING TRAINING & ANALYSIS FOR: {dataset_name.upper()}\n" + "="*80)
    is_path = os.path.isdir(dataset_name)
    pretrained_encoder_path = None
    
    if use_pretraining:
        print(f"\n🚀 STEP 1: Starting Autoencoder Pretraining...")
        pretrained_encoder_path = _pretrain_autoencoder_loop(
            model_name="YatCNN", dataset_name=dataset_name,
            rng_seed=42, learning_rate=learning_rate, optimizer_constructor=wandb.config.get('optimizer', None) or __import__('optax').novograd,
            dataset_config=dataset_config, fallback_configs=fallback_configs,
        )
    elif use_simo2_pretraining:
        print(f"\n🚀 STEP 1: Starting SIMO2 Pretraining...")
        pretrained_encoder_path = _pretrain_simo2_loop(
            model_name="YatCNN", dataset_name=dataset_name,
            rng_seed=42, learning_rate=learning_rate, optimizer_constructor=wandb.config.get('optimizer', None) or __import__('optax').adam,
            dataset_config=dataset_config, fallback_configs=fallback_configs,
        )
    
    if (use_pretraining or use_simo2_pretraining) and pretrained_encoder_path is None:
        print("\nPretraining failed or was skipped. Aborting fine-tuning.")
        return
    
    pretraining_type = "SIMO2" if use_simo2_pretraining else "Autoencoder" if use_pretraining else "None"
    print(f"\n🚀 STEP 2: {'Fine-tuning' if (use_pretraining or use_simo2_pretraining) else 'Training'} Model ({pretraining_type} pretraining)...");
    
    model, metrics_history, test_ds, class_names, detailed_metrics = _train_model_loop(
        YatCNN, "YatCNN", dataset_name, 0, learning_rate, __import__('optax').novograd,
        dataset_config=dataset_config, fallback_configs=fallback_configs,
        pretrained_encoder_path=pretrained_encoder_path,
        freeze_encoder=freeze_encoder,
        orthogonality_weight=orthogonality_weight
    )
    print("\n📊 STEP 3: Running Final Performance Analysis...")
    plot_training_curves(metrics_history, "YatCNN")
    print_final_metrics(metrics_history, "YatCNN")
    test_iter = test_ds.batch(dataset_config.get('batch_size')).as_numpy_iterator()
    predictions_data = detailed_test_evaluation(model, test_iter, class_names=class_names, model_name="YatCNN")
    if len(class_names) <= 50:
        plot_confusion_matrix(predictions_data, "YatCNN")
    print("\n🔬 STEP 4: Model Interpretability and Robustness Analysis...")
    if run_saliency_analysis:
        from analysis import plot_saliency_maps
        plot_saliency_maps(model, test_ds, class_names)
    if run_kernel_analysis:
        from analysis import visualize_kernel_similarity
        visualize_kernel_similarity(model)
    if run_adversarial_analysis:
        from analysis import test_adversarial_robustness
        adv_test_iter = test_ds.batch(dataset_config.get('batch_size')).as_numpy_iterator()
        test_adversarial_robustness(model, adv_test_iter, class_names, epsilon=adversarial_epsilon)
    print("\n🎨 Generating t-SNE plot for the test set representations...")
    tsne_test_iter = test_ds.batch(dataset_config.get('batch_size', fallback_configs['batch_size'])).as_numpy_iterator()
    visualize_tsne(model, tsne_test_iter, class_names, title="t-SNE of Final Model Representations on Test Set")
    print("\n" + "="*80 + f"\nANALYSIS FOR {dataset_name.upper()} COMPLETE! ✅\n" + "="*80)
    # --- WANDB LOGGING ---
    for i, (train_loss, train_acc, test_loss, test_acc) in enumerate(zip(
        metrics_history['train_loss'], metrics_history['train_accuracy'],
        metrics_history['test_loss'], metrics_history['test_accuracy'])):
        log_metrics({
            'step': i,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
        })
    # Optionally log confusion matrix, t-SNE, or model artifacts here
    return {
        'model': model, 
        'metrics_history': metrics_history, 
        'predictions_data': predictions_data,
        'detailed_metrics': detailed_metrics
    } 