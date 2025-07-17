# ==============================================================================
# analysis.py -- Analysis and Visualization Functions
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, pairwise
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import jax
import jax.numpy as jnp
from functools import partial
from flax import nnx

def plot_training_curves(history, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'Fine-Tuning Curves for {model_name}', fontsize=16, fontweight='bold')
    steps = range(len(history['train_loss']))
    ax1.plot(steps, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(steps, history['test_loss'], 'r--', label='Test Loss', linewidth=2)
    ax1.set_title('Loss')
    ax1.set_xlabel('Evaluation Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(steps, history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(steps, history['test_accuracy'], 'r--', label='Test Accuracy', linewidth=2)
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Evaluation Steps')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_final_metrics(history, model_name):
    print(f"\n📊 FINAL METRICS FOR {model_name}" + "\n" + "=" * 40)
    final_metrics = {metric: hist[-1] for metric, hist in history.items() if hist}
    if not final_metrics:
        print("No metrics recorded.")
        return
    print(f"{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    for metric, value in final_metrics.items():
        print(f"{metric:<20} {value:<15.4f}")
    print("\n🏆 SUMMARY:")
    print(f"    Final Test Accuracy: {final_metrics.get('test_accuracy', 0):.4f}")

def detailed_test_evaluation(model, test_ds_iter, class_names: list[str], model_name: str):
    print(f"\n🔬 Running detailed test evaluation for {model_name}...")
    num_classes = len(class_names)
    predictions, true_labels = [], []
    for batch in tqdm(test_ds_iter, desc="Detailed Evaluation"):
        preds = jnp.argmax(model(batch['image'], training=False), axis=1)
        predictions.extend(preds.tolist())
        true_labels.extend(batch['label'].tolist())
    predictions, true_labels = np.array(predictions), np.array(true_labels)
    print("\n🎯 PER-CLASS ACCURACY" + "\n" + "=" * 50)
    print(f"{'Class':<15} {'Accuracy':<10} {'Sample Count':<12}")
    print("-" * 50)
    for i in range(num_classes):
        mask = true_labels == i
        if np.sum(mask) > 0:
            acc = np.mean(predictions[mask] == true_labels[mask])
            print(f"{class_names[i]:<15} {acc:<10.4f} {np.sum(mask):<12}")
    return {'predictions': predictions, 'true_labels': true_labels, 'class_names': class_names}

def plot_confusion_matrix(predictions_data, model_name):
    cm = confusion_matrix(predictions_data['true_labels'], predictions_data['predictions'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=predictions_data['class_names'], yticklabels=predictions_data['class_names'])
    plt.title(f'{model_name} - Confusion Matrix', fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

def visualize_tsne(encoder, dataset_iter, class_names: list[str], title: str, num_samples: int = 1000):
    print(f"\n🎨 Generating t-SNE plot: {title}...")
    all_embeddings = []
    all_labels = []
    batch_size = 32
    total_iters = int(np.ceil(num_samples / batch_size))
    for batch in tqdm(dataset_iter, desc="Extracting embeddings for t-SNE", total=total_iters):
        embeddings = encoder(batch['image'], training=False, return_activations_for_layer='representation')
        all_embeddings.append(np.array(embeddings))
        all_labels.append(np.array(batch['label']))
        if len(np.concatenate(all_labels)) >= num_samples:
            break
    all_embeddings = np.concatenate(all_embeddings, axis=0)[:num_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:num_samples]
    if all_embeddings.shape[0] < 2:
        print("Not enough samples for t-SNE plot.")
        return
    output_neurons = np.array(encoder.out_linear.kernel.value).T
    num_neurons = output_neurons.shape[0]
    combined_data = np.vstack((all_embeddings, output_neurons))
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data)
    perplexity = min(30, combined_data_scaled.shape[0] - 1)
    tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, n_iter=300, random_state=42)
    combined_tsne_results = tsne.fit_transform(combined_data_scaled)
    image_tsne_results = combined_tsne_results[:-num_neurons]
    neuron_tsne_results = combined_tsne_results[-num_neurons:]
    plt.figure(figsize=(14, 12))
    cmap = plt.cm.get_cmap("jet", len(class_names))
    scatter = plt.scatter(image_tsne_results[:,0], image_tsne_results[:,1], c=all_labels, cmap=cmap, alpha=0.6)
    plt.scatter(neuron_tsne_results[:, 0], neuron_tsne_results[:, 1], marker='*', c=range(len(class_names)), cmap=cmap, s=800, edgecolor='black', linewidth=1.5)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    handles, _ = scatter.legend_elements(prop="colors", alpha=0.6)
    neuron_handle = plt.Line2D([0], [0], marker='*', color='w', label='Class Neuron', markerfacecolor='grey', markeredgecolor='k', markersize=20)
    plt.legend(handles=handles + [neuron_handle], labels=class_names + ['Class Neuron'], title="Classes")
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_reconstructions(autoencoder, dataset_iter, title: str, num_images: int = 8):
    print(f"\n🖼️  Generating image reconstructions: {title}...")
    try:
        batch = next(dataset_iter)
    except StopIteration:
        print("Could not get a batch to visualize reconstructions.")
        return
    original_images = batch['original_image'][:num_images]
    augmented_images = batch['augmented_image'][:num_images]
    reconstructed_images = autoencoder(augmented_images, training=False)
    original_images = np.array(original_images)
    augmented_images = np.array(augmented_images)
    reconstructed_images = np.array(reconstructed_images)
    n = min(num_images, len(original_images))
    plt.figure(figsize=(20, 6))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(augmented_images[i])
        plt.title("Augmented Input")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(reconstructed_images[i])
        plt.title("Reconstructed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(original_images[i])
        plt.title("Original Target")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- Saliency Map Generation, Kernel Similarity, Adversarial Attack, and Robustness Analysis functions should also be moved here from main.py.
# For brevity, only the function signatures and comments are included here. Move the full function bodies from main.py and update imports as needed. 