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
import optax

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

# --- Saliency Map Generation ---
@partial(jax.jit, static_argnames=['model'])
def get_saliency_map(model, image, label):
    def model_output_for_class(img):
        logits = model(img[None, ...], training=False)
        return logits[0, label]
    grad_fn = jax.grad(model_output_for_class)
    grads = grad_fn(image)
    return jnp.abs(grads)

def plot_saliency_maps(model, test_ds, class_names: list, num_images: int = 5):
    print("\n🗺️  Generating Saliency Maps...")
    vis_batch = next(test_ds.batch(num_images).as_numpy_iterator())
    images, labels = vis_batch['image'], vis_batch['label']
    logits = model(images, training=False)
    predicted_labels = jnp.argmax(logits, axis=1)
    plt.figure(figsize=(num_images * 4, 8))
    for i in range(num_images):
        saliency = get_saliency_map(model, images[i], predicted_labels[i])
        saliency_map = np.array(jnp.max(saliency, axis=-1))
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i])
        title = (f"Pred: {class_names[predicted_labels[i]]}\n" f"True: {class_names[labels[i]]}")
        ax.set_title(title, color="green" if predicted_labels[i] == labels[i] else "red")
        ax.axis('off')
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(images[i])
        plt.imshow(saliency_map, cmap='hot', alpha=0.6)
        ax.set_title("Saliency Map")
        ax.axis('off')
    plt.suptitle("Saliency Map Analysis", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- Kernel Similarity ---
def visualize_kernel_similarity(model):
    print("\n🕸️  Analyzing Kernel Similarity...")
    from models import YatConvBlock, YatConv
    conv_layers = {}
    for block_name, block in vars(model).items():
        if isinstance(block, YatConvBlock):
            for layer_name, layer in vars(block).items():
                if hasattr(layer, 'kernel'):
                    conv_layers[f"{block_name}_{layer_name}"] = layer
    for name, layer in conv_layers.items():
        kernels = np.array(layer.kernel.value)
        num_filters = kernels.shape[3]
        kernels_flat = kernels.reshape(-1, num_filters).T
        similarity_matrix = cosine_similarity(kernels_flat)
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, cmap='vlag', vmin=-1, vmax=1)
        plt.title(f"Kernel Cosine Similarity for Layer: {name}", fontweight='bold')
        plt.xlabel("Kernel Index")
        plt.ylabel("Kernel Index")
        plt.show()

# --- Adversarial Attack (FGSM) ---
@partial(jax.jit, static_argnames=['model'])
def generate_adversarial_example_fgsm(model, image, label, epsilon):
    def loss_for_grad(img):
        logits = model(img[None, ...], training=False)
        return optax.softmax_cross_entropy_with_integer_labels(logits, label[None, ...]).mean()
    grad_fn = jax.grad(loss_for_grad)
    grads = grad_fn(image)
    perturbation = epsilon * jnp.sign(grads)
    adversarial_image = image + perturbation
    adversarial_image = jnp.clip(adversarial_image, 0, 1)
    return adversarial_image

def test_adversarial_robustness(model, test_ds_iter, class_names: list, epsilon: float, num_vis_images: int = 5):
    print(f"\n🛡️  Testing Adversarial Robustness with Epsilon = {epsilon}...")
    total_correct = 0
    total_images = 0
    vis_images, vis_adv_images, vis_labels, vis_orig_preds, vis_adv_preds = [], [], [], [], []
    for batch in tqdm(test_ds_iter, desc="Adversarial Evaluation"):
        images, labels = batch['image'], batch['label']
        adv_images = jax.vmap(generate_adversarial_example_fgsm, in_axes=(None, 0, 0, None))(
            model, images, labels, epsilon
        )
        adv_logits = model(adv_images, training=False)
        adv_preds = jnp.argmax(adv_logits, axis=1)
        total_correct += jnp.sum(adv_preds == labels)
        total_images += len(labels)
        if len(vis_images) < num_vis_images:
            orig_logits = model(images, training=False)
            orig_preds = jnp.argmax(orig_logits, axis=1)
            num_to_add = min(num_vis_images - len(vis_images), len(images))
            vis_images.extend(images[:num_to_add])
            vis_adv_images.extend(adv_images[:num_to_add])
            vis_labels.extend(labels[:num_to_add])
            vis_orig_preds.extend(orig_preds[:num_to_add])
            vis_adv_preds.extend(adv_preds[:num_to_add])
    accuracy = (total_correct / total_images) * 100
    print(f"\nAdversarial Accuracy: {accuracy:.2f}%")
    plt.figure(figsize=(num_vis_images * 4, 12))
    for i in range(num_vis_images):
        ax = plt.subplot(3, num_vis_images, i + 1)
        plt.imshow(vis_images[i])
        title = (f"Pred: {class_names[vis_orig_preds[i]]}\n" f"True: {class_names[vis_labels[i]]}")
        ax.set_title(title, color="green" if vis_orig_preds[i] == vis_labels[i] else "red")
        ax.axis('off')
        ax = plt.subplot(3, num_vis_images, i + 1 + num_vis_images)
        perturbation = vis_adv_images[i] - vis_images[i]
        plt.imshow((perturbation - perturbation.min()) / (perturbation.max() - perturbation.min()))
        ax.set_title(f"Perturbation (ε={epsilon})")
        ax.axis('off')
        ax = plt.subplot(3, num_vis_images, i + 1 + 2 * num_vis_images)
        plt.imshow(vis_adv_images[i])
        title = (f"Adv Pred: {class_names[vis_adv_preds[i]]}\n" f"True: {class_names[vis_labels[i]]}")
        ax.set_title(title, color="green" if vis_adv_preds[i] == vis_labels[i] else "red")
        ax.axis('off')
    plt.suptitle("Adversarial Attack Analysis (FGSM)", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show() 