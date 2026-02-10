import numpy as np
import os

# Load the full datasets
supervised_embeddings = np.load("vec2vec/data/protein_embeddings.npy")
protein_labels = np.load("vec2vec/data/protein_labels.npy")
unsupervised_embeddings = np.load("vec2vec/data/unlabeled_embeddings.npy")

# Ensure the output directory exists
output_dir = "vec2vec/data"
os.makedirs(output_dir, exist_ok=True)

# Set the number of samples
num_samples = 1000

# Randomly sample 1,000 supervised embeddings and their labels
supervised_indices = np.random.choice(
    supervised_embeddings.shape[0], num_samples, replace=False
)
sampled_supervised_embeddings = supervised_embeddings[supervised_indices]
sampled_protein_labels = protein_labels[supervised_indices]

# Randomly sample 1,000 unsupervised embeddings
unsupervised_indices = np.random.choice(
    unsupervised_embeddings.shape[0], num_samples, replace=False
)
sampled_unsupervised_embeddings = unsupervised_embeddings[unsupervised_indices]

# Save the sampled datasets
np.save(
    f"{output_dir}/{num_samples}_sampled_protein_embeddings.npy",
    sampled_supervised_embeddings,
)
np.save(
    f"{output_dir}/{num_samples}_sampled_protein_labels.npy", sampled_protein_labels
)
np.save(
    f"{output_dir}/{num_samples}_sampled_unlabeled_embeddings.npy",
    sampled_unsupervised_embeddings,
)

print("Sampled datasets saved successfully!")
