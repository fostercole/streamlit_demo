import os
import pandas as pd
import torch
import numpy as np

# Function to load embeddings from .pt files
def load_embeddings_from_pt(file_path):
    return torch.load(file_path)

# Function to compute and save the first ten eigenvalues
def compute_eigenvalues_and_save(seller_folder_path='embeddings', output_csv='seller_eigenvalues.csv'):
    if not os.path.exists(seller_folder_path):
        print(f"No seller datasets found in the {seller_folder_path} folder.")
        return

    seller_files = os.listdir(seller_folder_path)
    seller_eigenvalues = []

    for seller_file in seller_files:
        if seller_file.endswith('.pt'):
            seller_file_path = os.path.join(seller_folder_path, seller_file)
            try:
                embeddings = load_embeddings_from_pt(seller_file_path).numpy()
                cov_s = np.cov(embeddings, rowvar=False)

                # Eigendecomposition of seller's covariance
                eigvals_s, eigvecs_s = np.linalg.eigh(cov_s)

                # Sort by decreasing order
                index_s = np.argsort(eigvals_s)[::-1]
                sorted_eigvals_s = eigvals_s[index_s][:10]  # First 10 eigenvalues
                seller_eigenvalues.append((seller_file, *sorted_eigvals_s))

            except Exception as e:
                print(f"Failed to load or process embeddings from {seller_file_path}: {str(e)}")
                continue

    # Save eigenvalues to CSV
    df = pd.DataFrame(seller_eigenvalues, columns=['Seller'] + [f'Eigenvalue {i+1}' for i in range(10)])
    df.to_csv(output_csv, index=False)
    print(f"Eigenvalues have been saved to {output_csv}.")

# Run the computation and save process
compute_eigenvalues_and_save()
