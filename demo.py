import streamlit as st
import pandas as pd
import torch
import numpy as np
import os
import shutil
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from valuation import get_measurements  # Ensure this is the correct import path
import zipfile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load CLIP model and processor
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    model.eval()  # Set the model to evaluation mode
    return processor, model

processor, model = load_clip()

# Function to convert text data to CLIP embeddings in batches
def get_clip_embeddings_text(texts, processor, model, batch_size=32):
    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    with st.spinner('Embedding text data into CLIP model... this may take a while on large datasets, so try to test on smaller ones'):
        progress_bar = st.progress(0)
        for batch_idx in range(total_batches):
            batch_texts = texts[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            inputs = processor(text=batch_texts, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = model.get_text_features(**inputs)
            embeddings.append(outputs)
            progress_bar.progress((batch_idx + 1) / total_batches)
        progress_bar.empty()

    if embeddings:
        return torch.cat(embeddings, dim=0)
    else:
        return torch.empty(0)

# Function to convert image data to CLIP embeddings
def get_clip_embeddings_images(images, processor, model, batch_size=32):
    embeddings = []
    total_batches = (len(images) + batch_size - 1) // batch_size

    with st.spinner('Embedding image data into CLIP model... this may take a while on large datasets, so try to test on smaller ones'):
        progress_bar = st.progress(0)
        for batch_idx in range(total_batches):
            batch_images = images[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            if not batch_images:  # Check if the batch is empty
                continue
            try:
                # Convert all images to RGB format before processing
                batch_images = [img.convert("RGB") for img in batch_images]
                inputs = processor(images=batch_images, return_tensors='pt', padding=True)
                with torch.no_grad():
                    outputs = model.get_image_features(**inputs)
                embeddings.append(outputs)
            except Exception as e:
                st.warning(f"Failed to process a batch of images: {e}")
            progress_bar.progress((batch_idx + 1) / total_batches)
        progress_bar.empty()

    if embeddings:
        return torch.cat(embeddings, dim=0)
    else:
        return torch.empty(0)

# Function to read CSV files with fallback encoding
def read_csv_with_fallback(file):
    try:
        return pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file, encoding='ISO-8859-1')

# Function to load images from an uploaded folder
def load_images_from_folder(folder_path):
    images = []
    for root, _, files in os.walk(folder_path):  # Walk through all directories
        for file_name in files:
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')) and not file_name.startswith('._'):
                try:
                    image = Image.open(os.path.join(root, file_name))
                    image = image.convert("RGB")  # Ensure the image is in RGB mode
                    images.append(image)
                except Exception as e:
                    st.warning(f"Failed to open image {file_name}: {e}")
    return images

# Function to clear a directory
def clear_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)

# Custom CSS for button styling
st.markdown(
    """
    <style>
    .stFileUploader {
        font-size: 20px !important;
        height: 50px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state for caching
if 'seller_data' not in st.session_state:
    st.session_state['seller_data'] = {}
if 'buyer_data' not in st.session_state:
    st.session_state['buyer_data'] = None

# Streamlit app layout
st.title("Data Valuation")

# Options for pre-saved datasets
datasets = {
    "Pokemon Images": "pokemon_images.zip",
    "Stop Sign Images": "stop_signs.zip",
    "Chest X-Ray Images": "chest_xray_images.zip",
}

# Section to choose or upload a buyer dataset
st.header("Add your own buyer dataset or choose one of the following preexisting datasets:")
buyer_option = st.selectbox("Choose a dataset", ["Upload your own CSV", "Upload a folder of JPG/PNG images"] + list(datasets.keys()), key="buyer_selectbox")

# Ensure buyer data is only computed once
if st.session_state['buyer_data'] is None:
    if buyer_option == "Upload your own CSV":
        buyer_file = st.file_uploader("Upload Buyer CSV", type="csv", key="buyer_file")
        if buyer_file is not None:
            buyer_df = read_csv_with_fallback(buyer_file)
            buyer_file_name = buyer_file.name
            if not buyer_df.empty:
                buyer_texts = buyer_df.iloc[:, 0].dropna().astype(str).tolist()
                buyer_embeddings = get_clip_embeddings_text(buyer_texts, processor, model).detach().numpy()

                # Cache the results
                st.session_state['buyer_data'] = {
                    'embeddings': buyer_embeddings,
                    'file_name': buyer_file_name
                }
    elif buyer_option == "Upload a folder of JPG/PNG images":
        image_folder = st.file_uploader("Upload a ZIP file of JPG/PNG images", type="zip", key="buyer_image_folder")
        if image_folder is not None:
            with open("temp_images_buyer.zip", "wb") as f:
                f.write(image_folder.read())

            # Clear the directory before extracting new images
            clear_directory("temp_images_buyer")
            with zipfile.ZipFile("temp_images_buyer.zip", 'r') as zip_ref:
                zip_ref.extractall("temp_images_buyer")

            buyer_images = load_images_from_folder("temp_images_buyer")
            st.write(f"Found {len(buyer_images)} images in the uploaded folder.")  # Debugging info
            if not buyer_images:
                st.warning("No images found in the uploaded folder.")
            else:
                buyer_embeddings = get_clip_embeddings_images(buyer_images, processor, model).detach().numpy()

                # Cache the results
                st.session_state['buyer_data'] = {
                    'embeddings': buyer_embeddings,
                    'file_name': "Uploaded Image Folder"
                }

    else:
        selected_dataset = datasets[buyer_option]
        if selected_dataset.endswith('.csv'):
            buyer_df = read_csv_with_fallback(selected_dataset)
            buyer_file_name = buyer_option
            if not buyer_df.empty:
                buyer_texts = buyer_df.iloc[:, 0].dropna().astype(str).tolist()
                buyer_embeddings = get_clip_embeddings_text(buyer_texts, processor, model).detach().numpy()

                # Cache the results
                st.session_state['buyer_data'] = {
                    'embeddings': buyer_embeddings,
                    'file_name': buyer_file_name
                }
        elif selected_dataset.endswith('.zip'):
            # Clear the directory before extracting new images
            clear_directory("temp_images_buyer")
            with zipfile.ZipFile(selected_dataset, 'r') as zip_ref:
                zip_ref.extractall("temp_images_buyer")

            buyer_images = load_images_from_folder("temp_images_buyer")
            st.write(f"Found {len(buyer_images)} images in the selected folder.")  # Debugging info
            if not buyer_images:
                st.warning("No images found in the selected folder.")
            else:
                buyer_embeddings = get_clip_embeddings_images(buyer_images, processor, model).detach().numpy()

                # Cache the results
                st.session_state['buyer_data'] = {
                    'embeddings': buyer_embeddings,
                    'file_name': buyer_option
                }

# Function to load embeddings from .pt files
def load_embeddings_from_pt(file_path):
    return torch.load(file_path)

# Automatically process all .pt files in the 'embeddings' folder as seller datasets
seller_folder_path = 'embeddings'
if os.path.exists(seller_folder_path):
    seller_files = os.listdir(seller_folder_path)
    for seller_file in seller_files:
        seller_file_path = os.path.join(seller_folder_path, seller_file)
        if seller_file.endswith('.pt') and seller_file not in st.session_state['seller_data']:
            try:
                embeddings = load_embeddings_from_pt(seller_file_path)
                st.session_state['seller_data'][seller_file] = {
                    'embeddings': embeddings.numpy()  # Convert tensors to numpy for consistency with other parts of your code
                }
            except Exception as e:
                st.error(f"Failed to load embeddings from {seller_file_path}: {str(e)}")
else:
    st.write("No seller datasets found in the embeddings folder.")

# Load seller eigenvalues from CSV
def load_seller_eigenvalues(file_path='seller_eigenvalues.csv'):
    return pd.read_csv(file_path, index_col=0)

# Proceed with calculations if buyer data is available
if st.session_state['buyer_data']:
    # Create dictionaries from session state data
    buyer_name = st.session_state['buyer_data']['file_name']
    X_b = st.session_state['buyer_data']['embeddings']

    # Covariance matrix for buyer data
    cov_b = np.cov(X_b, rowvar=False)

    # Eigendecomposition of buyer's covariance
    eigvals_b, eigvecs_b = np.linalg.eigh(cov_b)

    # Sort by decreasing order
    index_b = np.argsort(eigvals_b)[::-1]
    sorted_eigvals_b = eigvals_b[index_b][:10]  # First 10 eigenvalues

    # Prepare a DataFrame to collect the buyer eigenvalues in the same format
    buyer_eigenvalues_df = pd.DataFrame(sorted_eigvals_b).transpose()
    buyer_eigenvalues_df.columns = [f"Eigenvalue {i+1}" for i in range(len(sorted_eigvals_b))]
    buyer_eigenvalues_df.index = [buyer_name]

    # Display the results
    st.write("Top 10 Sorted Eigenvalues (Buyer):")
    st.write(buyer_eigenvalues_df)

    # Load precomputed seller eigenvalues
    if 'seller_eigenvalues' not in st.session_state:
        st.session_state['seller_eigenvalues'] = load_seller_eigenvalues()

    # Display all seller eigenvalues
    st.write("Top 10 Sorted Eigenvalues (Sellers):")
    st.write(st.session_state['seller_eigenvalues'])

    st.header("Diversity and Relevance Metrics to Help the Buyer Choose a Seller:")
    # Number of principal components to use
    n_components = 10

    # Create tabs for each seller if there is at least one seller
    if st.session_state['seller_data']:
        seller_tabs = st.tabs(list(st.session_state['seller_data'].keys()))

        # Use shorter metric names
        measurement_labels = {
            'correlation': 'Correlation',
            'overlap': 'Overlap',
            'l2': 'L2 Distance',
            'cosine': 'Cosine Similarity',
            'difference': 'Difference',
            'volume': 'Volume',
            'vendi': 'Vendi Score',
            'dispersion': 'Dispersion'
        }

        # Dictionary to hold all measurements
        all_measurements = {key: [] for key in measurement_labels.keys()}

        for seller_name, seller_info in zip(st.session_state['seller_data'], seller_tabs):
            with seller_info:
                X_s = st.session_state['seller_data'][seller_name]['embeddings']
                seller_measurements = get_measurements(X_b, X_s, n_components=n_components)

                # Collect measurements for each seller
                for key in measurement_labels.keys():
                    all_measurements[key].append((seller_name, seller_measurements[key]))

                # Map original keys to new descriptive labels
                labeled_measurements = {measurement_labels[key]: value for key, value in seller_measurements.items()}

                # Display the results for each seller
                st.write(f"Measurements for Seller: {seller_name}")

                # Convert measurements to a DataFrame and display it
                if labeled_measurements:
                    measurements_df = pd.DataFrame.from_dict(labeled_measurements, orient='index', columns=['Value'])
                    st.table(measurements_df)

                # Display the first 10 PCA directions as a horizontal table for seller
                pca_directions_seller = seller_measurements.get('pca_directions', [])
                if pca_directions_seller:
                    directions_seller_df = pd.DataFrame(pca_directions_seller[:n_components]).transpose()
                    directions_seller_df.columns = [f"Direction {i+1}" for i in range(len(pca_directions_seller[:n_components]))]
                    st.write("First 10 PCA Directions (Seller):")
                    st.dataframe(directions_seller_df)

        # Create tabs for each measurement category
        measurement_tabs = st.tabs(list(measurement_labels.keys()))

        for key, tab in zip(measurement_labels.keys(), measurement_tabs):
            with tab:
                st.subheader(f"Comparison of {measurement_labels[key]}")
                with st.spinner(f"Generating graph for {measurement_labels[key]}..."):
                    # Extract seller names and values
                    sellers, values = zip(*all_measurements[key])

                    # Define colors for sellers
                    num_sellers = len(sellers)
                    colors = list(mcolors.TABLEAU_COLORS.values())[:num_sellers]  # Use discrete colors from Tableau colormap

                    # Plot the number line
                    plt.figure(figsize=(12, 3))
                    plt.hlines(0, min(values), max(values), colors='gray', linestyles='dashed', linewidth=1)
                    scatter = plt.scatter(values, np.zeros_like(values), c=colors, s=100, edgecolor='black')

                    plt.grid(axis='x', linestyle='--', alpha=0.5)
                    plt.yticks([])
                    plt.xlabel('Value')
                    plt.xlim(min(values) - 0.1 * (max(values) - min(values)), max(values) + 0.1 * (max(values) - min(values)))

                    # Create custom legend handles
                    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=8, label=seller) for i, seller in enumerate(sellers)]
                    legend = plt.legend(handles=handles, title="Sellers", loc="upper right", bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize='x-small')

                    # Adjust layout to prevent clipping of legend or labels
                    plt.tight_layout()
                    plt.subplots_adjust(right=0.75)  # Adjust the right margin to provide space for the legend
                    st.pyplot(plt)

    else:
        st.write("No seller datasets available.")
else:
    st.write("Please upload a buyer dataset to proceed.")

# Main logic to process datasets and display results
def process_and_display_data():
    # Prepare the plot
    with st.spinner("Generating relevance vs. diversity graph..."):
        fig, ax = plt.subplots()
        colors = list(mcolors.TABLEAU_COLORS.values())  # Using Tableau colors for distinction

        if 'buyer_data' in st.session_state and st.session_state['buyer_data']:
            buyer_data = st.session_state['buyer_data']
            X_b = buyer_data['embeddings']

            if 'seller_data' in st.session_state and st.session_state['seller_data']:
                for index, (seller_name, seller_info) in enumerate(st.session_state['seller_data'].items()):
                    X_s = seller_info['embeddings']

                    # Debugging information
                    if X_b is not None and X_s is not None:
                        try:
                            seller_measurements = get_measurements(X_b, X_s, n_components=10)
                            volume = seller_measurements.get('volume', 0)
                            overlap = seller_measurements.get('overlap', 0)

                            # Add point to the plot
                            ax.scatter(volume, overlap, color=colors[index % len(colors)], label=seller_name)

                        except Exception as e:
                            st.error(f"Error processing measurements for {seller_name}: {str(e)}")
                    else:
                        st.error("Invalid embeddings data.")
                ax.set_xlabel('Relevance (Volume)')
                ax.set_ylabel('Diversity (Overlap)')
                ax.set_title('Relevance vs. Diversity for All Sellers')
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("No seller data available.")
        else:
            st.warning("No buyer data available.")

# Automatically process and display data on load
process_and_display_data()
